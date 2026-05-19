import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
from tqdm import tqdm
import pandas as pd
import subprocess
import os
import numpy as np
import scipy.linalg


def parse_format_spec(format_name):
    fmt = format_name.lower()
    if fmt == "e2m1":
        return {"ebits": 2, "mbits": 1, "bias": 1, "has_inf_nan": False, "has_sign": True}
    elif fmt == "e1m2":
        return {"ebits": 1, "mbits": 2, "bias": -1, "has_inf_nan": False, "has_sign": True}
    elif fmt == "ue5m3":
        return {"ebits": 5, "mbits": 3, "bias": 15, "has_inf_nan": True, "has_sign": False}
    elif fmt == "e8m0":
        return {"ebits": 8, "mbits": 0, "bias": 127, "has_inf_nan": True, "has_sign": False}
    elif fmt == "e4m3":
        return {"ebits": 4, "mbits": 3, "bias": 7, "has_inf_nan": True, "has_sign": True}
    elif fmt == "bf16":
        return {"ebits": 8, "mbits": 7, "bias": 127, "has_inf_nan": True, "has_sign": True}
    elif fmt in ("fp16", "f16"):
        return {"ebits": 5, "mbits": 10, "bias": 15, "has_inf_nan": True, "has_sign": True}
    
    has_sign = True
    if fmt.startswith("u"):
        has_sign = False
        fmt = fmt[1:]
        
    if fmt.startswith("e") and "m" in fmt:
        parts = fmt[1:].split("m")
        ebits = int(parts[0])
        mbits = int(parts[1])
        bias = (2 ** (ebits - 1)) - 1 if ebits > 0 else 0
        has_inf_nan = True
        return {"ebits": ebits, "mbits": mbits, "bias": bias, "has_inf_nan": has_inf_nan, "has_sign": has_sign}
        
    raise ValueError(f"Unknown format: {format_name}")

def generate_float_grid(ebits, mbits, bias, has_inf_nan=True, device="cuda"):
    grid = [0.0]
    if mbits > 0:
        sub_scale = 2.0 ** (1.0 - bias)
        for m in range(2 ** mbits):
            val = (m / (2 ** mbits)) * sub_scale
            if val > 0.0:
                grid.append(val)
    
    e_max = (2 ** ebits) - 2 if has_inf_nan else (2 ** ebits) - 1
    for e in range(1, e_max + 1):
        scale = 2.0 ** (e - bias)
        if mbits > 0:
            for m in range(2 ** mbits):
                val = (1.0 + m / (2 ** mbits)) * scale
                grid.append(val)
        else:
            grid.append(scale)
            
    grid = sorted(list(set(grid)))
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)
    
    if mbits == 0:
        th_t = (grid_t[:-1] + grid_t[1:]) / 2.0
        if len(grid_t) > 2:
            th_t[1:] = torch.sqrt(grid_t[1:-1] * grid_t[2:])
    else:
        th_t = (grid_t[:-1] + grid_t[1:]) / 2.0
        
    max_rep = grid_t[-1].item()
    return grid_t, th_t, max_rep

def get_element_format_grid(format_name, device="cuda"):
    fmt = format_name.lower()
    if fmt == "int8":
        grid = torch.arange(128, dtype=torch.float32, device=device)
        th = (grid[:-1] + grid[1:]) / 2.0
        max_rep = 127.0
        return grid, th, max_rep
    elif fmt == "int4":
        grid = torch.arange(8, dtype=torch.float32, device=device)
        th = (grid[:-1] + grid[1:]) / 2.0
        max_rep = 7.0
        return grid, th, max_rep

    spec = parse_format_spec(format_name)
    return generate_float_grid(
        ebits=spec["ebits"],
        mbits=spec["mbits"],
        bias=spec["bias"],
        has_inf_nan=spec["has_inf_nan"],
        device=device
    )

def get_ue5m3_grid():
    grid, th, _ = get_element_format_grid("ue5m3")
    return grid, th

def quantize_scale_simulate(val, format="e4m3", prevent_zero=True, rounding="round"):
    if format == "e4m3":
        val_clipped = torch.clamp(val, max=448.0)
        quant = val_clipped.to(torch.float8_e4m3fn).to(val.dtype)
        min_scale = 2**-9
        if prevent_zero:
            quant = torch.where(quant == 0.0, min_scale, quant)
        return quant.reshape(val.shape)
    elif format == "bf16":
        quant = val.to(torch.bfloat16).to(val.dtype)
        if prevent_zero:
            # min positive normal for bf16 is 2^-126
            min_scale = 2**-126
            quant = torch.where(quant == 0.0, torch.tensor(min_scale, dtype=quant.dtype, device=quant.device), quant)
        return quant.reshape(val.shape)
    elif format in ("fp16", "f16"):
        quant = val.to(torch.float16).to(val.dtype)
        if prevent_zero:
            # min positive normal for fp16 is 2^-14
            min_scale = 2**-14
            quant = torch.where(quant == 0.0, torch.tensor(min_scale, dtype=quant.dtype, device=quant.device), quant)
        return quant.reshape(val.shape)

        
    spec = parse_format_spec(format)
    grid, th, max_rep = generate_float_grid(
        ebits=spec["ebits"],
        mbits=spec["mbits"],
        bias=spec["bias"],
        has_inf_nan=spec["has_inf_nan"],
        device=val.device
    )
    
    min_scale = grid[1].item() if len(grid) > 1 else 0.0
    val_clipped = torch.clamp(val, max=max_rep)
    
    if rounding == "round":
        idx = torch.bucketize(val_clipped, th)
        quant = grid[idx]
    elif rounding == "ceil":
        idx = torch.bucketize(val_clipped, grid, right=False)
        idx = torch.clamp(idx, max=len(grid) - 1)
        quant = grid[idx]
    elif rounding == "floor":
        idx = torch.bucketize(val_clipped, grid, right=True)
        idx = torch.clamp(idx - 1, min=0)
        quant = grid[idx]
    else:
        raise ValueError(f"Unsupported rounding mode: {rounding}")
        
    if prevent_zero:
        quant = torch.where(quant == 0.0, min_scale, quant)
        
    return quant.reshape(val.shape)

def quantize_fp8_simulate(val, prevent_zero=True, format="e4m3"):
    return quantize_scale_simulate(val, format=format, prevent_zero=prevent_zero, rounding="round")

# -------------------------------------------------------------------------
# MXFP4 and Hadamard Transform Logic
# -------------------------------------------------------------------------
def get_hadamard_matrix(n, device="cuda", dtype=torch.float32):
    if n < 1 or (n & (n - 1) != 0):
        raise ValueError("n must be a power of 2.")
    had = scipy.linalg.hadamard(n)
    return torch.tensor(had, device=device, dtype=dtype)

def apply_hadamard_torch(mat, had_mat, is_lhs, apply_scale=True):
    h = had_mat.shape[0]
    init_shape = mat.shape
    if is_lhs:
        # Mat is (..., K). Reshape to 2D: (N, K)
        mat_2d = mat.reshape(-1, init_shape[-1])
        d0, d1 = mat_2d.shape
        mat_reshaped = mat_2d.reshape(d0, d1 // h, h)
        res_reshaped = torch.matmul(mat_reshaped, had_mat)
        res = res_reshaped.reshape(init_shape)
    else:
        # Mat is (K, Out)
        d0, d1 = mat.shape
        mat_reshaped = mat.reshape(d0 // h, h, d1)
        res_reshaped = torch.matmul(had_mat.unsqueeze(0), mat_reshaped)
        res = res_reshaped.reshape(init_shape)
        
    if apply_scale:
        res = res / np.sqrt(h)
    return res

def had_mod_torch(x, w, had_size=256, seed=42):
    device = x.device
    dtype = x.dtype
    had_mat = get_hadamard_matrix(had_size, device=device, dtype=torch.float32)
    
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        
        d1 = torch.randint(0, 2, (had_size,), generator=g, device=device).float() * 2.0 - 1.0
        d2 = torch.randint(0, 2, (had_size,), generator=g, device=device).float() * 2.0 - 1.0
        
        d1_mat = torch.diag(d1)
        d2_mat = torch.diag(d2)
        had_mat = d1_mat @ had_mat @ d2_mat
        
    x_had = apply_hadamard_torch(x.float(), had_mat, is_lhs=True, apply_scale=True).to(dtype)
    w_had = apply_hadamard_torch(w.float(), had_mat, is_lhs=True, apply_scale=True).to(dtype)
    
    return x_had, w_had

def mse_select_quant(x, scale_a, scale_b, snap_fn):
    # Safe division for scale_a
    safe_scale_a = torch.where(scale_a != 0.0, scale_a, torch.ones_like(scale_a))
    scaled_a = x / safe_scale_a
    scaled_a = torch.where(scale_a != 0.0, scaled_a, torch.zeros_like(scaled_a))
    quant_a = snap_fn(scaled_a)
    dequant_a = quant_a * scale_a
    
    # Safe division for scale_b
    safe_scale_b = torch.where(scale_b != 0.0, scale_b, torch.ones_like(scale_b))
    scaled_b = x / safe_scale_b
    scaled_b = torch.where(scale_b != 0.0, scaled_b, torch.zeros_like(scaled_b))
    quant_b = snap_fn(scaled_b)
    dequant_b = quant_b * scale_b
    
    mse_a = torch.mean((x - dequant_a)**2, dim=-1, keepdim=True)
    mse_b = torch.mean((x - dequant_b)**2, dim=-1, keepdim=True)
    
    use_a = mse_a < mse_b
    dequant = torch.where(use_a, dequant_a, dequant_b)
    quant = torch.where(use_a, quant_a, quant_b)
    scale = torch.where(use_a, scale_a, scale_b)
    
    return dequant, quant, scale, use_a

def quantize_mx_torch(x, block_size, elem_format="e2m1", scale_format="e4m3", prevent_zero=True, four_over_six=False, use_hierarchical=False):
    init_shape = x.shape
    
    grid, th, max_rep = get_element_format_grid(elem_format, device=x.device)
    
    if use_hierarchical:
        axes = list(range(len(x.shape)))
        axes.remove(0)
        max_channel = torch.amax(torch.abs(x), dim=axes, keepdim=True)
        
        if scale_format == "e4m3":
            max_scale_val = 448.0
        else:
            spec = parse_format_spec(scale_format)
            _, _, max_scale_val = generate_float_grid(
                ebits=spec["ebits"],
                mbits=spec["mbits"],
                bias=spec["bias"],
                has_inf_nan=spec["has_inf_nan"],
                device=x.device
            )
            
        channel_scale = max_channel / max_scale_val
        channel_scale = torch.where(channel_scale != 0, channel_scale, torch.ones_like(channel_scale))
        x_norm = x / channel_scale
    else:
        x_norm = x
        
    x_reshaped = x_norm.reshape(-1, block_size)
    
    def snap_fp4(val):
        clipped = torch.clamp(val, min=-max_rep, max=max_rep)
        abs_clipped = torch.abs(clipped)
        idx = torch.bucketize(abs_clipped, th)
        quant = grid[idx]
        return quant * torch.sign(clipped)
        
    if not four_over_six:
        max_abs = torch.max(torch.abs(x_reshaped), dim=-1, keepdim=True).values
        max_abs = torch.clamp(max_abs, min=1e-7)
        
        raw_scale = max_abs / max_rep
        scaling_factor = quantize_scale_simulate(raw_scale, format=scale_format, prevent_zero=prevent_zero, rounding="ceil" if scale_format == "e8m0" else "round")
        
        # Safe division to mimic JAX behavior without NaN propagation
        safe_scale = torch.where(scaling_factor != 0.0, scaling_factor, torch.ones_like(scaling_factor))
        scaled = x_reshaped / safe_scale
        scaled = torch.where(scaling_factor != 0.0, scaled, torch.zeros_like(scaled))
        
        quant = snap_fp4(scaled)
        dequant = quant * scaling_factor
        use_4 = torch.zeros_like(scaling_factor, dtype=torch.bool)
    else:
        max_abs = torch.max(torch.abs(x_reshaped), dim=-1, keepdim=True).values
        max_abs = torch.clamp(max_abs, min=1e-7)
        
        if scale_format == "e4m3":
            is_exponential = False
        else:
            spec = parse_format_spec(scale_format)
            is_exponential = (spec["mbits"] == 0)
            
        if is_exponential:
            raw_scale = max_abs / max_rep
            scale_4 = quantize_scale_simulate(raw_scale, format=scale_format, prevent_zero=prevent_zero, rounding="floor")
            scale_6 = quantize_scale_simulate(raw_scale, format=scale_format, prevent_zero=prevent_zero, rounding="ceil")
        else:
            raw_scale_4 = (max_abs / max_rep) * 1.5
            raw_scale_6 = max_abs / max_rep
            scale_4 = quantize_scale_simulate(raw_scale_4, format=scale_format, prevent_zero=prevent_zero)
            scale_6 = quantize_scale_simulate(raw_scale_6, format=scale_format, prevent_zero=prevent_zero)
            
        dequant, quant, scaling_factor, use_4 = mse_select_quant(x_reshaped, scale_4, scale_6, snap_fp4)
        
    dequant = dequant.reshape(init_shape)
    if use_hierarchical:
        dequant = dequant * channel_scale
        
    return dequant.to(x.dtype), quant.reshape(init_shape), scaling_factor, use_4

def FP4_quant_torch(x, block_size, prevent_zero=True, four_over_six=False, use_hierarchical=False, scale_format="e4m3", format="e2m1"):
    return quantize_mx_torch(x, block_size, elem_format=format, scale_format=scale_format, prevent_zero=prevent_zero, four_over_six=four_over_six, use_hierarchical=use_hierarchical)

class TorchMXLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, block_size=32, prevent_zero=True, four_over_six=False,
                 elem_format="e2m1", scale_format="e4m3", use_hierarchical=False, hadamard_size=0, hadamard_seed=42):
        super().__init__(in_features, out_features, bias)
        self.block_size = block_size
        self.prevent_zero = prevent_zero
        self.four_over_six = four_over_six
        self.use_hierarchical = use_hierarchical
        self.elem_format = elem_format
        self.scale_format = scale_format
            
        self.hadamard_size = hadamard_size
        self.hadamard_seed = hadamard_seed
        
    def forward(self, input):
        w = self.weight
        x = input
        
        if self.hadamard_size > 0:
            x_rot, w_rot = had_mod_torch(x, w, had_size=self.hadamard_size, seed=self.hadamard_seed)
        else:
            x_rot, w_rot = x, w
            
        q_weight, _, _, _ = quantize_mx_torch(w_rot, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical)
        q_input, _, _, _ = quantize_mx_torch(x_rot, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical)
        
        return F.linear(q_input, q_weight, self.bias)

def run_eval(model_id, block_size=None, prevent_zero=True, four_over_six=False,
             elem_format="e2m1", scale_format="e4m3", num_steps=None, use_hierarchical=False,
             hadamard_size=0, hadamard_seed=42):
    
    print(f"Evaluating {model_id} with block size {block_size}, prevent_zero={prevent_zero}, four_over_six={four_over_six}, elem_format={elem_format}, scale_format={scale_format}, num_steps={num_steps}, use_hierarchical={use_hierarchical}, hadamard_size={hadamard_size}, hadamard_seed={hadamard_seed}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
    
    if block_size is not None:
        head_name = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                head_name = name
        print(f"Identified head: {head_name}")
        
        for name, module in model.named_modules():
            if name == head_name:
                continue
            if "attn" in name:
                continue
            if isinstance(module, torch.nn.Linear):
                idx = name.rfind(".")
                if idx == -1:
                    idx = 0
                father_name = name[:idx]
                father_module = model
                if father_name:
                     for part in father_name.split("."):
                          father_module = getattr(father_module, part)
                
                idx = idx + 1 if idx != 0 else idx
                new_m = TorchMXLinear(
                    module.in_features, module.out_features, module.bias is not None,
                    block_size=block_size, prevent_zero=prevent_zero, four_over_six=four_over_six,
                    elem_format=elem_format, scale_format=scale_format,
                    use_hierarchical=use_hierarchical, hadamard_size=hadamard_size, hadamard_seed=hadamard_seed
                )
                new_m.weight.data = module.weight.data
                new_m.bias = module.bias
                print(f"Replacing layer: {name}")
                setattr(father_module, name[idx:], new_m)
                
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(testdata["text"])
    encodings = tokenizer(text, return_tensors="pt")
    
    seq_len = 2048
    stride = 2048
    
    nlls = []
    
    loop = range(0, encodings.input_ids.size(1), stride)
    if num_steps is not None:
        print(f"Limiting evaluation to {num_steps} steps.")
        loop = zip(loop, range(num_steps))
        
    for item in tqdm(loop):
        if num_steps is not None:
            i, _ = item
        else:
            i = item
            
        begin_loc = i
        end_loc = min(i + seq_len, encodings.input_ids.size(1))
        trg_len = end_loc - begin_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            
        nlls.append(neg_log_likelihood)
        
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity: {ppl.item()}")
    
    del model
    torch.cuda.empty_cache()
    
    return ppl.item()

def update_csv_and_readme(model_id, bs, ppl, base_ppl, option="nvfp4", csv_suffix=""):
    disp_name = model_id
    for k, v in {"granite": "Granite", "llama": "Llama", "qwen": "Qwen", "deepseek": "DeepSeek"}.items():
        if k in model_id.lower():
            disp_name = v
            break
            
    path_csv = f"results_{disp_name.lower()}{csv_suffix}.csv"
    
    if os.path.exists(path_csv):
        df = pd.read_csv(path_csv, index_col=0)
    else:
        columns = ["Baseline", "BS=4", "BS=8", "BS=16", "BS=32", "BS=64", "BS=128", "BS=256"]
        df = pd.DataFrame(columns=columns)
        df.index.name = "Model_Config"
        
    row_name = f"{disp_name}_{option}"
    
    if bs is None:
        df.loc[row_name, "Baseline"] = ppl
    else:
        gap = ppl - base_ppl
        df.loc[row_name, f"BS={bs}"] = gap
        
    df.to_csv(path_csv)
    
    if os.environ.get("SKIP_GIT", "false").lower() != "true":
        subprocess.run(["git", "add", path_csv])
        subprocess.run(["git", "commit", "-m", f"Update results for {row_name} BS={bs}"])
        subprocess.run(["git", "pull"])
        subprocess.run(["git", "push"])


if __name__ == "__main__":
    model_id = sys.argv[1]
    
    def read_base_from_csv(model_id):
        path_csv = "results.csv"
        if os.path.exists(path_csv):
            df = pd.read_csv(path_csv, index_col=0)
            disp_name = model_id
            for k, v in {"granite": "Granite", "llama": "Llama", "qwen": "Qwen", "deepseek": "DeepSeek"}.items():
                if k in model_id.lower():
                    disp_name = v
                    break
            
            # Search for any row starting with disp_name and having non-empty Baseline
            for idx in df.index:
                if idx.startswith(disp_name) and not pd.isna(df.loc[idx, "Baseline"]):
                    return float(df.loc[idx, "Baseline"])
        return None
        
    if len(sys.argv) > 2:
        if sys.argv[2].lower() == "none" or sys.argv[2] == "":
            block_sizes = [None]
        else:
            block_sizes = [int(x) for x in sys.argv[2].split(",")]
            
        prevent_zero = True
        four_over_six = False
        scale_format = "e4m3"
        use_hierarchical = False
        num_steps = None
        csv_suffix = ""
        format = "e2m1"
        hadamard_size = 0
        hadamard_seed = 42
        
        if len(sys.argv) > 3:
            prevent_zero = sys.argv[3].lower() == "true"
        if len(sys.argv) > 4:
            four_over_six = sys.argv[4].lower() == "true"
        if len(sys.argv) > 5:
            val = sys.argv[5]
            if val.lower() == "true":
                scale_format = "ue5m3"
            elif val.lower() == "false":
                scale_format = "e4m3"
            else:
                scale_format = val
        if len(sys.argv) > 6:
            use_hierarchical = sys.argv[6].lower() == "true"
        if len(sys.argv) > 7 and sys.argv[7].lower() != "none":
            num_steps = int(sys.argv[7])
        if len(sys.argv) > 8:
            csv_suffix = sys.argv[8]
        if len(sys.argv) > 9:
            if sys.argv[9].lower() == "true":
                scale_format = "e8m0"
        if len(sys.argv) > 10:
            format = sys.argv[10]
        if len(sys.argv) > 11 and sys.argv[11].lower() != "none":
            hadamard_size = int(sys.argv[11])
        if len(sys.argv) > 12 and sys.argv[12].lower() != "none":
            hadamard_seed = int(sys.argv[12])
            
        elem_format = format
            
        if scale_format == "e8m0":
            option = f"mxfp4 ({elem_format})"
            if four_over_six:
                option += " + 4o6"
        elif elem_format in ("int8", "int4"):
            option = f"{elem_format}"
            if four_over_six:
                option += " + 4o6"
            if prevent_zero:
                option += " + PZ"
        else:
            option = f"{scale_format}"
            if four_over_six:
                option += " + 4o6"
            if prevent_zero:
                option += " + PZ"
            
        if use_hierarchical:
            option += " + H"
            
        if hadamard_size > 0:
            option += f" + RH{hadamard_size}"
            
        if block_sizes == [None]:
            base_ppl = None
        else:
            base_ppl = read_base_from_csv(model_id)
        
        for bs in block_sizes:
            ppl = run_eval(
                model_id, bs, prevent_zero=prevent_zero, four_over_six=four_over_six,
                elem_format=elem_format, scale_format=scale_format, num_steps=num_steps,
                use_hierarchical=use_hierarchical, hadamard_size=hadamard_size, hadamard_seed=hadamard_seed
            )
            if base_ppl is None and bs is not None:
                print(f"Baseline not found for {model_id}. Please run it first.")
                continue
            update_csv_and_readme(model_id, bs, ppl, base_ppl, option=option, csv_suffix=csv_suffix)
    else:
        ppl = run_eval(model_id, None)
        update_csv_and_readme(model_id, None, ppl, None)
