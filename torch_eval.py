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

_UE5M3_GRID = None
_UE5M3_TH = None

def get_ue5m3_grid():
    global _UE5M3_GRID, _UE5M3_TH
    if _UE5M3_GRID is None:
        grid = []
        # Subnormals (e=0)
        for m in range(8):
            val = (m / 8.0) * (2**-14)
            grid.append(val)
            
        # Normals (e=1..30)
        for e in range(1, 31):
            for m in range(8):
                val = (1.0 + m / 8.0) * (2**(e - 15))
                grid.append(val)
                
        _UE5M3_GRID = torch.tensor(sorted(list(set(grid))), dtype=torch.float32).cuda()
        _UE5M3_TH = (_UE5M3_GRID[:-1] + _UE5M3_GRID[1:]) / 2
    return _UE5M3_GRID, _UE5M3_TH

def quantize_fp8_simulate(val, prevent_zero=True, use_ue5m3=False):
    if not use_ue5m3:
        # Clamp to max value of float8_e4m3fn to prevent overflow to NaN
        val_clipped = torch.clamp(val, max=448.0)
        quant = val_clipped.to(torch.float8_e4m3fn).to(val.dtype)
    else:
        grid, th = get_ue5m3_grid()
        # Use bucketize for O(log K) search instead of O(K) distance calculation!
        idx = torch.bucketize(val, th)
        quant = grid[idx]
        
    if prevent_zero:
        min_scale = 2**-9
        quant = torch.where(quant == 0.0, min_scale, quant)
    
    return quant.reshape(val.shape)

# -------------------------------------------------------------------------
# Quantization Logic (Pure PyTorch Simulation)
# -------------------------------------------------------------------------
def FP4_quant_torch(x, block_size, prevent_zero=True, four_over_six=False, use_ue5m3=False):
    init_shape = x.shape
    x_reshaped = x.reshape(-1, block_size)
    
    max_val = torch.max(torch.abs(x_reshaped), dim=-1, keepdim=True).values
    
    def quantize_fp4(abs_clipped, sign):
        th = torch.tensor([0.250125, 0.749765, 1.250495, 1.749515, 2.500990, 3.499030, 5.001970], dtype=torch.float32).cuda()
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32).cuda()
        idx = torch.bucketize(abs_clipped, th)
        quant = grid[idx]
        return quant * sign
        
    if not four_over_six:
        raw_scale = max_val / 6.0
        scaling_factor = quantize_fp8_simulate(raw_scale, prevent_zero=prevent_zero, use_ue5m3=use_ue5m3)
        
        # Safe division to mimic JAX behavior without NaN propagation
        safe_scale = torch.where(scaling_factor != 0.0, scaling_factor, torch.ones_like(scaling_factor))
        scaled = x_reshaped / safe_scale
        scaled = torch.where(scaling_factor != 0.0, scaled, torch.zeros_like(scaled))
        
        clipped = torch.clamp(scaled, min=-6.0, max=6.0)
        
        sign = torch.sign(clipped)
        abs_clipped = torch.abs(clipped)
        
        quant = quantize_fp4(abs_clipped, sign)
        dequant = quant * scaling_factor
        use_4 = torch.zeros_like(scaling_factor, dtype=torch.bool)
    else:
        raw_scale_4 = (max_val / 6.0) * 1.5
        raw_scale_6 = max_val / 6.0
        
        scale_4 = quantize_fp8_simulate(raw_scale_4, prevent_zero=prevent_zero, use_ue5m3=use_ue5m3)
        scale_6 = quantize_fp8_simulate(raw_scale_6, prevent_zero=prevent_zero, use_ue5m3=use_ue5m3)
        
        # Safe division for scale 4
        safe_scale_4 = torch.where(scale_4 != 0.0, scale_4, torch.ones_like(scale_4))
        scaled_4 = x_reshaped / safe_scale_4
        scaled_4 = torch.where(scale_4 != 0.0, scaled_4, torch.zeros_like(scaled_4))
        clipped_4 = torch.clamp(scaled_4, min=-6.0, max=6.0)
        quant_4 = quantize_fp4(torch.abs(clipped_4), torch.sign(clipped_4))
        dequant_4 = quant_4 * scale_4
        
        # Safe division for scale 6
        safe_scale_6 = torch.where(scale_6 != 0.0, scale_6, torch.ones_like(scale_6))
        scaled_6 = x_reshaped / safe_scale_6
        scaled_6 = torch.where(scale_6 != 0.0, scaled_6, torch.zeros_like(scaled_6))
        clipped_6 = torch.clamp(scaled_6, min=-6.0, max=6.0)
        quant_6 = quantize_fp4(torch.abs(clipped_6), torch.sign(clipped_6))
        dequant_6 = quant_6 * scale_6
        
        mse_4 = torch.mean((x_reshaped - dequant_4)**2, dim=-1, keepdim=True)
        mse_6 = torch.mean((x_reshaped - dequant_6)**2, dim=-1, keepdim=True)
        
        use_4 = mse_4 < mse_6
        
        dequant = torch.where(use_4, dequant_4, dequant_6)
        quant = torch.where(use_4, quant_4, quant_6)
        scaling_factor = torch.where(use_4, scale_4, scale_6)
        
    return dequant.reshape(init_shape).to(x.dtype), quant.reshape(init_shape), scaling_factor, use_4

class TorchMXLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, block_size=32, prevent_zero=True, four_over_six=False, use_ue5m3=False):
        super().__init__(in_features, out_features, bias)
        self.block_size = block_size
        self.prevent_zero = prevent_zero
        self.four_over_six = four_over_six
        self.use_ue5m3 = use_ue5m3
        
    def forward(self, input):
        q_weight, _, _, _ = FP4_quant_torch(self.weight, self.block_size, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_ue5m3=self.use_ue5m3)
        q_input, _, _, _ = FP4_quant_torch(input, self.block_size, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_ue5m3=self.use_ue5m3)
        return F.linear(q_input, q_weight, self.bias)

def run_eval(model_id, block_size=None, prevent_zero=True, four_over_six=False, use_ue5m3=False, num_steps=None):
    print(f"Evaluating {model_id} with block size {block_size}, prevent_zero={prevent_zero}, four_over_six={four_over_six}, use_ue5m3={use_ue5m3}, num_steps={num_steps}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    
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
                new_m = TorchMXLinear(module.in_features, module.out_features, module.bias is not None, block_size=block_size, prevent_zero=prevent_zero, four_over_six=four_over_six, use_ue5m3=use_ue5m3)
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
        use_ue5m3 = False
        num_steps = None
        csv_suffix = ""
        
        if len(sys.argv) > 3:
            prevent_zero = sys.argv[3].lower() == "true"
        if len(sys.argv) > 4:
            four_over_six = sys.argv[4].lower() == "true"
        if len(sys.argv) > 5:
            use_ue5m3 = sys.argv[5].lower() == "true"
        if len(sys.argv) > 6 and sys.argv[6].lower() != "none":
            num_steps = int(sys.argv[6])
        if len(sys.argv) > 7:
            csv_suffix = sys.argv[7]
            
        if not prevent_zero and not four_over_six and not use_ue5m3:
            option = "e4m3"
        elif prevent_zero and not four_over_six and not use_ue5m3:
            option = "e4m3 + PZ"
        elif not prevent_zero and four_over_six and not use_ue5m3:
            option = "e4m3 + 4o6"
        elif prevent_zero and four_over_six and not use_ue5m3:
            option = "e4m3 + 4o6 + PZ"
        elif prevent_zero and not four_over_six and use_ue5m3:
            option = "ue5m3 + PZ"
        elif not prevent_zero and four_over_six and use_ue5m3:
            option = "ue5m3 + 4o6"
        elif prevent_zero and four_over_six and use_ue5m3:
            option = "ue5m3 + 4o6 + PZ"
        else:
            option = "ue5m3_unknown"
            
        if block_sizes == [None]:
            base_ppl = None
        else:
            base_ppl = read_base_from_csv(model_id)
        
        for bs in block_sizes:
            ppl = run_eval(model_id, bs, prevent_zero=prevent_zero, four_over_six=four_over_six, use_ue5m3=use_ue5m3, num_steps=num_steps)
            if base_ppl is None and bs is not None:
                print(f"Baseline not found for {model_id}. Please run it first.")
                continue
            update_csv_and_readme(model_id, bs, ppl, base_ppl, option=option, csv_suffix=csv_suffix)
    else:
        ppl = run_eval(model_id, None)
        update_csv_and_readme(model_id, None, ppl, None)
