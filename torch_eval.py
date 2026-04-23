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

def quantize_fp8_simulate(val):
    # Clamp to max representable in FP8 UE4M3 (bias=7): 240.0
    val = torch.clamp(val, min=0.0, max=240.0)
    
    is_normal = val >= 2**-6
    
    # Subnormal quantization: step is 2**-9
    quant_sub = torch.round(val / 2**-9) * 2**-9
    
    # Normal quantization:
    e = torch.floor(torch.log2(val))
    e_clipped = torch.clamp(e, max=7.0)
    m = torch.round((val / 2**e_clipped - 1.0) * 8.0)
    
    # Carry-over handling
    carry = m == 8.0
    e_clipped = torch.where(carry, e_clipped + 1.0, e_clipped)
    m = torch.where(carry, torch.zeros_like(m), m)
    
    quant_norm = (1.0 + m/8.0) * 2**e_clipped
    
    quant = torch.where(is_normal, quant_norm, quant_sub)
    
    # Prevent Zero trick
    min_scale = 2**-9
    quant = torch.where(quant == 0.0, min_scale, quant)
    
    return quant

# -------------------------------------------------------------------------
# Quantization Logic (Pure PyTorch Simulation)
# -------------------------------------------------------------------------
def FP4_quant_torch(x, block_size):
    init_shape = x.shape
    x_reshaped = x.reshape(-1, block_size)
    
    # Line 2 in JAX: max_val = jnp.max(jnp.abs(x), axis=0, keepdims=True)
    max_val = torch.max(torch.abs(x_reshaped), dim=-1, keepdim=True).values
    
    # Line 3 in JAX: raw_scale = max_val / 6.0
    raw_scale = max_val / 6.0
    
    # Line 4 in JAX: scaling_factor = jnp.float32(jnp.float8_e4m3fn(raw_scale))
    # Use our proper FP8 simulation!
    scaling_factor = quantize_fp8_simulate(raw_scale)
    
    # Line 5 in JAX: scaled = jnp.where(scaling_factor != 0, x / scaling_factor, 0.0)
    scaled = x_reshaped / scaling_factor
    
    # Line 6 in JAX: clipped = jnp.clip(scaled, -6.0, 6.0)
    clipped = torch.clamp(scaled, min=-6.0, max=6.0)
    
    # Preserve sign!
    sign = torch.sign(clipped)
    abs_clipped = torch.abs(clipped)
    
    # Line 7 in JAX: quant = np.float32(jnp.float4_e2m1fn(clipped))
    # Map to the discrete FP4 grid: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32).cuda()
    dist = torch.abs(abs_clipped.unsqueeze(-1) - grid)
    idx = torch.argmin(dist, dim=-1)
    quant = grid[idx]
    
    # Apply sign back!
    quant = quant * sign
    
    # Dequantize
    dequant = quant * scaling_factor
    
    # Return dequant, quant, and scaling_factor!
    return dequant.reshape(init_shape).to(x.dtype), quant.reshape(init_shape), scaling_factor

class TorchMXLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, block_size=32):
        super().__init__(in_features, out_features, bias)
        self.block_size = block_size
        
    def forward(self, input):
        # Only use the first return value (dequantized) for linear op!
        q_weight, _, _ = FP4_quant_torch(self.weight, self.block_size)
        q_input, _, _ = FP4_quant_torch(input, self.block_size)
        return F.linear(q_input, q_weight, self.bias)

# -------------------------------------------------------------------------
# Evaluation Logic
# -------------------------------------------------------------------------
def run_eval(model_id, block_size=None):
    print(f"Evaluating {model_id} with block size {block_size}")
    
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
                new_m = TorchMXLinear(module.in_features, module.out_features, module.bias is not None, block_size=block_size)
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
    for i, _ in zip(tqdm(range(0, encodings.input_ids.size(1), stride)), range(20)):
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

def update_csv_and_readme(model_id, bs, ppl, base_ppl):
    path_csv = "/home/cjsschaefer_google_com/finer_is_better/results_torch.csv"
    if os.path.exists(path_csv):
        df = pd.read_csv(path_csv, index_col=0)
    else:
        models = ["Llama 3.1 8B", "Granite 3.3 8B", "Qwen 2.5 14B", "DeepSeek 7B"]
        columns = ["Baseline", "BS=4", "BS=8", "BS=16", "BS=32", "BS=64", "BS=128", "BS=256"]
        df = pd.DataFrame(index=models, columns=columns)
        df.index.name = "Model"
        
    disp_name = model_id
    for k, v in {"granite": "Granite 3.3 8B", "llama": "Llama 3.1 8B", "qwen": "Qwen 2.5 14B", "deepseek": "DeepSeek 7B"}.items():
        if k in model_id.lower():
            disp_name = v
            break
            
    if bs is None:
        df.loc[disp_name, "Baseline"] = ppl
    else:
        gap = ppl - base_ppl
        df.loc[disp_name, f"BS={bs}"] = gap
        
    df.to_csv(path_csv)
    
    subprocess.run(["python3", "/home/cjsschaefer_google_com/finer_is_better/update_readme.py"])

if __name__ == "__main__":
    model_id = sys.argv[1]
    
    def read_base_from_csv(model_id):
        path_csv = "/home/cjsschaefer_google_com/finer_is_better/results.csv"
        if os.path.exists(path_csv):
            df = pd.read_csv(path_csv, index_col=0)
            disp_name = model_id
            for k, v in {"granite": "Granite 3.3 8B", "llama": "Llama 3.1 8B", "qwen": "Qwen 2.5 14B", "deepseek": "DeepSeek 7B"}.items():
                if k in model_id.lower():
                    disp_name = v
                    break
            return df.loc[disp_name, "Baseline"]
        return None
        
    if len(sys.argv) > 2:
        block_sizes = [int(x) for x in sys.argv[2].split(",")]
        base_ppl = read_base_from_csv(model_id)
        
        for bs in block_sizes:
            ppl = run_eval(model_id, bs)
            if base_ppl is None:
                print(f"Baseline not found for {model_id}. Please run it first.")
                continue
            update_csv_and_readme(model_id, bs, ppl, base_ppl)
    else:
        ppl = run_eval(model_id, None)
        update_csv_and_readme(model_id, None, ppl, None)
