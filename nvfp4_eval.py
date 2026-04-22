import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
from tqdm import tqdm
from mx.mx_ops import quantize_mx_op
from mx import MxSpecs
import pandas as pd
import subprocess
import os
from mx.formats import _get_format_params

# -------------------------------------------------------------------------
# Quantization Logic - leveraging paper's fast ops
# -------------------------------------------------------------------------
def quantize_nvfp4(x, block_size):
    ebits, mbits, emax, max_norm, min_norm = _get_format_params("fp8_e4m3")
    
    mx_specs = MxSpecs(
        scale_bits=8,
        a_elem_format="fp4_e2m1",
        block_size=block_size,
        custom_cuda=True,
        a_scale_mode=152,
        w_scale_mode=152,
        scale_min=min_norm,
    )
    
    init_shape = x.shape
    assert init_shape[-1] % block_size == 0, f"Last dimension {init_shape[-1]} must be divisible by block_size {block_size}"
    
    x_reshaped = x.reshape(-1, block_size)
    
    qx = quantize_mx_op(
        x_reshaped.float(),
        mx_specs,
        elem_format="fp4_e2m1",
        axes=[-1],
        round=mx_specs["round_mx_output"],
    )
    
    return qx.reshape(init_shape).to(x.dtype)

class MXLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, block_size=32):
        super().__init__(in_features, out_features, bias)
        self.block_size = block_size
        
    def forward(self, input):
        q_weight = quantize_nvfp4(self.weight, self.block_size)
        q_input = quantize_nvfp4(input, self.block_size)
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
                new_m = MXLinear(module.in_features, module.out_features, module.bias is not None, block_size=block_size)
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
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
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
    path_csv = "/home/cjsschaefer_google_com/finer_is_better/results_pz.csv"
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
