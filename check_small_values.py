import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import sys

def check_model(model_id):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    
    # Check layer 2 down_proj weights
    layer_name = "model.layers.2.mlp.down_proj"
    module = model
    for part in layer_name.split("."):
        module = getattr(module, part)
    
    weights = module.weight.detach().cpu().float().numpy().flatten()
    abs_w = np.abs(weights)
    
    # Range between 2^-17 and 2^-9
    p_small = np.mean((abs_w >= 2**-17) & (abs_w < 2**-9)) * 100
    p_very_small = np.mean((abs_w > 0) & (abs_w < 2**-17)) * 100
    
    print(f"\n{model_id} Results:")
    print(f"  % of weights in [2^-17, 2^-9): {p_small:.4f}%")
    print(f"  % of non-zero weights < 2^-17: {p_very_small:.4f}%")
    
    del model
    torch.cuda.empty_cache()

def main():
    models = [
        "ibm-granite/granite-3.3-8b-base",
        "meta-llama/Llama-3.1-8B",
        "deepseek-ai/deepseek-llm-7b-base",
        "Qwen/Qwen2.5-14B"
    ]
    for m in models:
        try:
            check_model(m)
        except Exception as e:
            print(f"Failed for {m}: {e}")

if __name__ == "__main__":
    main()
