import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
import os
import tensorstore as ts
import numpy as np
from tqdm import tqdm
import ml_dtypes # registers bfloat16 in numpy

# Hardcoded random selections (seeded with 42)
# format: (layer_idx, sub_layer_name, type_annotation)
CONFIGS = {
    "meta-llama/Llama-3.1-8B": [
        (1, "gate_proj", "FFN1"),
        (7, "up_proj", "FFN1"),
        (14, "gate_proj", "FFN1"),
        (15, "gate_proj", "FFN1"),
        (17, "gate_proj", "FFN1"),
    ],
    "Qwen/Qwen2.5-14B": [
        (12, "down_proj", "FFN2"),
        (35, "up_proj", "FFN1"),
        (41, "down_proj", "FFN2"),
        (44, "gate_proj", "FFN1"),
        (45, "down_proj", "FFN2"),
    ]
}

def get_layer_module(model, layer_idx, sub_layer_name):
    # Both Llama 3.1 and Qwen 2.5 have the structure: model.layers[idx].mlp
    layer = model.model.layers[layer_idx]
    module = getattr(layer.mlp, sub_layer_name)
    return module

def extract_tensors(model_id, out_dir, num_steps=10):
    if model_id not in CONFIGS:
        raise ValueError(f"Unknown model_id: {model_id}")
        
    configs = CONFIGS[model_id]
    
    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
    
    hook_handles = []
    ts_datasets = {}
    
    os.makedirs(out_dir, exist_ok=True)
    
    # We use a list to make step_counter mutable inside the hook closure
    step_counter = [0]
    
    for layer_idx, sub_layer, ffn_type in configs:
        module = get_layer_module(model, layer_idx, sub_layer)
        
        # Get weight shape
        w_shape = list(module.weight.shape) # [out_features, in_features]
        
        # Input activation matches weight's input features dimension
        in_features = module.weight.shape[1]
        act_shape = [num_steps, 2048, in_features]
        
        layer_name = f"layer_{layer_idx}_{sub_layer}_{ffn_type}"
        layer_dir = os.path.join(out_dir, layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        
        # 1. Create TensorStore for weight (save immediately as it is static)
        w_path = os.path.join(layer_dir, "weight")
        w_spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': w_path,
            },
            'metadata': {
                'dtype': 'bfloat16',
                'shape': w_shape,
                'chunks': w_shape, # 1 single chunk is fine for weight
            },
            'create': True,
            'delete_existing': True,
        }
        w_ds = ts.open(w_spec).result()
        # Safe conversion to numpy bfloat16
        w_np_bf16 = module.weight.detach().cpu().to(torch.float32).numpy().astype(ml_dtypes.bfloat16)
        w_ds[...] = w_np_bf16
        print(f"Saved weight for {layer_name} (shape: {w_shape})")
        
        # 2. Create TensorStore for activations
        act_path = os.path.join(layer_dir, "act")
        act_spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': act_path,
            },
            'metadata': {
                'dtype': 'bfloat16',
                'shape': act_shape,
                'chunks': [1, 2048, in_features], # Chunk by step (1 step per chunk)
            },
            'create': True,
            'delete_existing': True,
        }
        act_ds = ts.open(act_spec).result()
        ts_datasets[layer_name] = act_ds
        
        # Define forward hook to capture activations
        def make_hook(name, dataset):
            def hook(mod, inp, out):
                # inp[0] is the input tensor of shape (B, S, H)
                act_tensor = inp[0]
                idx = step_counter[0]
                if idx < num_steps:
                    # Squeeze batch dimension of size 1 to match TensorStore slice shape, then convert to numpy
                    act_np = act_tensor.squeeze(0).detach().cpu().to(torch.float32).numpy().astype(ml_dtypes.bfloat16)
                    # Write into specific step chunk index
                    dataset[idx, ...] = act_np
            return hook
            
        handle = module.register_forward_hook(make_hook(layer_name, act_ds))
        hook_handles.append(handle)
        
    # Load Wikitext-2 test split for forward pass inputs
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(testdata["text"])
    encodings = tokenizer(text, return_tensors="pt")
    
    seq_len = 2048
    stride = 2048
    
    loop = range(0, encodings.input_ids.size(1), stride)
    # Limit evaluation steps to precisely what we need to satisfy num_steps
    loop = list(loop)[:num_steps]
    
    print(f"Starting forward passes for {num_steps} steps to extract activations...")
    for i in tqdm(loop):
        begin_loc = i
        end_loc = min(i + seq_len, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            _ = model(input_ids, labels=target_ids)
            
        step_counter[0] += 1
        
    # Cleanup hooks
    for h in hook_handles:
        h.remove()
        
    print(f"Extraction fully completed! Tensors successfully saved to: {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 extract_tensors.py <model_id> <out_dir> [num_steps]")
        sys.exit(1)
        
    model_id = sys.argv[1]
    out_dir = sys.argv[2]
    num_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    extract_tensors(model_id, out_dir, num_steps)
