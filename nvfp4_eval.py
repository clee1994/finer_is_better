import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
from tqdm import tqdm
from mx.mx_ops import quantize_mx_op
from mx import MxSpecs

# -------------------------------------------------------------------------
# Quantization Logic (leveraging paper's fast ops)
# -------------------------------------------------------------------------
def quantize_nvfp4(x, block_size):
    mx_specs = MxSpecs(
        scale_bits=8,
        a_elem_format='fp4_e2m1',
        block_size=block_size,
        custom_cuda=True,
        a_scale_mode=152,
        w_scale_mode=152,
    )
    
    init_shape = x.shape
    assert init_shape[-1] % block_size == 0, f'Last dimension {init_shape[-1]} must be divisible by block_size {block_size}'
    
    x_reshaped = x.reshape(-1, block_size)
    
    qx = quantize_mx_op(
        x_reshaped.float(),
        mx_specs,
        elem_format='fp4_e2m1',
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
    print(f'Evaluating {model_id} with block size {block_size}')
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to('cuda')
    
    if block_size is not None:
        head_name = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                head_name = name
        print(f'Identified head: {head_name}')
        
        for name, module in model.named_modules():
            if name == head_name:
                continue
            if "attn" in name:
                continue
            if isinstance(module, torch.nn.Linear):
                idx = name.rfind('.')
                if idx == -1:
                    idx = 0
                father_name = name[:idx]
                father_module = model
                if father_name:
                     for part in father_name.split('.'):
                         father_module = getattr(father_module, part)
                
                idx = idx + 1 if idx != 0 else idx
                new_m = MXLinear(module.in_features, module.out_features, module.bias is not None, block_size=block_size)
                new_m.weight.data = module.weight.data
                new_m.bias = module.bias
                print(f'Replacing layer: {name}')
                setattr(father_module, name[idx:], new_m)
                
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(testdata['text'])
    encodings = tokenizer(text, return_tensors='pt')
    
    seq_len = 2048
    stride = 2048
    
    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = i
        end_loc = min(i + seq_len, encodings.input_ids.size(1))
        trg_len = end_loc - begin_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cuda')
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            
        nlls.append(neg_log_likelihood)
        
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f'Perplexity: {ppl.item()}')
    
    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    model_id = sys.argv[1]
    if len(sys.argv) > 2:
        block_sizes = [int(x) for x in sys.argv[2].split(',')]
        for bs in block_sizes:
            run_eval(model_id, bs)
    else:
        run_eval(model_id, None)
