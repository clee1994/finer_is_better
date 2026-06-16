import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
import sys
from tqdm import tqdm
import pandas as pd
import subprocess
import os
import numpy as np
import scipy.linalg
import custom_rot_matrices as crm


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

def get_signed_fp4_values(fp4_format, device="cuda"):
    grid_pos, _, _ = get_element_format_grid(fp4_format, device=device)
    grid_neg = -grid_pos[1:].flip(dims=[0])
    fp4_vals = torch.cat([grid_neg, grid_pos])
    return fp4_vals

# -------------------------------------------------------------------------
# Discrete Wavelet Transform (Haar) & Stamp Matmul Quantization Logic
# -------------------------------------------------------------------------
def quant_torch(x, bits, axis):
    scale = torch.max(torch.abs(x), dim=axis, keepdim=True).values
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    scale = (1.0 / scale) * (2**bits - 1)
    
    x_scaled = x * scale
    x_rounded = torch.round(x_scaled)
    return x_rounded / scale

def quant_fp8_torch(x, axis):
    # Dynamic dynamic scaled FP8 (e4m3) quantizer, no microscales/block-sizes
    # Maps per-tensor (channel-wise/token-wise) relative to OCP FP8 max limit 448.0
    scale = torch.max(torch.abs(x), dim=axis, keepdim=True).values
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    
    scale_factor = 448.0 / scale
    
    x_scaled = x * scale_factor
    x_fp8 = quantize_fp8_simulate(x_scaled, prevent_zero=True, format="e4m3")
    
    return x_fp8 / scale_factor

def apply_kpass_butterfly_torch(tensor, passes=2, dim=-1):
    assert tensor.shape[dim] == 32
    scale = 0.5 ** (passes / 2.0)
    device = tensor.device
    dtype = tensor.dtype
    
    all_strides = [16, 8, 4, 2, 1]
    strides_to_apply = []
    for i in range(passes):
        strides_to_apply.append(all_strides[i % 5])
        
    if dim != -1 and dim != tensor.ndim - 1:
        tensor = tensor.transpose(dim, -1)
        
    orig_shape = tensor.shape
    current = tensor.float()
    for stride in strides_to_apply:
        chunk_size = 2 * stride
        num_chunks = 32 // chunk_size
        reshaped = current.reshape(orig_shape[:-1] + (num_chunks, 2, stride))
        
        a = reshaped.narrow(-2, 0, 1)
        b = reshaped.narrow(-2, 1, 1)
        
        sum_val = a + b
        diff_val = a - b
        
        passed = torch.cat([sum_val, diff_val], dim=-2)
        current = passed.reshape(orig_shape)
        
    tensor_out = (current * scale).to(dtype)
    
    if dim != -1 and dim != len(orig_shape) - 1:
        tensor_out = tensor_out.transpose(dim, -1)
        
    return tensor_out



def get_haar_matrix(N, device="cuda", dtype=torch.float32):
    if N == 1:
        return torch.tensor([[1.0]], device=device, dtype=dtype)
    
    H_1 = torch.zeros(N, N, device=device, dtype=dtype)
    for i in range(N // 2):
        H_1[i, 2*i] = 1.0 / math.sqrt(2)
        H_1[i, 2*i+1] = 1.0 / math.sqrt(2)
        H_1[N//2 + i, 2*i] = 1.0 / math.sqrt(2)
        H_1[N//2 + i, 2*i+1] = -1.0 / math.sqrt(2)
        
    if N > 2:
        H_sub = get_haar_matrix(N // 2, device=device, dtype=dtype)
        H_1[:N//2, :] = torch.matmul(H_sub, H_1[:N//2, :])
        
    return H_1

def get_haar_l1_matrix(N, device="cuda", dtype=torch.float32):
    H_1 = torch.zeros(N, N, device=device, dtype=dtype)
    h = 1.0 / math.sqrt(2.0)
    for i in range(N // 2):
        H_1[i, 2*i] = h
        H_1[i, 2*i+1] = h
        H_1[N//2 + i, 2*i] = h
        H_1[N//2 + i, 2*i+1] = -h
    return H_1

def make_grp128_l2_matrix(k_dim, device="cuda", dtype=torch.float32):
    assert k_dim % 128 == 0, "Hidden dimension K must be a multiple of 128."
    num_groups = k_dim // 128
    
    W128_L1 = get_haar_l1_matrix(128, device=device, dtype=dtype)
    W64_L1 = get_haar_l1_matrix(64, device=device, dtype=dtype)
    
    W_block = torch.eye(128, dtype=dtype, device=device)
    W_block[:64, :64] = W64_L1
    W128_L2 = torch.matmul(W_block, W128_L1)
    
    Q = torch.kron(torch.eye(num_groups, dtype=dtype, device=device), W128_L2)
    return Q

def make_grp128_l2_packet_matrix(k_dim, device="cuda", dtype=torch.float32):
    assert k_dim % 128 == 0, "Hidden dimension K must be a multiple of 128."
    num_groups = k_dim // 128
    
    W128_L1 = get_haar_l1_matrix(128, device=device, dtype=dtype)
    W64_L1 = get_haar_l1_matrix(64, device=device, dtype=dtype)
    
    W_block = torch.zeros((128, 128), dtype=dtype, device=device)
    W_block[:64, :64] = W64_L1
    W_block[64:, 64:] = W64_L1
    W128_L2_packet = torch.matmul(W_block, W128_L1)
    
    Q = torch.kron(torch.eye(num_groups, dtype=dtype, device=device), W128_L2_packet)
    return Q


def hadamard_matrix_torch(n, device="cuda", dtype=torch.float32):
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H * (1.0 / math.sqrt(n))

def sequency_ordered_hadamard(n, device="cuda", dtype=torch.float32):
    H = hadamard_matrix_torch(n, device=device, dtype=dtype)
    sign_changes = torch.sum((H[:, 1:] * H[:, :-1]) < 0, dim=1)
    sorted_idx = torch.argsort(sign_changes)
    return H[sorted_idx, :], sorted_idx

def get_ortho_matrix(N, transform_name, device="cuda", dtype=torch.float32):
    i = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
    j = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)
    if transform_name == "dct":
        D = torch.sqrt(torch.tensor(2.0 / N, device=device, dtype=dtype)) * torch.cos(math.pi / N * (j + 0.5) * i)
        D[0, :] = 1.0 / math.sqrt(N)
        return D
    elif transform_name == "dst":
        S = torch.sqrt(torch.tensor(2.0 / (N + 1), device=device, dtype=dtype)) * torch.sin(math.pi / (N + 1) * (i + 1) * (j + 1))
        return S
    elif transform_name == "dht":
        angle = 2.0 * math.pi * i * j / N
        H = (1.0 / math.sqrt(N)) * (torch.cos(angle) + torch.sin(angle))
        return H
    elif transform_name == "haar":
        return get_haar_matrix(N, device=device, dtype=dtype)
    elif transform_name == "haar_grp128_l2":
        return make_grp128_l2_matrix(N, device=device, dtype=dtype)
    elif transform_name == "haar_grp128_l2_packet":
        return make_grp128_l2_packet_matrix(N, device=device, dtype=dtype)
    elif transform_name == "wht":
        H_wht, _ = sequency_ordered_hadamard(N, device=device, dtype=dtype)
        return H_wht
    else:
        raise ValueError(f"Unknown orthonormal transform: {transform_name}")

def transform_seq_matmul_bf16_fp_torch(x, y, stamp_size=64, transform_name="dct"):
    S = x.shape[0]
    D = get_ortho_matrix(S, transform_name, device=x.device, dtype=x.dtype)
    x_trans = torch.matmul(D, x)
    
    act_q8_stamp = x_trans[:stamp_size, :].to(torch.bfloat16)
    act_q4_rest, _, _, _ = quantize_mx_torch(
        x_trans[stamp_size:, :], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    wgt_q8 = y.to(torch.bfloat16)
    wgt_q4, _, _, _ = quantize_mx_torch(
        y, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    out_q8 = torch.matmul(act_q8_stamp, wgt_q8.t())
    out_q4 = torch.matmul(act_q4_rest, wgt_q4.t())
    
    out_mixed = torch.cat((out_q8, out_q4), dim=0)
    out_spatial = torch.matmul(D.t(), out_mixed)
    return out_spatial

def spec_feat_matmul_bf16_fp_torch(x, y, D, stamp_size=64, pattern="contiguous", stamp_indices=None):
    H_pad = x.shape[1]
    act_trans = torch.matmul(x, D.t())
    wgt_trans = torch.matmul(y, D.t())
    
    if pattern == "contiguous":
        stamp_idx = torch.arange(stamp_size, device=x.device)
        rest_idx = torch.arange(stamp_size, H_pad, device=x.device)
    elif pattern == "interleaved":
        step = H_pad // stamp_size
        stamp_idx = torch.arange(0, H_pad, step, device=x.device)[:stamp_size]
        mask = torch.ones(H_pad, dtype=torch.bool, device=x.device)
        mask[stamp_idx] = False
        rest_idx = torch.arange(H_pad, device=x.device)[mask]
    elif pattern == "calibrated":
        assert stamp_indices is not None, "stamp_indices must be calibrated and provided!"
        stamp_idx = stamp_indices
        mask = torch.ones(H_pad, dtype=torch.bool, device=x.device)
        mask[stamp_idx] = False
        rest_idx = torch.arange(H_pad, device=x.device)[mask]
    else:
        raise ValueError(f"Unknown SPEC pattern: {pattern}")
        
    act_q8_stamp = act_trans[:, stamp_idx].to(torch.bfloat16)
    Rest_size = len(rest_idx)
    if Rest_size > 0:
        act_q4_rest, _, _, _ = quantize_mx_torch(
            act_trans[:, rest_idx], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
        )
    else:
        act_q4_rest = torch.zeros(act_trans.shape[0], 0, device=x.device, dtype=x.dtype)
        
    wgt_q8 = wgt_trans[:, stamp_idx].to(torch.bfloat16)
    if Rest_size > 0:
        wgt_q4, _, _, _ = quantize_mx_torch(
            wgt_trans[:, rest_idx], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
        )
    else:
        wgt_q4 = torch.zeros(wgt_trans.shape[0], 0, device=x.device, dtype=x.dtype)
        
    out_q8 = torch.matmul(act_q8_stamp, wgt_q8.t())
    if Rest_size > 0:
        out_q4 = torch.matmul(act_q4_rest, wgt_q4.t())
    else:
        out_q4 = 0
        
    return (out_q8 + out_q4).to(x.dtype)

def transform_feat_matmul_bf16_fp_torch(x, y, stamp_size=64, transform_name="dct"):
    H = x.shape[1]
    D = get_ortho_matrix(H, transform_name, device=x.device, dtype=x.dtype)
    act_trans = torch.matmul(x, D.t())
    wgt_trans = torch.matmul(y, D.t())
    
    act_q8_stamp = act_trans[:, :stamp_size].to(torch.bfloat16)
    act_q4_rest, _, _, _ = quantize_mx_torch(
        act_trans[:, stamp_size:], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    wgt_q8 = wgt_trans[:, :stamp_size].to(torch.bfloat16)
    wgt_q4, _, _, _ = quantize_mx_torch(
        wgt_trans[:, stamp_size:], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    out_q8 = torch.matmul(act_q8_stamp, wgt_q8.t())
    out_q4 = torch.matmul(act_q4_rest, wgt_q4.t())
    return (out_q8 + out_q4).to(x.dtype)

def one_level_dwt_columns_torch(x):
    assert x.shape[1] % 2 == 0
    trend = (x[:, 0::2] + x[:, 1::2]) * (1.0 / math.sqrt(2))
    detail = (x[:, 0::2] - x[:, 1::2]) * (1.0 / math.sqrt(2))
    return torch.cat((trend, detail), dim=1)

def transform_iterative_matmul_bf16_fp_torch(x, y, stamp_size=64, step_size=16):
    device = x.device
    dtype = x.dtype
    
    num_iters = stamp_size // step_size
    x_active = x.clone()
    y_active = y.clone()
    
    act_stamps = []
    wgt_stamps = []
    
    for t in range(num_iters):
        H_t = x_active.shape[1]
        x_trans = one_level_dwt_columns_torch(x_active)
        y_trans = one_level_dwt_columns_torch(y_active)
        
        energy = torch.sum(x_trans**2, dim=0)
        sorted_vals, sorted_idx = torch.sort(energy, descending=True)
        selected_cols = sorted_idx[:step_size]
        
        act_stamps.append(x_trans[:, selected_cols].to(torch.bfloat16))
        wgt_stamps.append(y_trans[:, selected_cols].to(torch.bfloat16))
        
        mask = torch.ones(H_t, dtype=torch.bool, device=device)
        mask[selected_cols] = False
        x_active = x_trans[:, mask]
        y_active = y_trans[:, mask]
        
    act_quant, _, _, _ = quantize_mx_torch(
        x_active, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    wgt_quant, _, _, _ = quantize_mx_torch(
        y_active, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    act_stamp_all = torch.cat(act_stamps, dim=1)
    wgt_stamp_all = torch.cat(wgt_stamps, dim=1)
    
    out_stamp = torch.matmul(act_stamp_all, wgt_stamp_all.t())
    out_rest = torch.matmul(act_quant, wgt_quant.t())
    
    return (out_stamp + out_rest).to(x.dtype)

def svd_matmul_bf16_fp_torch(x, y, stamp_size=64):
    U, S_vals, Vh = torch.linalg.svd(x.float(), full_matrices=False)
    Uw, Sw_vals, Vhw = torch.linalg.svd(y.float(), full_matrices=False)
    
    k = min(stamp_size, len(S_vals), len(Sw_vals))
    
    act_q8_stamp = (U[:, :k] * S_vals[:k]) @ Vh[:k, :]
    wgt_q8_stamp = (Uw[:, :k] * Sw_vals[:k]) @ Vhw[:k, :]
    
    act_residual = x.float() - act_q8_stamp
    wgt_residual = y.float() - wgt_q8_stamp
    
    act_q4_rest, _, _, _ = quantize_mx_torch(
        act_residual.to(x.dtype), 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    wgt_q4_rest, _, _, _ = quantize_mx_torch(
        wgt_residual.to(y.dtype), 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    out_q8 = torch.matmul(act_q8_stamp.to(torch.bfloat16), wgt_q8_stamp.to(torch.bfloat16).t())
    out_q4 = torch.matmul(act_q4_rest, wgt_q4_rest.t())
    return (out_q8 + out_q4).to(x.dtype)

def hybrid_matmul_haar_grp128_l2_torch(act, wgt, Q_k, stamp_size=256):
    device = act.device
    dtype = act.dtype
    
    K = Q_k.shape[0]
    H_w = wgt.shape[1]
    H_x = act.shape[1]
    
    if K > H_x:
        act_padded, _ = pad_columns_to_power_of_2(act, K)
    else:
        act_padded = act
        
    if K > H_w:
        wgt_padded, _ = pad_columns_to_power_of_2(wgt, K)
    else:
        wgt_padded = wgt
        
    M = act_padded.shape[0]
    Out = wgt_padded.shape[0]
    num_groups = K // 128
    stamp_per_group = stamp_size // num_groups
    
    act_trans = torch.matmul(act_padded, Q_k)       # (M, K)
    wgt_trans = torch.matmul(wgt_padded, Q_k)       # (Out, K)
    
    act_trans_reshaped = act_trans.view(M, num_groups, 128)
    energy_per_group = torch.sum(act_trans_reshaped ** 2, dim=0)  # (num_groups, 128)
    
    sorted_idx_per_group = torch.argsort(energy_per_group, dim=1, descending=True)  # (num_groups, 128)
    top_idx_per_group = sorted_idx_per_group[:, :stamp_per_group]                   # (num_groups, stamp_per_group)
    
    offsets = (torch.arange(num_groups, device=device) * 128).unsqueeze(1)          # (num_groups, 1)
    stamp_cols = (top_idx_per_group + offsets).view(-1)                             # (stamp_size,)
    
    mask = torch.ones(K, dtype=torch.bool, device=device)
    mask[stamp_cols] = False
    rem_cols = torch.where(mask)[0]                                                 # (K - stamp_size,)
    
    act_stamp = act_trans[:, stamp_cols].to(torch.bfloat16)
    wgt_stamp = wgt_trans[:, stamp_cols].to(torch.bfloat16)
    out_stamp = torch.matmul(act_stamp, wgt_stamp.t())
    
    act_rem = act_trans[:, rem_cols]
    wgt_rem = wgt_trans[:, rem_cols]
    
    rem_dim = act_rem.shape[1]
    pad_len = (32 - (rem_dim % 32)) % 32
    if pad_len > 0:
        act_rem_padded = torch.cat([act_rem, torch.zeros((act_rem.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
        wgt_rem_padded = torch.cat([wgt_rem, torch.zeros((wgt_rem.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
    else:
        act_rem_padded = act_rem
        wgt_rem_padded = wgt_rem
        
    act_rem_q_padded, _, _, _ = quantize_mx_torch(act_rem_padded, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    wgt_rem_q_padded, _, _, _ = quantize_mx_torch(wgt_rem_padded, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    if pad_len > 0:
        act_rem_q = act_rem_q_padded[:, :rem_dim]
        wgt_rem_q = wgt_rem_q_padded[:, :rem_dim]
    else:
        act_rem_q = act_rem_q_padded
        wgt_rem_q = wgt_rem_q_padded
        
    out_rem = torch.matmul(act_rem_q, wgt_rem_q.t())
    
    Y_recon = (out_stamp + out_rem).to(dtype)
    return Y_recon

def hybrid_matmul_haar_grp128_l2_trends_only_torch(act, wgt, Q_k, stamp_size=256):
    device = act.device
    dtype = act.dtype
    
    K = Q_k.shape[0]
    H_w = wgt.shape[1]
    H_x = act.shape[1]
    
    if K > H_x:
        act_padded, _ = pad_columns_to_power_of_2(act, K)
    else:
        act_padded = act
        
    if K > H_w:
        wgt_padded, _ = pad_columns_to_power_of_2(wgt, K)
    else:
        wgt_padded = wgt
        
    M = act_padded.shape[0]
    Out = wgt_padded.shape[0]
    num_groups = K // 128
    stamp_per_group = stamp_size // num_groups
    
    act_trans = torch.matmul(act_padded, Q_k)       # (M, K)
    wgt_trans = torch.matmul(wgt_padded, Q_k)       # (Out, K)
    
    act_trans_reshaped = act_trans.view(M, num_groups, 128)
    
    # Calculate energy ONLY on the first 32 channels (Level-2 trends) of each group
    act_trends = act_trans_reshaped[:, :, :32]
    energy_per_group = torch.sum(act_trends ** 2, dim=0)  # (num_groups, 32)
    
    # Sort ONLY within the 32 trend channels
    sorted_idx_per_group = torch.argsort(energy_per_group, dim=1, descending=True)  # (num_groups, 32)
    top_idx_per_group = sorted_idx_per_group[:, :stamp_per_group]                   # (num_groups, stamp_per_group)
    
    # Combine back into global stamp indices
    offsets = (torch.arange(num_groups, device=device) * 128).unsqueeze(1)          # (num_groups, 1)
    stamp_cols = (top_idx_per_group + offsets).view(-1)                             # (stamp_size,)
    
    mask = torch.ones(K, dtype=torch.bool, device=device)
    mask[stamp_cols] = False
    rem_cols = torch.where(mask)[0]                                                 # (K - stamp_size,)
    
    act_stamp = act_trans[:, stamp_cols].to(torch.bfloat16)
    wgt_stamp = wgt_trans[:, stamp_cols].to(torch.bfloat16)
    out_stamp = torch.matmul(act_stamp, wgt_stamp.t())
    
    act_rem = act_trans[:, rem_cols]
    wgt_rem = wgt_trans[:, rem_cols]
    
    rem_dim = act_rem.shape[1]
    pad_len = (32 - (rem_dim % 32)) % 32
    if pad_len > 0:
        act_rem_padded = torch.cat([act_rem, torch.zeros((act_rem.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
        wgt_rem_padded = torch.cat([wgt_rem, torch.zeros((wgt_rem.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
    else:
        act_rem_padded = act_rem
        wgt_rem_padded = wgt_rem
        
    act_rem_q_padded, _, _, _ = quantize_mx_torch(act_rem_padded, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    wgt_rem_q_padded, _, _, _ = quantize_mx_torch(wgt_rem_padded, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    if pad_len > 0:
        act_rem_q = act_rem_q_padded[:, :rem_dim]
        wgt_rem_q = wgt_rem_q_padded[:, :rem_dim]
    else:
        act_rem_q = act_rem_q_padded
        wgt_rem_q = wgt_rem_q_padded
        
    out_rem = torch.matmul(act_rem_q, wgt_rem_q.t())
    
    Y_recon = (out_stamp + out_rem).to(dtype)
    return Y_recon

def hybrid_matmul_haar_grp128_l2_first_k_torch(act, wgt, Q_k, stamp_size=256):
    device = act.device
    dtype = act.dtype
    
    K = Q_k.shape[0]
    H_w = wgt.shape[1]
    H_x = act.shape[1]
    
    if K > H_x:
        act_padded, _ = pad_columns_to_power_of_2(act, K)
    else:
        act_padded = act
        
    if K > H_w:
        wgt_padded, _ = pad_columns_to_power_of_2(wgt, K)
    else:
        wgt_padded = wgt
        
    M = act_padded.shape[0]
    Out = wgt_padded.shape[0]
    num_groups = K // 128
    stamp_per_group = stamp_size // num_groups
    
    act_trans = torch.matmul(act_padded, Q_k)       # (M, K)
    wgt_trans = torch.matmul(wgt_padded, Q_k)       # (Out, K)
    
    # Static Selection: pick the first `stamp_per_group` columns in each group of 128
    group_indices = torch.arange(stamp_per_group, device=device)  # [0, 1, ..., stamp_per_group - 1]
    offsets = (torch.arange(num_groups, device=device) * 128).unsqueeze(1)
    stamp_cols = (group_indices + offsets).view(-1)                 # Static stamp columns indices
    
    mask = torch.ones(K, dtype=torch.bool, device=device)
    mask[stamp_cols] = False
    rem_cols = torch.where(mask)[0]
    
    act_stamp = act_trans[:, stamp_cols].to(torch.bfloat16)
    wgt_stamp = wgt_trans[:, stamp_cols].to(torch.bfloat16)
    out_stamp = torch.matmul(act_stamp, wgt_stamp.t())
    
    act_rem = act_trans[:, rem_cols]
    wgt_rem = wgt_trans[:, rem_cols]
    
    rem_dim = act_rem.shape[1]
    pad_len = (32 - (rem_dim % 32)) % 32
    if pad_len > 0:
        act_rem_padded = torch.cat([act_rem, torch.zeros((act_rem.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
        wgt_rem_padded = torch.cat([wgt_rem, torch.zeros((wgt_rem.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
    else:
        act_rem_padded = act_rem
        wgt_rem_padded = wgt_rem
        
    act_rem_q_padded, _, _, _ = quantize_mx_torch(act_rem_padded, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    wgt_rem_q_padded, _, _, _ = quantize_mx_torch(wgt_rem_padded, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    if pad_len > 0:
        act_rem_q = act_rem_q_padded[:, :rem_dim]
        wgt_rem_q = wgt_rem_q_padded[:, :rem_dim]
    else:
        act_rem_q = act_rem_q_padded
        wgt_rem_q = wgt_rem_q_padded
        
    out_rem = torch.matmul(act_rem_q, wgt_rem_q.t())
    
    Y_recon = (out_stamp + out_rem).to(dtype)
    return Y_recon

def hybrid_matmul_spatial_grp128_torch(act, wgt, stamp_size=256):
    device = act.device
    dtype = act.dtype
    
    K = act.shape[1]
    assert K % 128 == 0
    
    M = act.shape[0]
    Out = wgt.shape[0]
    num_groups = K // 128
    stamp_per_group = stamp_size // num_groups
    
    act_trans = act
    wgt_trans = wgt
    
    act_trans_reshaped = act_trans.view(M, num_groups, 128)
    energy_per_group = torch.sum(act_trans_reshaped ** 2, dim=0)  # (num_groups, 128)
    
    sorted_idx_per_group = torch.argsort(energy_per_group, dim=1, descending=True)  # (num_groups, 128)
    top_idx_per_group = sorted_idx_per_group[:, :stamp_per_group]                   # (num_groups, stamp_per_group)
    
    offsets = (torch.arange(num_groups, device=device) * 128).unsqueeze(1)          # (num_groups, 1)
    stamp_cols = (top_idx_per_group + offsets).view(-1)                             # (stamp_size,)
    
    mask = torch.ones(K, dtype=torch.bool, device=device)
    mask[stamp_cols] = False
    rem_cols = torch.where(mask)[0]                                                 # (K - stamp_size,)
    
    act_stamp = act_trans[:, stamp_cols].to(torch.bfloat16)
    wgt_stamp = wgt_trans[:, stamp_cols].to(torch.bfloat16)
    out_stamp = torch.matmul(act_stamp, wgt_stamp.t())
    
    act_rem = act_trans[:, rem_cols]
    wgt_rem = wgt_trans[:, rem_cols]
    
    rem_dim = act_rem.shape[1]
    pad_len = (32 - (rem_dim % 32)) % 32
    if pad_len > 0:
        act_rem_padded = torch.cat([act_rem, torch.zeros((act_rem.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
        wgt_rem_padded = torch.cat([wgt_rem, torch.zeros((wgt_rem.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
    else:
        act_rem_padded = act_rem
        wgt_rem_padded = wgt_rem
        
    act_rem_q_padded, _, _, _ = quantize_mx_torch(act_rem_padded, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    wgt_rem_q_padded, _, _, _ = quantize_mx_torch(wgt_rem_padded, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    if pad_len > 0:
        act_rem_q = act_rem_q_padded[:, :rem_dim]
        wgt_rem_q = wgt_rem_q_padded[:, :rem_dim]
    else:
        act_rem_q = act_rem_q_padded
        wgt_rem_q = wgt_rem_q_padded
        
    out_rem = torch.matmul(act_rem_q, wgt_rem_q.t())
    
    Y_recon = (out_stamp + out_rem).to(dtype)
    return Y_recon

def hybrid_matmul_spatial_static_grp128_torch(act, wgt, stamp_size=256):
    device = act.device
    dtype = act.dtype
    
    K = act.shape[1]
    assert K % 128 == 0
    
    M = act.shape[0]
    Out = wgt.shape[0]
    num_groups = K // 128
    stamp_per_group = stamp_size // num_groups
    
    # Sort by STATIC weight energy column-wise per group
    wgt_reshaped = wgt.t().view(num_groups, 128, Out)
    wgt_energy_per_group = torch.sum(wgt_reshaped ** 2, dim=2)  # (num_groups, 128)
    
    sorted_idx_per_group = torch.argsort(wgt_energy_per_group, dim=1, descending=True)  # (num_groups, 128)
    top_idx_per_group = sorted_idx_per_group[:, :stamp_per_group]                       # (num_groups, stamp_per_group)
    
    offsets = (torch.arange(num_groups, device=device) * 128).unsqueeze(1)              # (num_groups, 1)
    stamp_cols = (top_idx_per_group + offsets).view(-1)                                 # (stamp_size,)
    
    mask = torch.ones(K, dtype=torch.bool, device=device)
    mask[stamp_cols] = False
    rem_cols = torch.where(mask)[0]                                                     # (K - stamp_size,)
    
    act_stamp = act[:, stamp_cols].to(torch.bfloat16)
    wgt_stamp = wgt[:, stamp_cols].to(torch.bfloat16)
    out_stamp = torch.matmul(act_stamp, wgt_stamp.t())
    
    act_rem = act[:, rem_cols]
    wgt_rem = wgt[:, rem_cols]
    
    rem_dim = act_rem.shape[1]
    pad_len = (32 - (rem_dim % 32)) % 32
    if pad_len > 0:
        act_rem_padded = torch.cat([act_rem, torch.zeros((act_rem.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
        wgt_rem_padded = torch.cat([wgt_rem, torch.zeros((wgt_rem.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
    else:
        act_rem_padded = act_rem
        wgt_rem_padded = wgt_rem
        
    act_rem_q_padded, _, _, _ = quantize_mx_torch(act_rem_padded, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    wgt_rem_q_padded, _, _, _ = quantize_mx_torch(wgt_rem_padded, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    if pad_len > 0:
        act_rem_q = act_rem_q_padded[:, :rem_dim]
        wgt_rem_q = wgt_rem_q_padded[:, :rem_dim]
    else:
        act_rem_q = act_rem_q_padded
        wgt_rem_q = wgt_rem_q_padded
        
    out_rem = torch.matmul(act_rem_q, wgt_rem_q.t())
    
    Y_recon = (out_stamp + out_rem).to(dtype)
    return Y_recon


def hybrid_matmul_zero_masked_grouped_torch(act, wgt, group_size=32, token_dynamic=True):
    device = act.device
    dtype = act.dtype
    
    T, C = act.shape
    O = wgt.shape[0]
    
    assert C % group_size == 0
    num_groups = C // group_size
    
    act_blocks = act.view(T, num_groups, group_size)
    
    if token_dynamic:
        outlier_idx = torch.argmax(torch.abs(act_blocks), dim=-1) # [T, num_groups]
        group_offsets = torch.arange(num_groups, device=device) * group_size
        global_idx = outlier_idx + group_offsets.unsqueeze(0) # [T, num_groups]
        
        X_heavy = torch.gather(act, dim=-1, index=global_idx) # [T, num_groups]
        
        global_idx_expanded = global_idx.unsqueeze(1).expand(-1, O, -1) # [T, O, num_groups]
        W_heavy = torch.gather(wgt.unsqueeze(0).expand(T, -1, -1), dim=-1, index=global_idx_expanded) # [T, O, num_groups]
        
        zeros = torch.zeros_like(X_heavy)
        act_light_blocks = torch.scatter(act_blocks.clone(), dim=-1, index=outlier_idx.unsqueeze(-1), src=zeros.unsqueeze(-1))
        act_light = act_light_blocks.view(T, C)
        
        Y_heavy = torch.einsum('tg,tog->to', X_heavy.to(torch.bfloat16), W_heavy.to(torch.bfloat16))
    else:
        scores = torch.sum(act_blocks ** 2, dim=0) # [num_groups, group_size]
        outlier_idx = torch.argmax(scores, dim=-1) # [num_groups]
        group_offsets = torch.arange(num_groups, device=device) * group_size
        global_idx = outlier_idx + group_offsets # [num_groups]
        
        X_heavy = act[:, global_idx] # [T, num_groups]
        W_heavy = wgt[:, global_idx] # [O, num_groups]
        
        act_light = act.clone()
        act_light[:, global_idx] = 0.0
        
        Y_heavy = torch.matmul(X_heavy.to(torch.bfloat16), W_heavy.to(torch.bfloat16).t())
        
    act_light_q, _, _, _ = quantize_mx_torch(act_light, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    wgt_q, _, _, _ = quantize_mx_torch(wgt, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    Y_light = torch.matmul(act_light_q, wgt_q.t())
    
    Y_recon = (Y_heavy + Y_light).to(dtype)
    return Y_recon


def hybrid_matmul_zero_masked_wht_grouped_torch(act, wgt, group_size=32, token_dynamic=True):
    device = act.device
    dtype = act.dtype
    
    T, C = act.shape
    O = wgt.shape[0]
    
    assert C % group_size == 0
    num_groups = C // group_size
    
    H, _ = sequency_ordered_hadamard(group_size, device=device, dtype=dtype)
    
    act_blocks = act.view(T, num_groups, group_size)
    
    if token_dynamic:
        outlier_idx = torch.argmax(torch.abs(act_blocks), dim=-1) # [T, num_groups]
        group_offsets = torch.arange(num_groups, device=device) * group_size
        global_idx = outlier_idx + group_offsets.unsqueeze(0) # [T, num_groups]
        
        X_heavy = torch.gather(act, dim=-1, index=global_idx) # [T, num_groups]
        
        global_idx_expanded = global_idx.unsqueeze(1).expand(-1, O, -1) # [T, O, num_groups]
        W_heavy = torch.gather(wgt.unsqueeze(0).expand(T, -1, -1), dim=-1, index=global_idx_expanded) # [T, O, num_groups]
        
        zeros = torch.zeros_like(X_heavy)
        act_light_blocks = torch.scatter(act_blocks.clone(), dim=-1, index=outlier_idx.unsqueeze(-1), src=zeros.unsqueeze(-1))
        
        Y_heavy = torch.einsum('tg,tog->to', X_heavy.to(torch.bfloat16), W_heavy.to(torch.bfloat16))
    else:
        scores = torch.sum(act_blocks ** 2, dim=0) # [num_groups, group_size]
        outlier_idx = torch.argmax(scores, dim=-1) # [num_groups]
        group_offsets = torch.arange(num_groups, device=device) * group_size
        global_idx = outlier_idx + group_offsets # [num_groups]
        
        X_heavy = act[:, global_idx] # [T, num_groups]
        W_heavy = wgt[:, global_idx] # [O, num_groups]
        
        act_light = act.clone()
        act_light[:, global_idx] = 0.0
        act_light_blocks = act_light.view(T, num_groups, group_size)
        
        Y_heavy = torch.matmul(X_heavy.to(torch.bfloat16), W_heavy.to(torch.bfloat16).t())
        
    act_light_trans_blocks = torch.matmul(act_light_blocks, H.t())
    act_light_trans = act_light_trans_blocks.view(T, C)
    
    wgt_blocks = wgt.view(O, num_groups, group_size)
    wgt_trans_blocks = torch.matmul(wgt_blocks, H.t())
    wgt_trans = wgt_trans_blocks.view(O, C)
    
    act_light_q, _, _, _ = quantize_mx_torch(act_light_trans, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    wgt_q, _, _, _ = quantize_mx_torch(wgt_trans, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    Y_light = torch.matmul(act_light_q, wgt_q.t())
    
    Y_recon = (Y_heavy + Y_light).to(dtype)
    return Y_recon


def hybrid_matmul_zero_masked_clipped_grouped_torch(act, wgt, group_size=32, token_dynamic=True, clip_ratio=None):
    device = act.device
    dtype = act.dtype
    
    T, C = act.shape
    O = wgt.shape[0]
    
    assert C % group_size == 0
    num_groups = C // group_size
    
    act_blocks = act.view(T, num_groups, group_size)
    
    if token_dynamic:
        outlier_idx = torch.argmax(torch.abs(act_blocks), dim=-1) # [T, num_groups]
        group_offsets = torch.arange(num_groups, device=device) * group_size
        global_idx = outlier_idx + group_offsets.unsqueeze(0) # [T, num_groups]
        
        X_heavy = torch.gather(act, dim=-1, index=global_idx) # [T, num_groups]
        
        global_idx_expanded = global_idx.unsqueeze(1).expand(-1, O, -1) # [T, O, num_groups]
        W_heavy = torch.gather(wgt.unsqueeze(0).expand(T, -1, -1), dim=-1, index=global_idx_expanded) # [T, O, num_groups]
        
        zeros = torch.zeros_like(X_heavy)
        act_light_blocks = torch.scatter(act_blocks.clone(), dim=-1, index=outlier_idx.unsqueeze(-1), src=zeros.unsqueeze(-1))
        
        if clip_ratio is not None:
            remaining_max = torch.max(torch.abs(act_light_blocks), dim=-1, keepdim=True).values
            threshold = clip_ratio * remaining_max
            act_light_blocks = torch.clamp(act_light_blocks, min=-threshold, max=threshold)
            
        act_light = act_light_blocks.view(T, C)
        Y_heavy = torch.einsum('tg,tog->to', X_heavy.to(torch.bfloat16), W_heavy.to(torch.bfloat16))
    else:
        scores = torch.sum(act_blocks ** 2, dim=0) # [num_groups, group_size]
        outlier_idx = torch.argmax(scores, dim=-1) # [num_groups]
        group_offsets = torch.arange(num_groups, device=device) * group_size
        global_idx = outlier_idx + group_offsets # [num_groups]
        
        X_heavy = act[:, global_idx] # [T, num_groups]
        W_heavy = wgt[:, global_idx] # [O, num_groups]
        
        act_light = act.clone()
        act_light[:, global_idx] = 0.0
        
        if clip_ratio is not None:
            act_light_blocks = act_light.view(T, num_groups, group_size)
            remaining_max = torch.max(torch.abs(act_light_blocks), dim=-1, keepdim=True).values
            threshold = clip_ratio * remaining_max
            act_light_blocks = torch.clamp(act_light_blocks, min=-threshold, max=threshold)
            act_light = act_light_blocks.view(T, C)
            
        Y_heavy = torch.matmul(X_heavy.to(torch.bfloat16), W_heavy.to(torch.bfloat16).t())
        
    act_light_q, _, _, _ = quantize_mx_torch(act_light, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    wgt_q, _, _, _ = quantize_mx_torch(wgt, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    Y_light = torch.matmul(act_light_q, wgt_q.t())
    
    Y_recon = (Y_heavy + Y_light).to(dtype)
    return Y_recon


def hybrid_matmul_zero_masked_butterfly_grouped_torch(act, wgt, wgt_transformed, group_size=32, token_dynamic=True, passes=2, n_extract=1):
    device = act.device
    dtype = act.dtype
    
    T, C = act.shape
    O = wgt.shape[0]
    
    assert C % group_size == 0
    num_groups = C // group_size
    
    act_blocks = act.view(T, num_groups, group_size)
    
    if token_dynamic:
        act_blocks_sliced = act_blocks[:, 0::n_extract, :]
        num_extract_groups = act_blocks_sliced.shape[1]
        
        outlier_idx_sliced = torch.argmax(torch.abs(act_blocks_sliced), dim=-1) # [T, num_extract_groups]
        
        group_offsets_sliced = torch.arange(0, num_groups, step=n_extract, device=device) * group_size
        global_idx_sliced = outlier_idx_sliced + group_offsets_sliced.unsqueeze(0) # [T, num_extract_groups]
        
        X_heavy = torch.gather(act, dim=-1, index=global_idx_sliced) # [T, num_extract_groups]
        global_idx_sliced_expanded = global_idx_sliced.unsqueeze(1).expand(-1, O, -1) # [T, O, num_extract_groups]
        W_heavy = torch.gather(wgt.unsqueeze(0).expand(T, -1, -1), dim=-1, index=global_idx_sliced_expanded) # [T, O, num_extract_groups]
        
        zeros = torch.zeros(T, num_extract_groups, 1, device=device, dtype=dtype)
        act_light_blocks = act_blocks.clone()
        act_light_blocks_sliced = act_light_blocks[:, 0::n_extract, :]
        torch.scatter(act_light_blocks_sliced, dim=-1, index=outlier_idx_sliced.unsqueeze(-1), src=zeros)
        
        Y_heavy = torch.einsum('tg,tog->to', X_heavy.to(torch.bfloat16), W_heavy.to(torch.bfloat16))
    else:
        act_blocks_sliced = act_blocks[:, 0::n_extract, :]
        scores_sliced = torch.sum(act_blocks_sliced ** 2, dim=0) # [num_extract_groups, group_size]
        outlier_idx_sliced = torch.argmax(scores_sliced, dim=-1) # [num_extract_groups]
        
        group_offsets_sliced = torch.arange(0, num_groups, step=n_extract, device=device) * group_size
        global_idx_sliced = outlier_idx_sliced + group_offsets_sliced # [num_extract_groups]
        
        X_heavy = act[:, global_idx_sliced] # [T, num_extract_groups]
        W_heavy = wgt[:, global_idx_sliced] # [O, num_extract_groups]
        
        act_light = act.clone()
        act_light[:, global_idx_sliced] = 0.0
        act_light_blocks = act_light.view(T, num_groups, group_size)
        
        Y_heavy = torch.matmul(X_heavy.to(torch.bfloat16), W_heavy.to(torch.bfloat16).t())
        
    act_light_transformed_blocks = apply_kpass_butterfly_torch(act_light_blocks, passes=passes, dim=-1)
    act_light_transformed = act_light_transformed_blocks.view(T, C)
    
    act_light_q, _, _, _ = quantize_mx_torch(act_light_transformed, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    wgt_q, _, _, _ = quantize_mx_torch(wgt_transformed, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    Y_light = torch.matmul(act_light_q, wgt_q.t())
    
    Y_recon = (Y_heavy + Y_light).to(dtype)
    return Y_recon


def hybrid_matmul_residual_quant_grouped_torch(act, wgt, group_size=32):
    device = act.device
    dtype = act.dtype
    
    act_q1, _, _, _ = quantize_mx_torch(act, group_size, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    act_res = act - act_q1
    act_res_q, _, _, _ = quantize_mx_torch(act_res, group_size, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    wgt_q, _, _, _ = quantize_mx_torch(wgt, group_size, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    Y_coarse = torch.matmul(act_q1, wgt_q.t())
    Y_fine = torch.matmul(act_res_q, wgt_q.t())
    
    Y_recon = (Y_coarse + Y_fine).to(dtype)
    return Y_recon


def hybrid_matmul_error_compaction_grouped_torch(act, wgt, wgt_trans_clipped, R, passes=5):
    device = act.device
    dtype = act.dtype
    
    T, C = act.shape
    O = wgt.shape[0]
    
    assert C % 32 == 0
    num_groups = C // 32
    
    act_q, _, _, _ = quantize_mx_torch(act, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    wgt_q, _, _, _ = quantize_mx_torch(wgt, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    err = act - act_q
    
    err_blocks = err.view(T, num_groups, 32)
    err_trans_blocks = apply_kpass_butterfly_torch(err_blocks, passes=passes, dim=-1)
    
    err_trans_clipped = err_trans_blocks.narrow(-1, 0, R).reshape(T, num_groups * R)
    
    err_trans_clipped_q, _, _, _ = quantize_mx_torch(err_trans_clipped, R, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    wgt_trans_clipped_q, _, _, _ = quantize_mx_torch(wgt_trans_clipped, R, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    Y_coarse = torch.matmul(act_q, wgt_q.t())
    Y_fine = torch.matmul(err_trans_clipped_q, wgt_trans_clipped_q.t())
    
    Y_recon = (Y_coarse + Y_fine).to(dtype)
    return Y_recon


def hybrid_matmul_spatial_grouped_torch(act, wgt, group_size, stamp_per_group, metric="energy"):
    device = act.device
    dtype = act.dtype
    
    K = act.shape[1]
    assert K % group_size == 0
    
    M = act.shape[0]
    Out = wgt.shape[0]
    num_groups = K // group_size
    stamp_size = num_groups * stamp_per_group
    
    act_trans = act
    wgt_trans = wgt
    
    act_trans_reshaped = act_trans.view(M, num_groups, group_size)
    
    if metric == "energy":
        metric_score = torch.sum(act_trans_reshaped ** 2, dim=0)  # (num_groups, group_size)
    elif metric == "abssum":
        metric_score = torch.sum(torch.abs(act_trans_reshaped), dim=0)
    elif metric == "pmr":
        peak = torch.max(torch.abs(act_trans_reshaped), dim=0).values
        median = torch.median(torch.abs(act_trans_reshaped), dim=0).values
        metric_score = peak / (median + 1e-5)
    elif metric == "mad":
        mean = torch.mean(act_trans_reshaped, dim=0)
        metric_score = torch.mean(torch.abs(act_trans_reshaped - mean), dim=0)
    else:
        raise ValueError(f"Unknown metric: {metric}")
        
    sorted_idx_per_group = torch.argsort(metric_score, dim=1, descending=True)  # (num_groups, group_size)
    top_idx_per_group = sorted_idx_per_group[:, :stamp_per_group]                   # (num_groups, stamp_per_group)
    
    offsets = (torch.arange(num_groups, device=device) * group_size).unsqueeze(1)    # (num_groups, 1)
    stamp_cols = (top_idx_per_group + offsets).view(-1)                             # (stamp_size,)
    
    mask = torch.ones(K, dtype=torch.bool, device=device)
    mask[stamp_cols] = False
    rem_cols = torch.where(mask)[0]
    
    act_stamp = act_trans[:, stamp_cols].to(torch.bfloat16)
    wgt_stamp = wgt_trans[:, stamp_cols].to(torch.bfloat16)
    out_stamp = torch.matmul(act_stamp, wgt_stamp.t())
    
    act_rem = act_trans[:, rem_cols]
    wgt_rem = wgt_trans[:, rem_cols]
    
    rem_dim = act_rem.shape[1]
    pad_len = (32 - (rem_dim % 32)) % 32
    if pad_len > 0:
        act_rem_padded = torch.cat([act_rem, torch.zeros((act_rem.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
        wgt_rem_padded = torch.cat([wgt_rem, torch.zeros((wgt_rem.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
    else:
        act_rem_padded = act_rem
        wgt_rem_padded = wgt_rem
        
    act_rem_q_padded, _, _, _ = quantize_mx_torch(act_rem_padded, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    wgt_rem_q_padded, _, _, _ = quantize_mx_torch(wgt_rem_padded, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True)
    
    if pad_len > 0:
        act_rem_q = act_rem_q_padded[:, :rem_dim]
        wgt_rem_q = wgt_rem_q_padded[:, :rem_dim]
    else:
        act_rem_q = act_rem_q_padded
        wgt_rem_q = wgt_rem_q_padded
        
    out_rem = torch.matmul(act_rem_q, wgt_rem_q.t())
    
    Y_recon = (out_stamp + out_rem).to(dtype)
    return Y_recon

def dwt_2d_torch(x):
    assert x.shape[0] % 2 == 0
    x_reshaped = x.view(x.shape[0] // 2, 2, x.shape[1])
    trend = (x_reshaped[:, 0, :] + x_reshaped[:, 1, :]) * (1.0 / math.sqrt(2))
    detail = (x_reshaped[:, 0, :] - x_reshaped[:, 1, :]) * (1.0 / math.sqrt(2))
    return torch.cat((trend, detail), dim=0)

def dwt_2d_inv_torch(x):
    assert x.shape[0] % 2 == 0
    half = x.shape[0] // 2
    a = (x[:half, :] + x[half:, :]) * (1.0 / math.sqrt(2))
    b = (x[:half, :] - x[half:, :]) * (1.0 / math.sqrt(2))
    
    out = torch.empty_like(x)
    out[0::2, :] = a
    out[1::2, :] = b
    return out

def dwt_2d_columns_torch(x):
    assert x.shape[1] % 2 == 0
    x_reshaped = x.view(x.shape[0], x.shape[1] // 2, 2)
    trend = (x_reshaped[:, :, 0] + x_reshaped[:, :, 1]) * (1.0 / math.sqrt(2))
    detail = (x_reshaped[:, :, 0] - x_reshaped[:, :, 1]) * (1.0 / math.sqrt(2))
    return torch.cat((trend, detail), dim=1)

def pad_columns_to_power_of_2(x, target_pow_2=None):
    H = x.shape[1]
    if target_pow_2 is None:
        if H < 1 or (H & (H - 1) == 0):
            target_pow_2 = H
        else:
            target_pow_2 = 1 << H.bit_length()
            
    if target_pow_2 == H:
        return x, target_pow_2
        
    pad_len = target_pow_2 - H
    padding = torch.zeros(x.shape[0], pad_len, dtype=x.dtype, device=x.device)
    return torch.cat((x, padding), dim=1), target_pow_2

def recursive_column_dwt(x, iters):
    dwt_act = x
    dwt_fin = []
    for _ in range(iters):
        dwt_act = dwt_2d_columns_torch(dwt_act)
        half = dwt_act.shape[1] // 2
        dwt_fin = [dwt_act[:, half:]] + dwt_fin
        dwt_act = dwt_act[:, :half]
        
    dwt_act = [dwt_act] + dwt_fin
    return torch.cat(dwt_act, dim=1)

def stamp_matmul_torch(x, y, stamp_size=64):
    S = x.shape[0]
    
    # Dynamic sequence length padding to 2048 (max evaluation seq_len)
    # Handles shorter/odd final wikitext segment steps robustly!
    if S < 2048:
        pad_len = 2048 - S
        last_row = x[-1:, :]
        padding = last_row.repeat(pad_len, 1)
        dwt_act = torch.cat((x, padding), dim=0)
    else:
        dwt_act = x
        
    assert dwt_act.shape[0] % 2 == 0
    iters = math.log2(dwt_act.shape[0]) - math.log2(stamp_size)
    
    dwt_fin = []
    for _ in range(int(iters)):
        dwt_act = dwt_2d_torch(dwt_act)
        half = dwt_act.shape[0] // 2
        dwt_fin = [dwt_act[half:, :]] + dwt_fin
        dwt_act = dwt_act[:half, :]
        
    dwt_act = [dwt_act] + dwt_fin
    dwt_act = torch.cat(dwt_act, dim=0)
    
    act_q8_64 = quant_torch(dwt_act[:stamp_size, :], 8, 1)
    act_q4_rest = quant_torch(dwt_act[stamp_size:, :], 4, 1)
    
    wgt_q8 = quant_torch(y, 8, 1)
    wgt_q4 = quant_torch(y, 4, 1)
    
    out_q8 = torch.matmul(act_q8_64, wgt_q8.t())
    out_q4 = torch.matmul(act_q4_rest, wgt_q4.t())
    
    out_mixed = torch.cat((out_q8, out_q4), dim=0)
    
    out_inv_dwt = out_mixed.clone()
    running_stamp = stamp_size
    for _ in range(int(iters)):
        running_stamp = running_stamp * 2
        tmp_out = dwt_2d_inv_torch(out_inv_dwt[:running_stamp, :])
        out_inv_dwt[:running_stamp, :] = tmp_out
        
    if S < 2048:
        return out_inv_dwt[:S, :]
    return out_inv_dwt

def stamp_matmul_mx_fp_torch(x, y, stamp_size=64):
    S = x.shape[0]
    
    # 1. Dynamic sequence length padding to 2048 (max evaluation seq_len)
    if S < 2048:
        pad_len = 2048 - S
        last_row = x[-1:, :]
        padding = last_row.repeat(pad_len, 1)
        dwt_act = torch.cat((x, padding), dim=0)
    else:
        dwt_act = x
        
    assert dwt_act.shape[0] % 2 == 0
    iters = math.log2(dwt_act.shape[0]) - math.log2(stamp_size)
    
    # 2. Forward DWT transform along activations sequence rows
    dwt_fin = []
    for _ in range(int(iters)):
        dwt_act = dwt_2d_torch(dwt_act)
        half = dwt_act.shape[0] // 2
        dwt_fin = [dwt_act[half:, :]] + dwt_fin
        dwt_act = dwt_act[:half, :]
        
    dwt_act = [dwt_act] + dwt_fin
    dwt_act = torch.cat(dwt_act, dim=0)
    
    # 3. Quantize activations (Standard Dynamic FP8 on Stamp, Block-wise MXFP4 on Rest!)
    # Act Stamp -> Standard Dynamic Scaled FP8 (no microscales / block-size!)
    act_q8_stamp = quant_fp8_torch(dwt_act[:stamp_size, :], 1)
    
    # Act Rest -> MXFP4 (e2m1 elements, e8m0 scales, BS=32)
    act_q4_rest, _, _, _ = quantize_mx_torch(
        dwt_act[stamp_size:, :], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    # 4. Quantize weights (Standard Dynamic FP8 on Stamp, Block-wise MXFP4 on Rest!)
    # Weight Stamp -> Standard Dynamic Scaled FP8 (no microscales / block-size!)
    wgt_q8 = quant_fp8_torch(y, 1)
    
    # Weight Rest -> MXFP4 (e2m1 elements, e8m0 scales, BS=32)
    wgt_q4, _, _, _ = quantize_mx_torch(
        y, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    # 5. Perform the two matmuls
    out_q8 = torch.matmul(act_q8_stamp, wgt_q8.t())
    out_q4 = torch.matmul(act_q4_rest, wgt_q4.t())
    
    # 6. Concatenate outputs
    out_mixed = torch.cat((out_q8, out_q4), dim=0)
    
    # 7. Undo DWT transform
    out_inv_dwt = out_mixed.clone()
    running_stamp = stamp_size
    for _ in range(int(iters)):
        running_stamp = running_stamp * 2
        tmp_out = dwt_2d_inv_torch(out_inv_dwt[:running_stamp, :])
        out_inv_dwt[:running_stamp, :] = tmp_out
        
    if S < 2048:
        return out_inv_dwt[:S, :]
    return out_inv_dwt

def stamp_matmul_bf16_fp_torch(x, y, stamp_size=64):
    S = x.shape[0]
    
    # 1. Dynamic sequence length padding to 2048 (max evaluation seq_len)
    if S < 2048:
        pad_len = 2048 - S
        last_row = x[-1:, :]
        padding = last_row.repeat(pad_len, 1)
        dwt_act = torch.cat((x, padding), dim=0)
    else:
        dwt_act = x
        
    assert dwt_act.shape[0] % 2 == 0
    iters = math.log2(dwt_act.shape[0]) - math.log2(stamp_size)
    
    # 2. Forward DWT transform along activations sequence rows
    dwt_fin = []
    for _ in range(int(iters)):
        dwt_act = dwt_2d_torch(dwt_act)
        half = dwt_act.shape[0] // 2
        dwt_fin = [dwt_act[half:, :]] + dwt_fin
        dwt_act = dwt_act[:half, :]
        
    dwt_act = [dwt_act] + dwt_fin
    dwt_act = torch.cat(dwt_act, dim=0)
    
    # 3. Represent activations (Lossless BF16 on Stamp, Block-wise MXFP4 on Rest!)
    # Act Stamp -> Native Lossless BF16! (No scaling / rounding!)
    act_q8_stamp = dwt_act[:stamp_size, :].to(torch.bfloat16)
    
    # Act Rest -> MXFP4 (e2m1 elements, e8m0 scales, BS=32)
    act_q4_rest, _, _, _ = quantize_mx_torch(
        dwt_act[stamp_size:, :], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    # 4. Represent weights (Lossless BF16 on Stamp, Block-wise MXFP4 on Rest!)
    # Weight Stamp -> Native Lossless BF16! (No scaling / rounding!)
    wgt_q8 = y.to(torch.bfloat16)
    
    # Weight Rest -> MXFP4 (e2m1 elements, e8m0 scales, BS=32)
    wgt_q4, _, _, _ = quantize_mx_torch(
        y, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    # 5. Perform the two matmuls
    out_q8 = torch.matmul(act_q8_stamp, wgt_q8.t())
    out_q4 = torch.matmul(act_q4_rest, wgt_q4.t())
    
    # 6. Concatenate outputs
    out_mixed = torch.cat((out_q8, out_q4), dim=0)
    
    # 7. Undo DWT transform
    out_inv_dwt = out_mixed.clone()
    running_stamp = stamp_size
    for _ in range(int(iters)):
        running_stamp = running_stamp * 2
        tmp_out = dwt_2d_inv_torch(out_inv_dwt[:running_stamp, :])
        out_inv_dwt[:running_stamp, :] = tmp_out
        
    if S < 2048:
        return out_inv_dwt[:S, :]
    return out_inv_dwt

def stamp_feat_matmul_bf16_fp_torch(x, y, stamp_size=64):
    S = x.shape[0]
    
    # 1. Dynamic sequence-length padding to 2048 along axis 0 (rows)
    if S < 2048:
        pad_len = 2048 - S
        last_row = x[-1:, :]
        padding = last_row.repeat(pad_len, 1)
        x_seq_padded = torch.cat((x, padding), dim=0)
    else:
        x_seq_padded = x
        
    # 2. Zero-pad hidden column dimension features axis 1 to next power of 2
    x_padded, target_pow_2 = pad_columns_to_power_of_2(x_seq_padded)
    y_padded, _ = pad_columns_to_power_of_2(y, target_pow_2)
    
    assert x_padded.shape[1] == target_pow_2
    assert y_padded.shape[1] == target_pow_2
    
    # 3. Calculate DWT iteration steps along feature dimension
    iters = math.log2(target_pow_2) - math.log2(stamp_size)
    iters_int = int(iters)
    
    # 4. Apply forward DWT transform along column features
    act_dwt = recursive_column_dwt(x_padded, iters_int)
    wgt_dwt = recursive_column_dwt(y_padded, iters_int)
    
    # 5. Represent activations (Lossless BF16 on Stamp, Block-wise MXFP4 on Rest!)
    act_q8_stamp = act_dwt[:, :stamp_size].to(torch.bfloat16)
    act_q4_rest, _, _, _ = quantize_mx_torch(
        act_dwt[:, stamp_size:], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    # 6. Represent weights (Lossless BF16 on Stamp, Block-wise MXFP4 on Rest!)
    wgt_q8 = wgt_dwt[:, :stamp_size].to(torch.bfloat16)
    wgt_q4, _, _, _ = quantize_mx_torch(
        wgt_dwt[:, stamp_size:], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    # 7. Perform the two matmuls directly in feature frequency space
    out_q8 = torch.matmul(act_q8_stamp, wgt_q8.t())
    out_q4 = torch.matmul(act_q4_rest, wgt_q4.t())
    
    # 8. Sum the outputs (natively reconstructs spatial domain!)
    out_spatial = out_q8 + out_q4
    
    # 9. Slice off sequence-length padding to recover original token sequence length S
    if S < 2048:
        return out_spatial[:S, :]
    return out_spatial

def ablate_seq_matmul_torch(x, y, stamp_size=64, strategy="random"):
    S = x.shape[0]
    
    # Target stamp size limit cap
    actual_stamp = min(stamp_size, S)
    
    if strategy == "random":
        g = torch.Generator(device=x.device)
        g.manual_seed(42)
        perm = torch.randperm(S, generator=g, device=x.device)
        lossless_indices = perm[:actual_stamp]
        compressed_indices = perm[actual_stamp:]
    elif strategy == "magnitude":
        # Measure row outlier magnitude by Lmax token activations
        row_magnitudes = torch.max(torch.abs(x), dim=1).values
        sorted_indices = torch.argsort(row_magnitudes, descending=True)
        lossless_indices = sorted_indices[:actual_stamp]
        compressed_indices = sorted_indices[actual_stamp:]
    else:
        raise ValueError(f"Unknown ablation strategy: {strategy}")
        
    # Represent activations (Lossless BF16 on Stamp, Block-wise MXFP4 on Rest!)
    act_bf16 = x[lossless_indices, :].to(torch.bfloat16)
    act_mxfp4, _, _, _ = quantize_mx_torch(
        x[compressed_indices, :], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    # Represent weights (Entire matrix BF16 or MXFP4)
    wgt_bf16 = y.to(torch.bfloat16)
    wgt_mxfp4, _, _, _ = quantize_mx_torch(
        y, 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    # Direct index matrix multiplications
    out = torch.empty(S, y.shape[0], dtype=x.dtype, device=x.device)
    
    if lossless_indices.numel() > 0:
        out_q8 = torch.matmul(act_bf16, wgt_bf16.t())
        out[lossless_indices, :] = out_q8.to(x.dtype)
        
    if compressed_indices.numel() > 0:
        out_q4 = torch.matmul(act_mxfp4, wgt_mxfp4.t())
        out[compressed_indices, :] = out_q4.to(x.dtype)
        
    return out

def ablate_feat_matmul_torch(x, y, stamp_size=64, strategy="random"):
    H = x.shape[1]
    
    actual_stamp = min(stamp_size, H)
    
    if strategy == "random":
        g = torch.Generator(device=x.device)
        g.manual_seed(42)
        perm = torch.randperm(H, generator=g, device=x.device)
        lossless_indices = perm[:actual_stamp]
        compressed_indices = perm[actual_stamp:]
    elif strategy == "magnitude":
        # Measure weight columns outlier magnitude directly in static weights
        col_magnitudes = torch.max(torch.abs(y), dim=0).values
        sorted_indices = torch.argsort(col_magnitudes, descending=True)
        lossless_indices = sorted_indices[:actual_stamp]
        compressed_indices = sorted_indices[actual_stamp:]
    else:
        raise ValueError(f"Unknown ablation strategy: {strategy}")
        
    # Represent activations (Lossless BF16 on Stamp, Block-wise MXFP4 on Rest!)
    act_bf16 = x[:, lossless_indices].to(torch.bfloat16)
    act_mxfp4, _, _, _ = quantize_mx_torch(
        x[:, compressed_indices], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    # Represent weights (Lossless BF16 on Stamp, Block-wise MXFP4 on Rest!)
    wgt_bf16 = y[:, lossless_indices].to(torch.bfloat16)
    wgt_mxfp4, _, _, _ = quantize_mx_torch(
        y[:, compressed_indices], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
    )
    
    # Direct sum matrix multiplications
    out_q8 = torch.matmul(act_bf16, wgt_bf16.t())
    out_q4 = torch.matmul(act_mxfp4, wgt_mxfp4.t())
    
    return (out_q8 + out_q4).to(x.dtype)

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

def apply_asymmetric_torch(x, gammas):
    y = x.clone().to(torch.float32)
    norm = 1.0 / np.sqrt(2.0)
    for stage in range(len(gammas)):
        stride = 2 ** stage
        # Reshape to block chunks of size 2 * stride
        y_view = y.reshape(-1, 1, 2, stride)
        x1 = y_view[:, :, 0, :]
        x2 = y_view[:, :, 1, :]
        new_x1 = (x1 + x2) * norm
        new_x2 = (x1 - x2) * (norm * gammas[stage])
        y_view = y_view.clone()
        y_view[:, :, 0, :] = new_x1
        y_view[:, :, 1, :] = new_x2
        y = y_view.reshape(x.shape)
    return y

def inverse_asymmetric_torch(y, gammas):
    x = y.clone().to(torch.float32)
    norm = 1.0 / np.sqrt(2.0)
    for stage in reversed(range(len(gammas))):
        stride = 2 ** stage
        x_view = x.reshape(-1, 1, 2, stride)
        y1 = x_view[:, :, 0, :]
        y2 = x_view[:, :, 1, :]
        orig_x1 = (y1 + y2 / gammas[stage]) * norm
        orig_x2 = (y1 - y2 / gammas[stage]) * norm
        x_view = x_view.clone()
        x_view[:, :, 0, :] = orig_x1
        x_view[:, :, 1, :] = orig_x2
        x = x_view.reshape(y.shape)
    return x

def apply_rotation_torch(x, thetas):
    y = x.clone().to(torch.float32)
    for stage in range(len(thetas)):
        stride = 2 ** stage
        y_view = y.reshape(-1, 1, 2, stride)
        x1 = y_view[:, :, 0, :]
        x2 = y_view[:, :, 1, :]
        c = torch.cos(thetas[stage])
        s = torch.sin(thetas[stage])
        new_x1 = x1 * c + x2 * s
        new_x2 = -x1 * s + x2 * c
        y_view = y_view.clone()
        y_view[:, :, 0, :] = new_x1
        y_view[:, :, 1, :] = new_x2
        y = y_view.reshape(x.shape)
    return y

def inverse_rotation_torch(y, thetas):
    x = y.clone().to(torch.float32)
    for stage in reversed(range(len(thetas))):
        stride = 2 ** stage
        x_view = x.reshape(-1, 1, 2, stride)
        y1 = x_view[:, :, 0, :]
        y2 = x_view[:, :, 1, :]
        c = torch.cos(thetas[stage])
        s = torch.sin(thetas[stage])
        orig_x1 = y1 * c - y2 * s
        orig_x2 = y1 * s + y2 * c
        x_view = x_view.clone()
        x_view[:, :, 0, :] = orig_x1
        x_view[:, :, 1, :] = orig_x2
        x = x_view.reshape(y.shape)
    return x

def get_custom_rotation_matrices_torch(name, device="cuda", dtype=torch.float32):
    identity = torch.eye(32, device=device, dtype=torch.float32)
    
    if name == "fp4_tilted_2s":
        gammas = torch.cat([torch.tensor(crm.FP4_GAMMAS_2, device=device), torch.tensor([1.0, 1.0, 1.0], device=device)])
        W = apply_asymmetric_torch(identity, gammas)
        W_inv = inverse_asymmetric_torch(identity, gammas)
    elif name == "fp4_tilted_3s":
        gammas = torch.cat([torch.tensor(crm.FP4_GAMMAS_3, device=device), torch.tensor([1.0, 1.0], device=device)])
        W = apply_asymmetric_torch(identity, gammas)
        W_inv = inverse_asymmetric_torch(identity, gammas)
    elif name == "fp4_tilted_5s":
        gammas = torch.tensor(crm.FP4_GAMMAS_5, device=device)
        W = apply_asymmetric_torch(identity, gammas)
        W_inv = inverse_asymmetric_torch(identity, gammas)
    elif name == "fp4_rot_2s":
        thetas = torch.cat([torch.tensor(crm.FP4_THETAS_2, device=device), torch.tensor([np.pi/4, np.pi/4, np.pi/4], device=device)])
        W = apply_rotation_torch(identity, thetas)
        W_inv = inverse_rotation_torch(identity, thetas)
    elif name == "fp4_rot_3s":
        thetas = torch.cat([torch.tensor(crm.FP4_THETAS_3, device=device), torch.tensor([np.pi/4, np.pi/4], device=device)])
        W = apply_rotation_torch(identity, thetas)
        W_inv = inverse_rotation_torch(identity, thetas)
    elif name == "fp4_rot_5s":
        thetas = torch.tensor(crm.FP4_THETAS_5, device=device)
        W = apply_rotation_torch(identity, thetas)
        W_inv = inverse_rotation_torch(identity, thetas)
    elif name == "int4_tilted_2s":
        gammas = torch.cat([torch.tensor(crm.INT4_GAMMAS_2, device=device), torch.tensor([1.0, 1.0, 1.0], device=device)])
        W = apply_asymmetric_torch(identity, gammas)
        W_inv = inverse_asymmetric_torch(identity, gammas)
    elif name == "int4_tilted_3s":
        gammas = torch.cat([torch.tensor(crm.INT4_GAMMAS_3, device=device), torch.tensor([1.0, 1.0], device=device)])
        W = apply_asymmetric_torch(identity, gammas)
        W_inv = inverse_asymmetric_torch(identity, gammas)
    elif name == "int4_tilted_5s":
        gammas = torch.tensor(crm.INT4_GAMMAS_5, device=device)
        W = apply_asymmetric_torch(identity, gammas)
        W_inv = inverse_asymmetric_torch(identity, gammas)
    elif name == "int4_rot_2s":
        thetas = torch.cat([torch.tensor(crm.INT4_THETAS_2, device=device), torch.tensor([np.pi/4, np.pi/4, np.pi/4], device=device)])
        W = apply_rotation_torch(identity, thetas)
        W_inv = inverse_rotation_torch(identity, thetas)
    elif name == "int4_rot_3s":
        thetas = torch.cat([torch.tensor(crm.INT4_THETAS_3, device=device), torch.tensor([np.pi/4, np.pi/4], device=device)])
        W = apply_rotation_torch(identity, thetas)
        W_inv = inverse_rotation_torch(identity, thetas)
    elif name == "int4_rot_5s":
        thetas = torch.tensor(crm.INT4_THETAS_5, device=device)
        W = apply_rotation_torch(identity, thetas)
        W_inv = inverse_rotation_torch(identity, thetas)
    elif name == "int4_ks_320p":
        W = torch.tensor(crm.W_DENSE_320, device=device, dtype=torch.float32)
        W_inv = torch.linalg.inv(W)
    elif name == "int4_rot_80p":
        W = torch.tensor(crm.W_DENSE_80, device=device, dtype=torch.float32)
        W_inv = W.T
    else:
        raise ValueError(f"Unknown custom rotation: {name}")
        
    return W.to(dtype), W_inv.to(dtype)

def had_mod_torch(x, w, had_size=256, seed=42, custom_rotation=None):
    device = x.device
    dtype = x.dtype
    
    if custom_rotation is not None:
        W, W_inv = get_custom_rotation_matrices_torch(custom_rotation, device=device, dtype=torch.float32)
        x_rot = apply_hadamard_torch(x.float(), W, is_lhs=True, apply_scale=False).to(dtype)
        w_rot = apply_hadamard_torch(w.float(), W_inv.t(), is_lhs=True, apply_scale=False).to(dtype)
        return x_rot, w_rot
        
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

def quantize_mx_torch(x, block_size, elem_format="e2m1", scale_format="e4m3", prevent_zero=True, four_over_six=False, use_hierarchical=False, clip_percentile=None, rounding=None):
    init_shape = x.shape
    
    if clip_percentile is not None:
        x_float = x.float()
        abs_x = torch.abs(x_float)
        numel = abs_x.numel()
        max_sample = 1000000
        if numel > max_sample:
            indices = torch.arange(0, max_sample, device=x.device) * (numel - 1) // (max_sample - 1)
            sample = abs_x.view(-1)[indices]
            threshold = torch.quantile(sample, clip_percentile)
        else:
            threshold = torch.quantile(abs_x, clip_percentile)
        x = torch.clamp(x, min=-threshold.to(x.dtype), max=threshold.to(x.dtype))
        
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
        
    foc_str = str(four_over_six).lower()
    
    if foc_str in ("false", "none", "0"):
        max_abs = torch.max(torch.abs(x_reshaped), dim=-1, keepdim=True).values
        max_abs = torch.clamp(max_abs, min=1e-7)
        if clip_percentile is not None:
            max_abs = max_abs * clip_percentile
        
        raw_scale = max_abs / max_rep
        
        if rounding is None:
            effective_rounding = "ceil" if scale_format == "e8m0" else "round"
        else:
            effective_rounding = rounding
            
        if effective_rounding == "squant":
            effective_scale_rounding = "ceil" if scale_format == "e8m0" else "round"
        else:
            effective_scale_rounding = effective_rounding
            
        scaling_factor = quantize_scale_simulate(raw_scale, format=scale_format, prevent_zero=prevent_zero, rounding=effective_scale_rounding)
        
        # Safe division to mimic JAX behavior without NaN propagation
        safe_scale = torch.where(scaling_factor != 0.0, scaling_factor, torch.ones_like(scaling_factor))
        scaled = x_reshaped / safe_scale
        scaled = torch.where(scaling_factor != 0.0, scaled, torch.zeros_like(scaled))
        
        if effective_rounding == "squant":
            fp4_vals = get_signed_fp4_values(elem_format, device=x.device)
            idx = torch.searchsorted(fp4_vals, scaled)
            idx = torch.clamp(idx, min=1, max=len(fp4_vals) - 1)
            left_val = fp4_vals[idx - 1]
            right_val = fp4_vals[idx]
            dist_left = torch.abs(scaled - left_val)
            dist_right = torch.abs(scaled - right_val)
            mask = dist_left < dist_right
            q_nearest = torch.where(mask, left_val, right_val)
            q_alt = torch.where(mask, right_val, left_val)
            
            delta = scaling_factor * (q_alt - q_nearest)
            E = torch.sum(scaling_factor * (scaled - q_nearest), dim=-1, keepdim=True)
            
            delta_flat = delta.reshape(delta.shape[0], -1)
            sort_idx = torch.argsort(torch.abs(delta_flat), dim=-1)
            delta_sorted = torch.gather(delta_flat, dim=-1, index=sort_idx)
            delta_cumsum = torch.cumsum(delta_sorted, dim=-1)
            
            E_flat = E.reshape(E.shape[0], 1)
            E_cum = E_flat - delta_cumsum
            E_zero = E_flat
            E_options = torch.cat([E_zero, E_cum], dim=-1)
            
            r_star = torch.argmin(torch.abs(E_options), dim=-1, keepdim=True)
            
            inv_sort_idx = torch.argsort(sort_idx, dim=-1)
            flip_mask_flat = inv_sort_idx < r_star
            flip_mask = flip_mask_flat.reshape(scaled.shape)
            
            quant = torch.where(flip_mask, q_alt, q_nearest)
            dequant = quant * scaling_factor
        else:
            quant = snap_fp4(scaled)
            dequant = quant * scaling_factor
        use_4 = torch.zeros_like(scaling_factor, dtype=torch.bool)
    else:
        max_abs = torch.max(torch.abs(x_reshaped), dim=-1, keepdim=True).values
        max_abs = torch.clamp(max_abs, min=1e-7)
        if clip_percentile is not None:
            max_abs = max_abs * clip_percentile
        
        if "e1m2" in elem_format:
            max_fp4 = 7.0
            alt_max_fp4 = 5.0
        else: # e2m1
            max_fp4 = 6.0
            alt_max_fp4 = 4.0
            
        if foc_str in ("floor_ceil", "foc"):
            # Floor-or-Ceil bracketing strategy (strictly exponential style E8M0 bracketing)
            ideal_scale = max_abs / max_fp4
            log_ideal = torch.log2(ideal_scale)
            
            exp_floor = torch.floor(log_ideal)
            exp_floor = torch.clamp(exp_floor, -127.0, 128.0)
            
            exp_ceil = torch.ceil(log_ideal)
            exp_ceil = torch.clamp(exp_ceil, -127.0, 128.0)
            
            scale_4 = torch.pow(2.0, exp_floor)
            scale_6 = torch.pow(2.0, exp_ceil)
        else:
            # Classic 4o6 mapping strategy (uses global static rounding parameter)
            s4_raw = max_abs / alt_max_fp4
            s6_raw = max_abs / max_fp4
            
            if rounding is None:
                effective_rounding = "ceil" if scale_format == "e8m0" else "round"
            else:
                effective_rounding = rounding
                
            scale_4 = quantize_scale_simulate(s4_raw, format=scale_format, prevent_zero=prevent_zero, rounding=effective_rounding)
            scale_6 = quantize_scale_simulate(s6_raw, format=scale_format, prevent_zero=prevent_zero, rounding=effective_rounding)
            
        dequant, quant, scaling_factor, use_4 = mse_select_quant(x_reshaped, scale_4, scale_6, snap_fp4)
        
    dequant = dequant.reshape(init_shape)
    if use_hierarchical:
        dequant = dequant * channel_scale
        
    return dequant.to(x.dtype), quant.reshape(init_shape), scaling_factor, use_4

def FP4_quant_torch(x, block_size, prevent_zero=True, four_over_six=False, use_hierarchical=False, scale_format="e4m3", format="e2m1"):
    return quantize_mx_torch(x, block_size, elem_format=format, scale_format=scale_format, prevent_zero=prevent_zero, four_over_six=four_over_six, use_hierarchical=use_hierarchical)

class TorchMXLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, block_size=32, prevent_zero=True, four_over_six=False,
                 elem_format="e2m1", scale_format="e4m3", use_hierarchical=False, hadamard_size=0, hadamard_seed=42,
                 clip_percentile=None, rounding=None, custom_rotation=None):
        super().__init__(in_features, out_features, bias)
        self.block_size = block_size
        self.prevent_zero = prevent_zero
        self.four_over_six = four_over_six
        self.use_hierarchical = use_hierarchical
        self.elem_format = elem_format
        self.scale_format = scale_format
            
        # Parse potential mixed formats (e.g. "e1m2,e2m1" -> wgt="e1m2", act="e2m1")
        if "," in elem_format:
            self.wgt_format, self.act_format = elem_format.split(",")
        else:
            self.wgt_format = self.act_format = elem_format

        self.hadamard_size = hadamard_size
        self.hadamard_seed = hadamard_seed
        self.clip_percentile = clip_percentile
        self.rounding = rounding
        self.custom_rotation = custom_rotation
        
    def forward(self, input):
        w = self.weight
        x = input
        
        if self.hadamard_size > 0 or self.custom_rotation is not None:
            x_rot, w_rot = had_mod_torch(x, w, had_size=self.hadamard_size, seed=self.hadamard_seed, custom_rotation=self.custom_rotation)
        else:
            x_rot, w_rot = x, w
            
        q_weight, _, _, _ = quantize_mx_torch(w_rot, self.block_size, elem_format=self.wgt_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
        q_input, _, _, _ = quantize_mx_torch(x_rot, self.block_size, elem_format=self.act_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
        
        return F.linear(q_input, q_weight, self.bias)

class TorchSeqMeanSubLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, group_size=32, block_size=32, prevent_zero=True, four_over_six=False,
                 elem_format="e2m1", scale_format="e4m3", use_hierarchical=False, hadamard_size=0, hadamard_seed=42,
                 clip_percentile=None, rounding=None, custom_rotation=None):
        super().__init__(in_features, out_features, bias)
        self.group_size = group_size
        self.block_size = block_size
        self.prevent_zero = prevent_zero
        self.four_over_six = four_over_six
        self.use_hierarchical = use_hierarchical
        self.elem_format = elem_format
        self.scale_format = scale_format
        self.hadamard_size = hadamard_size
        self.hadamard_seed = hadamard_seed
        self.clip_percentile = clip_percentile
        self.rounding = rounding
        self.custom_rotation = custom_rotation

    def forward(self, input):
        init_shape = input.shape
        x_2d = input.reshape(-1, init_shape[-1])
        w = self.weight
        
        # 1. Pad sequence dimension to multiple of group_size
        S = x_2d.shape[0]
        G = self.group_size
        if S % G != 0:
            pad_len = G - (S % G)
            last_row = x_2d[-1:, :]
            padding = last_row.repeat(pad_len, 1)
            x_padded = torch.cat((x_2d, padding), dim=0)
        else:
            x_padded = x_2d
            
        S_pad = x_padded.shape[0]
        num_groups = S_pad // G
        
        # 2. Reshape to [num_groups, G, H] to compute mean per group
        H = x_padded.shape[1]
        x_reshaped = x_padded.view(num_groups, G, H)
        mean_val = x_reshaped.mean(dim=1, keepdim=True) # [num_groups, 1, H]
        
        # 3. Center activations
        x_centered = x_reshaped - mean_val # [num_groups, G, H]
        x_centered_2d = x_centered.view(S_pad, H)
        
        # 4. Quantize centered activations and weights to MXFP4
        q_weight, _, _, _ = quantize_mx_torch(w, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
        q_act_centered, _, _, _ = quantize_mx_torch(x_centered_2d, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
        
        # 5. Centered matmul
        out_centered_2d = F.linear(q_act_centered, q_weight, bias=None) # [S_pad, O]
        
        # 6. High-precision correction: mean_val * w^T
        w_bf16 = w.to(torch.bfloat16)
        correction = torch.matmul(mean_val.to(torch.bfloat16), w_bf16.t()) # [num_groups, 1, O]
        
        # 7. Add correction
        out_centered_reshaped = out_centered_2d.view(num_groups, G, -1)
        out_recon_reshaped = out_centered_reshaped + correction
        out_recon_2d = out_recon_reshaped.view(S_pad, -1)
        
        # 8. Unpad and restore shape
        if S % G != 0:
            out_final_2d = out_recon_2d[:S, :]
        else:
            out_final_2d = out_recon_2d
            
        out_3d = out_final_2d.reshape(init_shape[:-1] + (self.out_features,))
        if self.bias is not None:
            out_3d = out_3d + self.bias
        return out_3d

class TorchSVDOutlierLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, k=1, block_size=32, prevent_zero=True, four_over_six=False,
                 elem_format="e2m1", scale_format="e4m3", use_hierarchical=False, hadamard_size=0, hadamard_seed=42,
                 clip_percentile=None, rounding=None, custom_rotation=None):
        super().__init__(in_features, out_features, bias)
        self.k = k
        self.block_size = block_size
        self.prevent_zero = prevent_zero
        self.four_over_six = four_over_six
        self.use_hierarchical = use_hierarchical
        self.elem_format = elem_format
        self.scale_format = scale_format
        self.hadamard_size = hadamard_size
        self.hadamard_seed = hadamard_seed
        self.clip_percentile = clip_percentile
        self.rounding = rounding
        self.custom_rotation = custom_rotation
        
        self.register_buffer("V_k", None)     # Outlier subspace [H, K]
        self.register_buffer("W_proj", None)  # Pre-projected weights [O, K]
        self.is_calibrated = False

    def forward(self, input):
        device = input.device
        dtype = input.dtype
        init_shape = input.shape
        x_2d = input.reshape(-1, init_shape[-1])
        w = self.weight
        
        # 1. Calibration Pass (first forward step)
        if not self.is_calibrated:
            with torch.no_grad():
                x_f32 = x_2d.float()
                U, S_vals, Vt = torch.linalg.svd(x_f32, full_matrices=False)
                
                # self.V_k: [H, K]
                self.V_k = Vt[:self.k, :].t().to(dtype)
                
                # self.W_proj = W * V_k  -> [O, K]
                self.W_proj = torch.matmul(w, self.V_k)
                
            self.is_calibrated = True
            
        # 2. Split Matmul Execution
        # projections = X * V_k  -> [S, K]
        projections = torch.matmul(x_2d, self.V_k)
        
        # Y_heavy = projections * W_proj^T  -> [S, O]
        y_heavy_2d = torch.matmul(projections, self.W_proj.t())
        
        # X_heavy = projections * V_k^T  -> [S, H]
        x_heavy_2d = torch.matmul(projections, self.V_k.t())
        
        # X_light = X - X_heavy  -> [S, H]
        x_light_2d = x_2d - x_heavy_2d
        
        # Quantize X_light and W to MXFP4
        q_weight, _, _, _ = quantize_mx_torch(w, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
        q_act_light, _, _, _ = quantize_mx_torch(x_light_2d, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
        
        # Y_light = X_light_quant * W_quant^T  -> [S, O]
        y_light_2d = F.linear(q_act_light, q_weight, bias=None)
        
        # Sum outputs
        out_2d = y_light_2d + y_heavy_2d
        
        out_3d = out_2d.reshape(init_shape[:-1] + (self.out_features,))
        if self.bias is not None:
            out_3d = out_3d + self.bias
            
        return out_3d

class TorchPermutedMLP(nn.Module):
    def __init__(self, original_mlp, rms_gamma, block_size=32, prevent_zero=True, four_over_six=False,
                 elem_format="e2m1", scale_format="e4m3", use_hierarchical=False, clip_percentile=None, rounding=None,
                 stamp_size=0, stamp_size_dff=0, search_box_size=0, search_box_size_dff=0):
        super().__init__()
        self.block_size = block_size
        self.prevent_zero = prevent_zero
        self.four_over_six = four_over_six
        self.use_hierarchical = use_hierarchical
        self.elem_format = elem_format
        self.scale_format = scale_format
        self.clip_percentile = clip_percentile
        self.rounding = rounding
        self.stamp_size = stamp_size
        self.stamp_size_dff = stamp_size_dff
        self.search_box_size = search_box_size
        self.search_box_size_dff = search_box_size_dff
        
        # Keep references to original layers
        w_gate = original_mlp.gate_proj.weight.data # [D_ff, C]
        w_up = original_mlp.up_proj.weight.data     # [D_ff, C]
        w_down = original_mlp.down_proj.weight.data # [O, D_ff]
        
        device = w_gate.device
        dtype = w_gate.dtype
        
        # 1. Sort contracting dimension C based on rms_gamma
        gamma_abs = torch.abs(rms_gamma.data)
        C_permutation = torch.argsort(gamma_abs, descending=True)
        self.register_buffer("C_permutation", C_permutation)
        
        # Permute columns (axis 1) of gate/up weights
        w_gate_perm_C = w_gate[:, C_permutation]
        w_up_perm_C = w_up[:, C_permutation]
        
        # 2. Sort intermediate dimension D_ff
        # Joint score: L2 norm of rows (dim 1) of gate and up weights
        gate_row_norms = torch.linalg.norm(w_gate_perm_C, dim=1) # [D_ff]
        up_row_norms = torch.linalg.norm(w_up_perm_C, dim=1)     # [D_ff]
        
        dff_scores = gate_row_norms * up_row_norms
        Dff_permutation = torch.argsort(dff_scores, descending=True)
        self.register_buffer("Dff_permutation", Dff_permutation)
        
        # Apply Dff permutation:
        # Permute output dimension of gate/up (axis 0)
        w_gate_final = w_gate_perm_C[Dff_permutation, :]
        w_up_final = w_up_perm_C[Dff_permutation, :]
        
        # Permute input dimension of down (axis 1)
        w_down_final = w_down[:, Dff_permutation]
        
        self.w_gate_perm = nn.Parameter(w_gate_final, requires_grad=False)
        self.w_up_perm = nn.Parameter(w_up_final, requires_grad=False)
        self.w_down_perm = nn.Parameter(w_down_final, requires_grad=False)
        
        self.act_fn = original_mlp.act_fn

    def forward(self, x):
        init_shape = x.shape
        x_2d = x.reshape(-1, init_shape[-1])
        device = x.device
        dtype = x.dtype
        
        # 1. Permute incoming activations
        x_perm = x_2d[:, self.C_permutation]
        
        # 2. Gate & Up Projections
        if self.stamp_size > 0:
            if self.search_box_size > 0:
                # Dynamic selection strictly within the search box (e.g. first 256 channels)
                x_box = x_perm[:, :self.search_box_size]
                box_energy = torch.sum(x_box ** 2, dim=0)
                top_box_idx = torch.argsort(box_energy, descending=True)[:self.stamp_size]
                stamp_cols = top_box_idx
            else:
                # Static split stamp
                stamp_cols = torch.arange(self.stamp_size, device=device)
                
            mask = torch.ones(x_perm.shape[1], dtype=torch.bool, device=device)
            mask[stamp_cols] = False
            rem_cols = torch.where(mask)[0]
            
            x_heavy = x_perm[:, stamp_cols]
            w_gate_heavy = self.w_gate_perm[:, stamp_cols]
            w_up_heavy = self.w_up_perm[:, stamp_cols]
            
            h_gate_heavy = F.linear(x_heavy, w_gate_heavy, bias=None)
            h_up_heavy = F.linear(x_heavy, w_up_heavy, bias=None)
            
            x_light = x_perm[:, rem_cols]
            w_gate_light = self.w_gate_perm[:, rem_cols]
            w_up_light = self.w_up_perm[:, rem_cols]
            
            # Pad light path to multiple of 32 for MXFP4
            rem_dim = x_light.shape[1]
            pad_len = (32 - (rem_dim % 32)) % 32
            if pad_len > 0:
                x_light_padded = torch.cat([x_light, torch.zeros((x_light.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
                w_gate_light_padded = torch.cat([w_gate_light, torch.zeros((w_gate_light.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
                w_up_light_padded = torch.cat([w_up_light, torch.zeros((w_up_light.shape[0], pad_len), device=device, dtype=dtype)], dim=1)
            else:
                x_light_padded = x_light
                w_gate_light_padded = w_gate_light
                w_up_light_padded = w_up_light
                
            q_x_light, _, _, _ = quantize_mx_torch(x_light_padded, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
            q_w_gate_light, _, _, _ = quantize_mx_torch(w_gate_light_padded, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
            q_w_up_light, _, _, _ = quantize_mx_torch(w_up_light_padded, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
            
            if pad_len > 0:
                q_x_light = q_x_light[:, :rem_dim]
                q_w_gate_light = q_w_gate_light[:, :rem_dim]
                q_w_up_light = q_w_up_light[:, :rem_dim]
                
            h_gate_light = F.linear(q_x_light, q_w_gate_light, bias=None)
            h_up_light = F.linear(q_x_light, q_w_up_light, bias=None)
            
            h_gate = h_gate_heavy + h_gate_light
            h_up = h_up_heavy + h_up_light
        else:
            q_x, _, _, _ = quantize_mx_torch(x_perm, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
            q_w_gate, _, _, _ = quantize_mx_torch(self.w_gate_perm, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
            q_w_up, _, _, _ = quantize_mx_torch(self.w_up_perm, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
            
            h_gate = F.linear(q_x, q_w_gate, bias=None)
            h_up = F.linear(q_x, q_w_up, bias=None)
            
        # 3. SwiGLU Element-wise Activation
        h_inter = self.act_fn(h_gate) * h_up
        
        # 4. Down Projection
        if self.stamp_size_dff > 0:
            if self.search_box_size_dff > 0:
                # Dynamic selection strictly within the intermediate search box (e.g. first 256 channels)
                h_box = h_inter[:, :self.search_box_size_dff]
                box_energy_dff = torch.sum(h_box ** 2, dim=0)
                top_box_idx_dff = torch.argsort(box_energy_dff, descending=True)[:self.stamp_size_dff]
                stamp_cols_dff = top_box_idx_dff
            else:
                # Static split stamp
                stamp_cols_dff = torch.arange(self.stamp_size_dff, device=device)
                
            mask_dff = torch.ones(h_inter.shape[1], dtype=torch.bool, device=device)
            mask_dff[stamp_cols_dff] = False
            rem_cols_dff = torch.where(mask_dff)[0]
            
            h_heavy_dff = h_inter[:, stamp_cols_dff]
            w_down_heavy = self.w_down_perm[:, stamp_cols_dff]
            
            y_heavy = F.linear(h_heavy_dff, w_down_heavy, bias=None)
            
            h_light_dff = h_inter[:, rem_cols_dff]
            w_down_light = self.w_down_perm[:, rem_cols_dff]
            
            # Pad light path to multiple of 32 for MXFP4
            rem_dim_dff = h_light_dff.shape[1]
            pad_len_dff = (32 - (rem_dim_dff % 32)) % 32
            if pad_len_dff > 0:
                h_light_dff_padded = torch.cat([h_light_dff, torch.zeros((h_light_dff.shape[0], pad_len_dff), device=device, dtype=dtype)], dim=1)
                w_down_light_padded = torch.cat([w_down_light, torch.zeros((w_down_light.shape[0], pad_len_dff), device=device, dtype=dtype)], dim=1)
            else:
                h_light_dff_padded = h_light_dff
                w_down_light_padded = w_down_light
                
            q_h_light_dff, _, _, _ = quantize_mx_torch(h_light_dff_padded, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
            q_w_down_light, _, _, _ = quantize_mx_torch(w_down_light_padded, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
            
            if pad_len_dff > 0:
                q_h_light_dff = q_h_light_dff[:, :rem_dim_dff]
                q_w_down_light = q_w_down_light[:, :rem_dim_dff]
                
            y_light = F.linear(q_h_light_dff, q_w_down_light, bias=None)
            
            y = y_heavy + y_light
        else:
            q_h_inter, _, _, _ = quantize_mx_torch(h_inter, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
            q_w_down, _, _, _ = quantize_mx_torch(self.w_down_perm, self.block_size, elem_format=self.elem_format, scale_format=self.scale_format, prevent_zero=self.prevent_zero, four_over_six=self.four_over_six, use_hierarchical=self.use_hierarchical, clip_percentile=self.clip_percentile, rounding=self.rounding)
            
            y = F.linear(q_h_inter, q_w_down, bias=None)
            
        return y.reshape(init_shape[:-1] + (y.shape[-1],))

class TorchStampLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, stamp_size=64):
        super().__init__(in_features, out_features, bias)
        self.stamp_size = stamp_size
        
    def forward(self, input):
        init_shape = input.shape
        x_2d = input.reshape(-1, init_shape[-1])
        
        out_2d = stamp_matmul_torch(x_2d, self.weight, self.stamp_size)
        
        out_shape = init_shape[:-1] + (self.out_features,)
        out_3d = out_2d.reshape(out_shape)
        
        if self.bias is not None:
            out_3d = out_3d + self.bias
            
        return out_3d

class TorchStampMXLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, stamp_size=64):
        super().__init__(in_features, out_features, bias)
        self.stamp_size = stamp_size
        
    def forward(self, input):
        init_shape = input.shape
        x_2d = input.reshape(-1, init_shape[-1])
        
        out_2d = stamp_matmul_mx_fp_torch(x_2d, self.weight, self.stamp_size)
        
        out_shape = init_shape[:-1] + (self.out_features,)
        out_3d = out_2d.reshape(out_shape)
        
        if self.bias is not None:
            out_3d = out_3d + self.bias
            
        return out_3d

class TorchStampBF16Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, stamp_size=64):
        super().__init__(in_features, out_features, bias)
        self.stamp_size = stamp_size
        
    def forward(self, input):
        init_shape = input.shape
        x_2d = input.reshape(-1, init_shape[-1])
        
        out_2d = stamp_matmul_bf16_fp_torch(x_2d, self.weight, self.stamp_size)
        
        out_shape = init_shape[:-1] + (self.out_features,)
        out_3d = out_2d.reshape(out_shape)
        
        if self.bias is not None:
            out_3d = out_3d + self.bias
            
        return out_3d

class TorchStampFeatLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, stamp_size=64):
        super().__init__(in_features, out_features, bias)
        self.stamp_size = stamp_size
        
    def forward(self, input):
        init_shape = input.shape
        x_2d = input.reshape(-1, init_shape[-1])
        
        out_2d = stamp_feat_matmul_bf16_fp_torch(x_2d, self.weight, self.stamp_size)
        
        out_shape = init_shape[:-1] + (self.out_features,)
        out_3d = out_2d.reshape(out_shape)
        
        if self.bias is not None:
            out_3d = out_3d + self.bias
            
        return out_3d

class TorchAblateSeqLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, stamp_size=64, strategy="random"):
        super().__init__(in_features, out_features, bias)
        self.stamp_size = stamp_size
        self.strategy = strategy
        
    def forward(self, input):
        init_shape = input.shape
        x_2d = input.reshape(-1, init_shape[-1])
        
        out_2d = ablate_seq_matmul_torch(x_2d, self.weight, self.stamp_size, self.strategy)
        
        out_shape = init_shape[:-1] + (self.out_features,)
        out_3d = out_2d.reshape(out_shape)
        
        if self.bias is not None:
            out_3d = out_3d + self.bias
            
        return out_3d

class TorchAblateFeatLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, stamp_size=64, strategy="random"):
        super().__init__(in_features, out_features, bias)
        self.stamp_size = stamp_size
        self.strategy = strategy
        
    def forward(self, input):
        init_shape = input.shape
        x_2d = input.reshape(-1, init_shape[-1])
        
        out_2d = ablate_feat_matmul_torch(x_2d, self.weight, self.stamp_size, self.strategy)
        
        out_shape = init_shape[:-1] + (self.out_features,)
        out_3d = out_2d.reshape(out_shape)
        
        if self.bias is not None:
            out_3d = out_3d + self.bias
            
        return out_3d

class TorchTransformSeqLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, stamp_size=64, transform_name="dct"):
        super().__init__(in_features, out_features, bias)
        self.stamp_size = stamp_size
        self.transform_name = transform_name
        
    def forward(self, input):
        init_shape = input.shape
        x_2d = input.reshape(-1, init_shape[-1])
        
        if self.transform_name == "svd":
            out_2d = svd_matmul_bf16_fp_torch(x_2d, self.weight, self.stamp_size)
        else:
            out_2d = transform_seq_matmul_bf16_fp_torch(x_2d, self.weight, self.stamp_size, self.transform_name)
            
        out_shape = init_shape[:-1] + (self.out_features,)
        out_3d = out_2d.reshape(out_shape)
        
        if self.bias is not None:
            out_3d = out_3d + self.bias
        return out_3d

class TorchTransformFeatLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, stamp_size=64, transform_name="dct", pattern="contiguous"):
        super().__init__(in_features, out_features, bias)
        self.stamp_size = stamp_size
        self.transform_name = transform_name
        self.pattern = pattern
        
        H = in_features
        is_pow2 = (H & (H - 1) == 0) and H > 0
        
        if transform_name.startswith("spatial_g"):
            parts = transform_name.split("_")
            group_size_str = parts[1][1:]
            if group_size_str == "global" or group_size_str == "0":
                self.target_pow_2 = H
            else:
                group_size = int(group_size_str)
                self.target_pow_2 = ((H + group_size - 1) // group_size) * group_size
        elif transform_name.startswith("blockmax_zero"):
            parts = transform_name.split("_")
            group_size = 32
            for part in parts:
                if part.startswith("g") and part[1:].isdigit():
                    group_size = int(part[1:])
                    break
            self.target_pow_2 = ((H + group_size - 1) // group_size) * group_size
        elif transform_name in ["haar_grp128_l2", "haar_grp128_l2_trends", "haar_grp128_l2_packet", "haar_grp128_l2_packet_first_k", "spatial_grp128", "spatial_static_grp128"]:
            self.target_pow_2 = ((H + 127) // 128) * 128
        elif (transform_name in ["haar", "wht"]) and not is_pow2:
            self.target_pow_2 = 1 << H.bit_length()
        else:
            self.target_pow_2 = H
            
        self.register_buffer("D", None)
        self.register_buffer("wgt_trans_q8", None)
        self.register_buffer("wgt_trans_q4", None)
        self.register_buffer("stamp_indices", None)
        self.is_calibrated = False
        
    def forward(self, input):
        init_shape = input.shape
        x_2d = input.reshape(-1, init_shape[-1])
        
        if self.transform_name == "svd":
            out_2d = svd_matmul_bf16_fp_torch(x_2d, self.weight, self.stamp_size)
            out_shape = init_shape[:-1] + (self.out_features,)
            out_3d = out_2d.reshape(out_shape)
            if self.bias is not None:
                out_3d = out_3d + self.bias
            return out_3d
            
        if self.transform_name == "ioet":
            out_2d = transform_iterative_matmul_bf16_fp_torch(x_2d, self.weight, self.stamp_size, step_size=16)
            out_shape = init_shape[:-1] + (self.out_features,)
            out_3d = out_2d.reshape(out_shape)
            if self.bias is not None:
                out_3d = out_3d + self.bias
            return out_3d
            
        if self.transform_name == "haar_grp128_l2":
            device = x_2d.device
            dtype = x_2d.dtype
            if self.D is None:
                self.D = get_ortho_matrix(self.target_pow_2, self.transform_name, device=device, dtype=dtype)
            out_2d = hybrid_matmul_haar_grp128_l2_torch(x_2d, self.weight, self.D, self.stamp_size)
            out_shape = init_shape[:-1] + (self.out_features,)
            out_3d = out_2d.reshape(out_shape)
            if self.bias is not None:
                out_3d = out_3d + self.bias
            return out_3d
            
        if self.transform_name == "haar_grp128_l2_trends":
            device = x_2d.device
            dtype = x_2d.dtype
            if self.D is None:
                self.D = get_ortho_matrix(self.target_pow_2, "haar_grp128_l2", device=device, dtype=dtype)
            out_2d = hybrid_matmul_haar_grp128_l2_trends_only_torch(x_2d, self.weight, self.D, self.stamp_size)
            out_shape = init_shape[:-1] + (self.out_features,)
            out_3d = out_2d.reshape(out_shape)
            if self.bias is not None:
                out_3d = out_3d + self.bias
            return out_3d
            
        if self.transform_name == "haar_grp128_l2_packet":
            device = x_2d.device
            dtype = x_2d.dtype
            if self.D is None:
                self.D = get_ortho_matrix(self.target_pow_2, self.transform_name, device=device, dtype=dtype)
            out_2d = hybrid_matmul_haar_grp128_l2_torch(x_2d, self.weight, self.D, self.stamp_size)
            out_shape = init_shape[:-1] + (self.out_features,)
            out_3d = out_2d.reshape(out_shape)
            if self.bias is not None:
                out_3d = out_3d + self.bias
            return out_3d
            
        if self.transform_name == "haar_grp128_l2_packet_first_k":
            device = x_2d.device
            dtype = x_2d.dtype
            if self.D is None:
                self.D = get_ortho_matrix(self.target_pow_2, "haar_grp128_l2_packet", device=device, dtype=dtype)
            out_2d = hybrid_matmul_haar_grp128_l2_first_k_torch(x_2d, self.weight, self.D, self.stamp_size)
            out_shape = init_shape[:-1] + (self.out_features,)
            out_3d = out_2d.reshape(out_shape)
            if self.bias is not None:
                out_3d = out_3d + self.bias
            return out_3d
            
        if self.transform_name.startswith("spatial_g"):
            device = x_2d.device
            dtype = x_2d.dtype
            parts = self.transform_name.split("_")
            group_size_str = parts[1][1:]
            stamp_per_group = int(parts[2][1:])
            metric = "energy"
            if len(parts) > 3:
                metric = parts[3]
                
            K = self.target_pow_2
            if group_size_str == "global" or group_size_str == "0":
                group_size = K
            else:
                group_size = int(group_size_str)
                
            if K > x_2d.shape[1]:
                act_padded, _ = pad_columns_to_power_of_2(x_2d, K)
            else:
                act_padded = x_2d
            if K > self.weight.shape[1]:
                wgt_padded, _ = pad_columns_to_power_of_2(self.weight, K)
            else:
                wgt_padded = self.weight
                
            out_2d = hybrid_matmul_spatial_grouped_torch(act_padded, wgt_padded, group_size, stamp_per_group, metric)
            out_shape = init_shape[:-1] + (self.out_features,)
            out_3d = out_2d.reshape(out_shape)
            if self.bias is not None:
                out_3d = out_3d + self.bias
            return out_3d
        if self.transform_name.startswith("error_compact"):
            device = x_2d.device
            dtype = x_2d.dtype
            parts = self.transform_name.split("_")
            R = 8
            passes = 5
            for part in parts:
                if part.startswith("r") and part[1:].isdigit():
                    R = int(part[1:])
                elif part.startswith("p") and part[1:].isdigit():
                    passes = int(part[1:])
                    
            K = self.target_pow_2
            if K > x_2d.shape[1]:
                act_padded, _ = pad_columns_to_power_of_2(x_2d, K)
            else:
                act_padded = x_2d
            if K > self.weight.shape[1]:
                wgt_padded, _ = pad_columns_to_power_of_2(self.weight, K)
            else:
                wgt_padded = self.weight
                
            if not hasattr(self, 'wgt_trans_clipped'):
                O, C = wgt_padded.shape
                num_groups = C // 32
                wgt_blocks = wgt_padded.view(O, num_groups, 32)
                wgt_trans_blocks = apply_kpass_butterfly_torch(wgt_blocks, passes=passes, dim=-1)
                wgt_trans_clipped_blocks = wgt_trans_blocks.narrow(-1, 0, R)
                self.wgt_trans_clipped = wgt_trans_clipped_blocks.reshape(O, num_groups * R)
                
            out_2d = hybrid_matmul_error_compaction_grouped_torch(act_padded, wgt_padded, self.wgt_trans_clipped, R, passes=passes)
            out_shape = init_shape[:-1] + (self.out_features,)
            out_3d = out_2d.reshape(out_shape)
            if self.bias is not None:
                out_3d = out_3d + self.bias
            return out_3d

        if self.transform_name.startswith("double_quant"):
            device = x_2d.device
            dtype = x_2d.dtype
            parts = self.transform_name.split("_")
            group_size = 32
            for part in parts:
                if part.startswith("g") and part[1:].isdigit():
                    group_size = int(part[1:])
                    break
            K = self.target_pow_2
            if K > x_2d.shape[1]:
                act_padded, _ = pad_columns_to_power_of_2(x_2d, K)
            else:
                act_padded = x_2d
            if K > self.weight.shape[1]:
                wgt_padded, _ = pad_columns_to_power_of_2(self.weight, K)
            else:
                wgt_padded = self.weight
                
            out_2d = hybrid_matmul_residual_quant_grouped_torch(act_padded, wgt_padded, group_size)
            out_shape = init_shape[:-1] + (self.out_features,)
            out_3d = out_2d.reshape(out_shape)
            if self.bias is not None:
                out_3d = out_3d + self.bias
            return out_3d

        if self.transform_name.startswith("blockmax_zero"):
            device = x_2d.device
            dtype = x_2d.dtype
            parts = self.transform_name.split("_")
            use_wht = ("wht" in parts)
            use_butterfly = ("butterfly" in parts)
            use_clip = False
            clip_ratio = None
            for part in parts:
                if part.startswith("clip"):
                    use_clip = True
                    val_str = part[4:]
                    try:
                        if "." in val_str:
                            clip_ratio = float(val_str)
                        else:
                            clip_ratio = float(val_str) / 100.0
                    except ValueError:
                        clip_ratio = 1.0
                    break
            
            passes = 2
            n_extract = 1
            for part in parts:
                if part.startswith("p") and part[1:].isdigit():
                    passes = int(part[1:])
                elif part.startswith("n") and part[1:].isdigit():
                    n_extract = int(part[1:])
                    
            token_dynamic = ("dyn" in parts)
            group_size = 32
            for part in parts:
                if part.startswith("g") and part[1:].isdigit():
                    group_size = int(part[1:])
                    break
            
            K = self.target_pow_2
            if K > x_2d.shape[1]:
                act_padded, _ = pad_columns_to_power_of_2(x_2d, K)
            else:
                act_padded = x_2d
            if K > self.weight.shape[1]:
                wgt_padded, _ = pad_columns_to_power_of_2(self.weight, K)
            else:
                wgt_padded = self.weight
                
            if use_butterfly:
                if not hasattr(self, 'weight_transformed'):
                    O, C = wgt_padded.shape
                    wgt_blocks = wgt_padded.view(O, C // group_size, group_size)
                    wgt_trans_blocks = apply_kpass_butterfly_torch(wgt_blocks, passes=passes, dim=-1)
                    self.weight_transformed = wgt_trans_blocks.view(O, C)
                out_2d = hybrid_matmul_zero_masked_butterfly_grouped_torch(
                    act_padded, wgt_padded, self.weight_transformed, group_size, token_dynamic,
                    passes=passes, n_extract=n_extract
                )
            elif use_clip:
                out_2d = hybrid_matmul_zero_masked_clipped_grouped_torch(act_padded, wgt_padded, group_size, token_dynamic, clip_ratio=clip_ratio)
            elif use_wht:
                out_2d = hybrid_matmul_zero_masked_wht_grouped_torch(act_padded, wgt_padded, group_size, token_dynamic)
            else:
                out_2d = hybrid_matmul_zero_masked_grouped_torch(act_padded, wgt_padded, group_size, token_dynamic)
                
            out_shape = init_shape[:-1] + (self.out_features,)
            out_3d = out_2d.reshape(out_shape)
            if self.bias is not None:
                out_3d = out_3d + self.bias
            return out_3d
            
        if self.transform_name == "spatial_grp128":
            device = x_2d.device
            dtype = x_2d.dtype
            K = self.target_pow_2
            if K > x_2d.shape[1]:
                act_padded, _ = pad_columns_to_power_of_2(x_2d, K)
            else:
                act_padded = x_2d
            if K > self.weight.shape[1]:
                wgt_padded, _ = pad_columns_to_power_of_2(self.weight, K)
            else:
                wgt_padded = self.weight
                
            out_2d = hybrid_matmul_spatial_grp128_torch(act_padded, wgt_padded, self.stamp_size)
            out_shape = init_shape[:-1] + (self.out_features,)
            out_3d = out_2d.reshape(out_shape)
            if self.bias is not None:
                out_3d = out_3d + self.bias
            return out_3d
            
        if self.transform_name == "spatial_static_grp128":
            device = x_2d.device
            dtype = x_2d.dtype
            K = self.target_pow_2
            if K > x_2d.shape[1]:
                act_padded, _ = pad_columns_to_power_of_2(x_2d, K)
            else:
                act_padded = x_2d
            if K > self.weight.shape[1]:
                wgt_padded, _ = pad_columns_to_power_of_2(self.weight, K)
            else:
                wgt_padded = self.weight
                
            out_2d = hybrid_matmul_spatial_static_grp128_torch(act_padded, wgt_padded, self.stamp_size)
            out_shape = init_shape[:-1] + (self.out_features,)
            out_3d = out_2d.reshape(out_shape)
            if self.bias is not None:
                out_3d = out_3d + self.bias
            return out_3d

            
        device = x_2d.device
        dtype = x_2d.dtype
        
        if self.D is None:
            self.D = get_ortho_matrix(self.target_pow_2, self.transform_name, device=device, dtype=dtype)
            
            if self.pattern != "calibrated":
                H_w = self.weight.shape[1]
                if self.target_pow_2 > H_w:
                    w_padded, _ = pad_columns_to_power_of_2(self.weight, self.target_pow_2)
                else:
                    w_padded = self.weight
                    
                wgt_trans = torch.matmul(w_padded, self.D.t())
                
                if self.pattern == "contiguous":
                    stamp_idx = torch.arange(self.stamp_size, device=device)
                    rest_idx = torch.arange(self.stamp_size, self.target_pow_2, device=device)
                elif self.pattern == "interleaved":
                    step = self.target_pow_2 // self.stamp_size
                    stamp_idx = torch.arange(0, self.target_pow_2, step, device=device)[:self.stamp_size]
                    mask = torch.ones(self.target_pow_2, dtype=torch.bool, device=device)
                    mask[stamp_idx] = False
                    rest_idx = torch.arange(self.target_pow_2, device=device)[mask]
                    
                self.wgt_trans_q8 = wgt_trans[:, stamp_idx].to(torch.bfloat16)
                if len(rest_idx) > 0:
                    wgt_q4, _, _, _ = quantize_mx_torch(
                        wgt_trans[:, rest_idx], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
                    )
                    self.wgt_trans_q4 = wgt_q4
                else:
                    self.wgt_trans_q4 = torch.zeros(wgt_trans.shape[0], 0, device=device, dtype=dtype)
                    
        if self.pattern == "calibrated" and not self.is_calibrated:
            H_x = x_2d.shape[1]
            if self.target_pow_2 > H_x:
                x_padded, _ = pad_columns_to_power_of_2(x_2d, self.target_pow_2)
                w_padded, _ = pad_columns_to_power_of_2(self.weight, self.target_pow_2)
            else:
                x_padded = x_2d
                w_padded = self.weight
                
            with torch.no_grad():
                act_trans = torch.matmul(x_padded, self.D.t())
                energy = torch.sum(act_trans**2, dim=0)
                sorted_idx = torch.argsort(energy, descending=True)
                self.stamp_indices = sorted_idx[:self.stamp_size]
                
                wgt_trans = torch.matmul(w_padded, self.D.t())
                stamp_idx = self.stamp_indices
                mask = torch.ones(self.target_pow_2, dtype=torch.bool, device=device)
                mask[stamp_idx] = False
                rest_idx = torch.arange(self.target_pow_2, device=device)[mask]
                
                self.wgt_trans_q8 = wgt_trans[:, stamp_idx].to(torch.bfloat16)
                if len(rest_idx) > 0:
                    wgt_q4, _, _, _ = quantize_mx_torch(
                        wgt_trans[:, rest_idx], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
                    )
                    self.wgt_trans_q4 = wgt_q4
                else:
                    self.wgt_trans_q4 = torch.zeros(wgt_trans.shape[0], 0, device=device, dtype=dtype)
            self.is_calibrated = True
            
        H_x = x_2d.shape[1]
        if self.target_pow_2 > H_x:
            x_padded, _ = pad_columns_to_power_of_2(x_2d, self.target_pow_2)
        else:
            x_padded = x_2d
            
        act_trans = torch.matmul(x_padded, self.D.t())
        
        if self.pattern == "contiguous":
            stamp_idx = torch.arange(self.stamp_size, device=device)
            rest_idx = torch.arange(self.stamp_size, self.target_pow_2, device=device)
        elif self.pattern == "interleaved":
            step = self.target_pow_2 // self.stamp_size
            stamp_idx = torch.arange(0, self.target_pow_2, step, device=device)[:self.stamp_size]
            mask = torch.ones(self.target_pow_2, dtype=torch.bool, device=device)
            mask[stamp_idx] = False
            rest_idx = torch.arange(self.target_pow_2, device=device)[mask]
        elif self.pattern == "calibrated":
            stamp_idx = self.stamp_indices
            mask = torch.ones(self.target_pow_2, dtype=torch.bool, device=device)
            mask[stamp_idx] = False
            rest_idx = torch.arange(self.target_pow_2, device=device)[mask]
            
        act_q8_stamp = act_trans[:, stamp_idx].to(torch.bfloat16)
        Rest_size = len(rest_idx)
        if Rest_size > 0:
            act_q4_rest, _, _, _ = quantize_mx_torch(
                act_trans[:, rest_idx], 32, elem_format="e2m1", scale_format="e8m0", prevent_zero=True
            )
        else:
            act_q4_rest = torch.zeros(act_trans.shape[0], 0, device=device, dtype=dtype)
            
        out_q8 = torch.matmul(act_q8_stamp, self.wgt_trans_q8.t())
        if Rest_size > 0:
            out_q4 = torch.matmul(act_q4_rest, self.wgt_trans_q4.t())
        else:
            out_q4 = 0
            
        out_2d = (out_q8 + out_q4).to(dtype)
        
        out_shape = init_shape[:-1] + (self.out_features,)
        out_3d = out_2d.reshape(out_shape)
        
        if self.bias is not None:
            out_3d = out_3d + self.bias
        return out_3d

def run_eval(model_id, block_size=None, prevent_zero=True, four_over_six=False,
             elem_format="e2m1", scale_format="e4m3", num_steps=None, use_hierarchical=False,
             hadamard_size=0, hadamard_seed=42, clip_percentile=None, rounding=None, custom_rotation=None):
    
    print(f"Evaluating {model_id} with block size {block_size}, prevent_zero={prevent_zero}, four_over_six={four_over_six}, elem_format={elem_format}, scale_format={scale_format}, num_steps={num_steps}, use_hierarchical={use_hierarchical}, hadamard_size={hadamard_size}, hadamard_seed={hadamard_seed}, clip_percentile={clip_percentile}, rounding={rounding}, custom_rotation={custom_rotation}")
    
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
            if elem_format.startswith("permuted_swiglu") and "mlp" in name and isinstance(module, torch.nn.Linear):
                continue
                
            if elem_format.startswith("permuted_swiglu") and type(module).__name__ == "LlamaMLP":
                idx = name.rfind(".")
                father_name = name[:idx]
                father_module = model
                if father_name:
                     for part in father_name.split("."):
                          father_module = getattr(father_module, part)
                
                layernorm_name = name.replace(".mlp", ".post_attention_layernorm")
                modules_dict = dict(model.named_modules())
                rms_gamma = modules_dict[layernorm_name].weight
                
                # Parse stamp and search box sizes
                stamp_size = 0
                stamp_size_dff = 0
                search_box_size = 0
                search_box_size_dff = 0
                parts = elem_format.split("_")
                for p in parts:
                    if p.startswith("boxd") and p[4:].isdigit():
                        search_box_size_dff = int(p[4:])
                    elif p.startswith("box") and p[3:].isdigit():
                        search_box_size = int(p[3:])
                    elif p.startswith("s") and p[1:].isdigit():
                        stamp_size = int(p[1:])
                    elif p.startswith("d") and p[1:].isdigit():
                        stamp_size_dff = int(p[1:])
                        
                new_mlp = TorchPermutedMLP(
                    module, rms_gamma, block_size=32, prevent_zero=prevent_zero, four_over_six=four_over_six,
                    elem_format="e2m1", scale_format=scale_format, use_hierarchical=use_hierarchical,
                    clip_percentile=clip_percentile, rounding=rounding,
                    stamp_size=stamp_size, stamp_size_dff=stamp_size_dff,
                    search_box_size=search_box_size, search_box_size_dff=search_box_size_dff
                )
                
                child_name = name[idx+1:] if idx != 0 else name
                setattr(father_module, child_name, new_mlp)
                print(f"Replaced entire SwiGLU MLP: {name} (box={search_box_size}, stamp={stamp_size}, dff_box={search_box_size_dff}, dff_stamp={stamp_size_dff})")
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
                if elem_format == "stamp":
                    new_m = TorchStampLinear(
                        module.in_features, module.out_features, module.bias is not None,
                        stamp_size=block_size
                    )
                elif elem_format == "stamp_mx":
                    new_m = TorchStampMXLinear(
                        module.in_features, module.out_features, module.bias is not None,
                        stamp_size=block_size
                    )
                elif elem_format == "stamp_bf16":
                    new_m = TorchStampBF16Linear(
                        module.in_features, module.out_features, module.bias is not None,
                        stamp_size=block_size
                    )
                elif elem_format == "stamp_feat":
                    new_m = TorchStampFeatLinear(
                        module.in_features, module.out_features, module.bias is not None,
                        stamp_size=block_size
                    )
                elif elem_format == "ablate_seq":
                    new_m = TorchAblateSeqLinear(
                        module.in_features, module.out_features, module.bias is not None,
                        stamp_size=block_size, strategy=rounding
                    )
                elif elem_format == "ablate_feat":
                    new_m = TorchAblateFeatLinear(
                        module.in_features, module.out_features, module.bias is not None,
                        stamp_size=block_size, strategy=rounding
                    )
                elif elem_format == "seq_mean_sub_grp32":
                    new_m = TorchSeqMeanSubLinear(
                        module.in_features, module.out_features, module.bias is not None,
                        group_size=32, block_size=32, prevent_zero=prevent_zero, four_over_six=four_over_six,
                        elem_format="e2m1", scale_format=scale_format,
                        use_hierarchical=use_hierarchical, hadamard_size=hadamard_size, hadamard_seed=hadamard_seed,
                        clip_percentile=clip_percentile, rounding=rounding, custom_rotation=custom_rotation
                    )
                elif elem_format.startswith("svd_outlier_k"):
                    k_val = int(elem_format[13:])
                    new_m = TorchSVDOutlierLinear(
                        module.in_features, module.out_features, module.bias is not None,
                        k=k_val, block_size=32, prevent_zero=prevent_zero, four_over_six=four_over_six,
                        elem_format="e2m1", scale_format=scale_format,
                        use_hierarchical=use_hierarchical, hadamard_size=hadamard_size, hadamard_seed=hadamard_seed,
                        clip_percentile=clip_percentile, rounding=rounding, custom_rotation=custom_rotation
                    )
                elif elem_format.endswith("_seq"):
                    t_name = elem_format[:-4]
                    new_m = TorchTransformSeqLinear(
                        module.in_features, module.out_features, module.bias is not None,
                        stamp_size=block_size, transform_name=t_name
                    )
                elif elem_format.endswith("_feat"):
                    base = elem_format[:-5]
                    known_patterns = ["contiguous", "interleaved", "calibrated"]
                    pattern = "contiguous"
                    t_name = base
                    for pat in known_patterns:
                        if base.endswith("_" + pat):
                            pattern = pat
                            t_name = base[:-(len(pat) + 1)]
                            break
                    new_m = TorchTransformFeatLinear(
                        module.in_features, module.out_features, module.bias is not None,
                        stamp_size=block_size, transform_name=t_name, pattern=pattern
                    )
                else:
                    new_m = TorchMXLinear(
                        module.in_features, module.out_features, module.bias is not None,
                        block_size=block_size, prevent_zero=prevent_zero, four_over_six=four_over_six,
                        elem_format=elem_format, scale_format=scale_format,
                        use_hierarchical=use_hierarchical, hadamard_size=hadamard_size, hadamard_seed=hadamard_seed,
                        clip_percentile=clip_percentile, rounding=rounding, custom_rotation=custom_rotation
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
            val = sys.argv[4].lower()
            if val in ("true", "4o6"):
                four_over_six = "4o6"
            elif val in ("floor_ceil", "foc"):
                four_over_six = "floor_ceil"
            elif val == "false":
                four_over_six = False
            else:
                four_over_six = val
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
        clip_percentile = None
        rounding = None
        if len(sys.argv) > 11 and sys.argv[11].lower() != "none":
            hadamard_size = int(sys.argv[11])
        if len(sys.argv) > 12 and sys.argv[12].lower() != "none":
            hadamard_seed = int(sys.argv[12])
        if len(sys.argv) > 13 and sys.argv[13].lower() != "none":
            clip_percentile = float(sys.argv[13])
        if len(sys.argv) > 14 and sys.argv[14].lower() != "none":
            rounding = sys.argv[14]
        custom_rotation = None
        if len(sys.argv) > 15 and sys.argv[15].lower() != "none":
            custom_rotation = sys.argv[15]
            
        elem_format = format
            
        if "," in elem_format:
            w_fmt, a_fmt = elem_format.split(",")
            base_lbl = f"wgt {w_fmt}, act {a_fmt}"
            opt_tags = []
            if use_hierarchical:
                if hadamard_size > 0:
                    opt_tags.append("hierarchical scales + random hadamard")
                else:
                    opt_tags.append("hierarchical scales")
            elif hadamard_size > 0:
                opt_tags.append("random hadamard")
                
            if opt_tags:
                option = f"{base_lbl} + " + " + ".join(opt_tags)
            else:
                option = base_lbl
        else:
            is_kitchen_sink = (use_hierarchical and (four_over_six in (True, "4o6", "floor_ceil", "foc")) and rounding == "ceil" and hadamard_size == 32)
            
            if elem_format == "stamp":
                option = f"stamp_s{block_sizes[0]}"
            elif elem_format == "stamp_mx":
                option = f"stamp_mx_s{block_sizes[0]}"
            elif elem_format == "stamp_bf16":
                option = f"stamp_bf16_s{block_sizes[0]}"
            elif elem_format == "stamp_feat":
                option = f"stamp_feat_s{block_sizes[0]}"
            elif elem_format == "ablate_seq":
                option = f"ablate_seq_{rounding}_s{block_sizes[0]}"
            elif elem_format == "ablate_feat":
                option = f"ablate_feat_{rounding}_s{block_sizes[0]}"
            elif elem_format.endswith("_seq") or elem_format.endswith("_feat"):
                option = f"{elem_format}_s{block_sizes[0]}"
            elif is_kitchen_sink:
                if scale_format == "e8m0":
                    option = f"mxfp4 ({elem_format}) + Hier+FoC+Ceil+RH32"
                else:
                    option = f"{scale_format} + Hier+FoC+Ceil+RH32"
            elif custom_rotation is not None:
                rot_label = None
                if "tilted_2s" in custom_rotation:
                    rot_label = "Tilted (2s)"
                elif "tilted_3s" in custom_rotation:
                    rot_label = "Tilted (3s)"
                elif "tilted_5s" in custom_rotation:
                    rot_label = "Tilted (5s)"
                elif "rot_2s" in custom_rotation:
                    rot_label = "Rot (2s)"
                elif "rot_3s" in custom_rotation:
                    rot_label = "Rot (3s)"
                elif "rot_5s" in custom_rotation:
                    rot_label = "Rot (5s)"
                elif "ks_320p" in custom_rotation:
                    rot_label = "KS (320p)"
                elif "rot_80p" in custom_rotation:
                    rot_label = "Rot (80p)"
                    
                if scale_format == "e8m0":
                    option = f"mxfp4 ({elem_format}) + {rot_label}"
                else:
                    option = f"{scale_format} + {rot_label}"
            else:
                foc_opt = ""
                if four_over_six in (True, "true", "4o6"):
                    foc_opt = " + 4o6"
                elif four_over_six in ("floor_ceil", "foc"):
                    foc_opt = " + FoC"
                    
                if scale_format == "e8m0":
                    option = f"mxfp4 ({elem_format}){foc_opt}"
                elif elem_format in ("int8", "int4"):
                    option = f"{elem_format}{foc_opt}"
                    if prevent_zero:
                        option += " + PZ"
                else:
                    option = f"{scale_format}{foc_opt}"
                    if prevent_zero:
                        option += " + PZ"
                    
                if use_hierarchical:
                    option += " + H"
                    
                if hadamard_size > 0:
                    option += f" + RH{hadamard_size}"
                    
                if clip_percentile is not None:
                    pct_int = int(round(clip_percentile * 100))
                    option += f" + c{pct_int}"
                    
                if rounding is not None:
                    option += f" + {rounding.capitalize()}"
            
        if block_sizes == [None]:
            base_ppl = None
        else:
            base_ppl = read_base_from_csv(model_id)
        
        for bs in block_sizes:
            ppl = run_eval(
                model_id, bs, prevent_zero=prevent_zero, four_over_six=four_over_six,
                elem_format=elem_format, scale_format=scale_format, num_steps=num_steps,
                use_hierarchical=use_hierarchical, hadamard_size=hadamard_size, hadamard_seed=hadamard_seed,
                clip_percentile=clip_percentile, rounding=rounding, custom_rotation=custom_rotation
            )
            if base_ppl is None and bs is not None:
                print(f"Baseline not found for {model_id}. Please run it first.")
                continue
            update_csv_and_readme(model_id, bs, ppl, base_ppl, option=option, csv_suffix=csv_suffix)
    else:
        ppl = run_eval(model_id, None)
        update_csv_and_readme(model_id, None, ppl, None)

