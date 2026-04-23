# Finer is Better: NVFP4 Quantization Evaluation

This directory contains our consolidated codebase for evaluating baseline perplexity and custom NVFP4 quantization with microscaling on large language models, intended to reproduce findings from paper 2601.19026.

This file is intended to be fully self-explanatory. Follow the steps below to set up the environment and run experiments.

## Setup

To set up the environment on a new machine or VM, follow these steps exactly:

1.  **Create and Activate Virtual Environment**:
    Run these commands in your terminal to create a new Python virtual environment and install the required packages:
    ```bash
    python3 -m venv ~/quant_eval_venv
    source ~/quant_eval_venv/bin/activate
    pip install --upgrade pip
    pip install torch transformers datasets tqdm pandas tabulate
    ```

2.  **Clone and Install Paper\'s Repository**:
    Run these commands to clone the paper\'s repository and install the `mx` package in editable mode. This adds support for the fast CUDA operations!
    ```bash
    git clone https://github.com/iclr2016codeshare/microscaling ~/MX-QLLM
    cd ~/MX-QLLM/microxcaling
    pip install -e .
    cd ~
    ```

## How to Run

### Baseline Perplexity Evaluation
To calculate the baseline perplexity (WikiText-2) for a model without any quantization applied, use this command:
```bash
python3 nvfp4_eval.py <model_id>
```

**Examples for our 4 target models:**
- **Llama 3.1 8B**: `python3 nvfp4_eval.py meta-llama/Llama-3.1-8B`
- **Granite 3.3 8B**: `python3 nvfp4_eval.py ibm-granite/granite-3.3-8b-base`
- **Qwen 2.5 14B**: `python3 nvfp4_eval.py Qwen/Qwen2.5-14B`
- **DeepSeek 7B**: `python3 nvfp4_eval.py deepseek-ai/deepseek-llm-7b-base`

### Quantized Perplexity Evaluation
To run the evaluation with NVFP4 element quantization and simulated FP8 scales for a specific microscale block size, use this command:
```bash
python3 nvfp4_eval.py <model_id> <block_size>
```

**Examples:**
- `python3 nvfp4_eval.py meta-llama/Llama-3.1-8B 4`
- `python3 nvfp4_eval.py Qwen/Qwen2.5-14B 16`

## Custom Modifications

We have implemented several custom modifications to the standard FP4 quantization scheme to improve precision and handle edge cases. They are controlled by arguments to `torch_eval.py`:

`python3 torch_eval.py <model_id> <block_sizes> [prevent_zero] [four_over_six] [use_ue5m3] [num_steps] [csv_suffix]`

1.  **Prevent Zero (PZ)**: Refuses to round scales to absolute zero, bounding them at a minimum value of $2^{-9}$. This prevents extreme degradation in models sensitive to small weights (like Qwen).
    *   **Activation**: Pass `true` as the 3rd argument (default: `true`).
2.  **Four Over Six (4o6)**: Dynamically selects between scaling by 4 or 6 based on which one minimizes Mean Squared Error (MSE) for each block.
    *   **Activation**: Pass `true` as the 4th argument (default: `false`).
3.  **UE5M3**: Uses a custom unsigned 8-bit floating point format for scales with 5 exponent bits and 3 mantissa bits, allowing a wider range than standard FP8.
    *   **Activation**: Pass `true` as the 5th argument (default: `false`).

## Distribution Analysis

To understand why certain models (like Qwen) are sensitive to quantization, we extracted the weight and activation distributions for several layers and plotted them as log-scale overlapping histograms (Ridge Plots).

### Qwen vs Granite Distributions

<table>
  <tr>
    <td><img src="plots/qwen_vs_granite_weights.png" alt="Weights Contrast" width="400"><p align="center">Weights Contrast</p></td>
    <td><img src="plots/qwen_vs_granite_acts.png" alt="Activations Contrast" width="400"><p align="center">Activations Contrast</p></td>
  </tr>
</table>

## Running Tests

To run unit tests for logic in this directory:
```bash
python3 test_nvfp4.py
```

## Results

This section is automatically updated by the evaluation script after each run. Do not edit manually.

<!-- RESULTS_START -->

### <<<<<<< HEAD

|              |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:-------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| <<<<<<< HEAD |        nan |    nan |    nan |     nan |     nan |     nan |      nan |      nan |

### =======

|         |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:--------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| ======= |        nan |    nan |    nan |     nan |     nan |     nan |      nan |      nan |

### >>>>>>> 604f619d0dd22c3458f8eaeb2dbbbb098d079bcb

|                                                  |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:-------------------------------------------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| >>>>>>> 604f619d0dd22c3458f8eaeb2dbbbb098d079bcb |        nan |    nan |    nan |     nan |     nan |     nan |      nan |      nan |

### DeepSeek

|                 |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:----------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| e4m3            |     nan    |   0.4  |   0.52 |    0.65 |    0.82 |    0.99 |     1.19 |     1.51 |
| e4m3 + PZ       |      12.39 |   0.4  |   0.52 |    0.66 |    0.82 |    0.98 |     1.19 |     1.51 |
| e4m3 + 4o6      |     nan    |   0.25 |   0.38 |    0.54 |    0.73 |    0.93 |     1.19 |     1.49 |
| e4m3 + 4o6 + PZ |     nan    |   0.25 |   0.39 |    0.54 |    0.74 |    0.94 |     1.19 |     1.49 |
| ue5m3           |     nan    |   0.32 |   0.48 |    0.64 |    0.82 |    0.97 |     1.22 |     1.53 |
| ue5m3 + PZ      |     nan    |   0.24 |   0.36 |    0.54 |    0.72 |    0.92 |     1.2  |     1.53 |
| nvfp4           |      12.28 | nan    | nan    |  nan    |  nan    |  nan    |   nan    |   nan    |
| DeepSeek 7B     |     nan    | nan    | nan    |  nan    |  nan    |  nan    |   nan    |   nan    |

### DeepSeek 7B

|             |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| DeepSeek 7B |        nan |    nan |    nan |     nan |     nan |     nan |      nan |      nan |

### Granite

|                 |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:----------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| e4m3            |     nan    |   1.06 |   0.89 |    0.83 |    0.86 |    0.96 |     1.13 |     1.5  |
| e4m3 + PZ       |       4.91 |   0.96 |   0.88 |    0.83 |    0.87 |    0.96 |     1.13 |     1.5  |
| e4m3 + 4o6      |     nan    |   0.62 |   0.67 |    0.75 |    0.88 |    0.97 |     1.13 |     1.48 |
| e4m3 + 4o6 + PZ |     nan    |   0.62 |   0.67 |    0.75 |    0.89 |    0.97 |     1.13 |     1.48 |
| ue5m3           |     nan    |   0.16 |   0.23 |    0.3  |    0.38 |    0.57 |     0.79 |     1.19 |
| ue5m3 + PZ      |     nan    |   0.11 |   0.17 |    0.25 |    0.36 |    0.57 |     0.77 |     1.23 |
| nvfp4           |       4.72 | nan    | nan    |  nan    |  nan    |  nan    |   nan    |   nan    |
| Granite 3.3 8B  |     nan    | nan    | nan    |  nan    |  nan    |  nan    |   nan    |   nan    |

### Granite 3.3 8B

|                |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:---------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| Granite 3.3 8B |        nan |    nan |    nan |     nan |     nan |     nan |      nan |      nan |

### Llama

|                 |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:----------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| e4m3            |     nan    |   0.81 |   0.65 |    0.62 |    0.67 |    0.75 |     0.88 |     1.07 |
| e4m3 + PZ       |     nan    |   0.68 |   0.63 |    0.62 |    0.66 |    0.76 |     0.89 |     1.07 |
| e4m3 + 4o6      |     nan    |   0.36 |   0.41 |    0.51 |    0.61 |    0.75 |     0.88 |     1.08 |
| e4m3 + 4o6 + PZ |     nan    |   0.34 |   0.4  |    0.51 |    0.62 |    0.75 |     0.88 |     1.08 |
| ue5m3           |     nan    |   0.28 |   0.39 |    0.53 |    0.62 |    0.74 |     0.86 |     1.09 |
| ue5m3 + PZ      |     nan    |   0.1  |   0.23 |    0.38 |    0.49 |    0.63 |     0.79 |     1    |
| nvfp4_no_pz     |     nan    |   0.91 |   0.75 |    0.72 |    0.79 |    0.87 |     0.98 |   nan    |
| nvfp4_pz        |     nan    |   0.79 |   0.73 |    0.73 |    0.77 |    0.88 |     0.99 |   nan    |
| nvfp4           |       6.33 | nan    | nan    |  nan    |  nan    |  nan    |   nan    |   nan    |
| Llama 3.1 8B    |     nan    |   1.74 |   1.43 |    1.29 |    1.22 |    1.23 |     1.35 |     1.53 |

### Llama 3.1 8B

|              |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:-------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| Llama 3.1 8B |        nan |   1.74 |   1.43 |    1.29 |    1.22 |    1.23 |     1.35 |     1.53 |

### Qwen

|                 |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:----------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| e4m3            |     nan    |   0.47 |   0.49 |    0.56 |    0.65 |    0.81 |     1.04 |     1.49 |
| e4m3 + PZ       |       4.77 | nan    | nan    |  nan    |  nan    |  nan    |   nan    |   nan    |
| e4m3 + 4o6      |     nan    |   0.86 |   0.95 |    1.06 |    1.19 |    1.37 |     1.62 |     2.05 |
| e4m3 + 4o6 + PZ |     nan    |   0.85 |   0.93 |    1.05 |    1.19 |    1.37 |     1.62 |     2.06 |
| ue5m3           |     nan    |   0.86 |   0.98 |    1.09 |    1.22 |    1.37 |     1.64 |     2.1  |
| ue5m3 + PZ      |     nan    |   0.78 |   0.9  |    1.03 |    1.18 |    1.37 |     1.64 |     2.09 |
| nvfp4           |       5.36 | nan    | nan    |  nan    |  nan    |  nan    |   nan    |   nan    |
| Qwen 2.5 14B    |     nan    | nan    | nan    |  nan    |  nan    |  nan    |   nan    |   nan    |

### Qwen 2.5 14B

|              |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:-------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| Qwen 2.5 14B |        nan |    nan |    nan |     nan |     nan |     nan |      nan |      nan |

### Graphs
<table>
  <tr>
    <td><img src="plots/gap_<<<<<<<_head.png" alt="<<<<<<< HEAD Gap" width="400">
<p align="center"><<<<<<< HEAD Gap</p></td>
    <td><img src="plots/gap_=======.png" alt="======= Gap" width="400">
<p align="center">======= Gap</p></td>
  </tr>
  <tr>
    <td><img src="plots/gap_>>>>>>>_604f619d0dd22c3458f8eaeb2dbbbb098d079bcb.png" alt=">>>>>>> 604f619d0dd22c3458f8eaeb2dbbbb098d079bcb Gap" width="400">
<p align="center">>>>>>>> 604f619d0dd22c3458f8eaeb2dbbbb098d079bcb Gap</p></td>
    <td><img src="plots/gap_deepseek.png" alt="DeepSeek Gap" width="400">
<p align="center">DeepSeek Gap</p></td>
  </tr>
  <tr>
    <td><img src="plots/gap_deepseek_7b.png" alt="DeepSeek 7B Gap" width="400">
<p align="center">DeepSeek 7B Gap</p></td>
    <td><img src="plots/gap_granite.png" alt="Granite Gap" width="400">
<p align="center">Granite Gap</p></td>
  </tr>
  <tr>
    <td><img src="plots/gap_granite_3.3_8b.png" alt="Granite 3.3 8B Gap" width="400">
<p align="center">Granite 3.3 8B Gap</p></td>
    <td><img src="plots/gap_llama.png" alt="Llama Gap" width="400">
<p align="center">Llama Gap</p></td>
  </tr>
  <tr>
    <td><img src="plots/gap_llama_3.1_8b.png" alt="Llama 3.1 8B Gap" width="400">
<p align="center">Llama 3.1 8B Gap</p></td>
    <td><img src="plots/gap_qwen.png" alt="Qwen Gap" width="400">
<p align="center">Qwen Gap</p></td>
  </tr>
  <tr>
    <td><img src="plots/gap_qwen_2.5_14b.png" alt="Qwen 2.5 14B Gap" width="400">
<p align="center">Qwen 2.5 14B Gap</p></td>
    <td></td>
  </tr>
</table>


<!-- RESULTS_END -->
