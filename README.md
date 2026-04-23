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
python3 mx_qllm_eval.py <model_id>
```

**Examples for our 4 target models:**
- **Llama 3.1 8B**: `python3 mx_qllm_eval.py meta-llama/Llama-3.1-8B`
- **Granite 3.3 8B**: `python3 mx_qllm_eval.py ibm-granite/granite-3.3-8b-base`
- **Qwen 2.5 14B**: `python3 mx_qllm_eval.py Qwen/Qwen2.5-14B`
- **DeepSeek 7B**: `python3 mx_qllm_eval.py deepseek-ai/deepseek-llm-7b-base`

### Quantized Perplexity Evaluation
To run the evaluation with NVFP4 element quantization and simulated FP8 scales for a specific microscale block size, use this command:
```bash
python3 mx_qllm_eval.py <model_id> <block_size>
```

**Examples:**
- `python3 mx_qllm_eval.py meta-llama/Llama-3.1-8B 4`
- `python3 mx_qllm_eval.py Qwen/Qwen2.5-14B 16`

## Running Tests

To run unit tests for logic in this directory:
```bash
python3 test_nvfp4.py
```

## Results

This section is automatically updated by the evaluation script after each run. Do not edit manually.

<!-- RESULTS_START -->

### DeepSeek

|              |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:-------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| e4m3_pz      |        nan |   0.4  |   0.52 |    0.66 |    0.82 |    0.98 |     1.19 |     1.51 |
| e4m3_no_pz   |        nan |   0.4  |   0.52 |    0.65 |    0.82 |    0.99 |     1.19 |     1.51 |
| 4over6_no_pz |        nan |   0.25 |   0.38 |    0.54 |    0.73 |    0.93 |     1.19 |     1.49 |
| 4over6_pz    |        nan |   0.25 |   0.39 |    0.54 |    0.74 |    0.94 |     1.19 |     1.49 |
| ue5m3_no_pz  |        nan |   0.32 |   0.48 |    0.64 |    0.82 |    0.97 |     1.22 |     1.53 |
| ue5m3_pz     |        nan |   0.24 |   0.36 |    0.54 |    0.72 |    0.92 |     1.2  |     1.53 |

### Granite

|              |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:-------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| e4m3_pz      |        nan |   0.96 |   0.88 |    0.83 |    0.87 |    0.96 |     1.13 |     1.5  |
| e4m3_no_pz   |        nan |   1.06 |   0.89 |    0.83 |    0.86 |    0.96 |     1.13 |     1.5  |
| 4over6_no_pz |        nan |   0.62 |   0.67 |    0.75 |    0.88 |    0.97 |     1.13 |     1.48 |
| 4over6_pz    |        nan |   0.62 |   0.67 |    0.75 |    0.89 |    0.97 |     1.13 |     1.48 |
| ue5m3_no_pz  |        nan |   0.16 |   0.23 |    0.3  |    0.38 |    0.57 |     0.79 |     1.19 |
| ue5m3_pz     |        nan |   0.11 |   0.17 |    0.25 |    0.36 |    0.57 |     0.77 |     1.23 |

### Llama

|              |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:-------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| 4over6_pz    |       6.33 |   0.34 |   0.4  |    0.51 |    0.62 |    0.75 |     0.88 |     1.08 |
| ue5m3_pz     |       6.33 |   0.19 |   0.32 |    0.47 |    0.58 |    0.73 |     0.88 |     1.09 |
| ue5m3_no_pz  |       6.33 |   0.28 |   0.39 |    0.53 |    0.62 |    0.74 |     0.86 |     1.09 |
| e4m3_no_pz   |       6.33 |   0.81 |   0.65 |    0.62 |    0.67 |    0.75 |     0.88 |     1.07 |
| e4m3_pz      |       6.33 |   0.68 |   0.63 |    0.62 |    0.66 |    0.76 |     0.89 |     1.07 |
| 4over6_no_pz |       6.33 |   0.36 |   0.41 |    0.51 |    0.61 |    0.75 |     0.88 |     1.08 |

### Qwen

|            |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:-----------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| e4m3_no_pz |        nan |   0.47 |   0.49 |    0.56 |    0.65 |    0.81 |     1.04 |     1.49 |

### Graphs
<table>
  <tr>
    <td><img src="plots/gap_deepseek.png" alt="DeepSeek Gap" width="400">
<p align="center">DeepSeek Gap</p></td>
    <td><img src="plots/gap_granite.png" alt="Granite Gap" width="400">
<p align="center">Granite Gap</p></td>
  </tr>
  <tr>
    <td><img src="plots/gap_llama.png" alt="Llama Gap" width="400">
<p align="center">Llama Gap</p></td>
    <td><img src="plots/gap_qwen.png" alt="Qwen Gap" width="400">
<p align="center">Qwen Gap</p></td>
  </tr>
</table>


<!-- RESULTS_END -->
