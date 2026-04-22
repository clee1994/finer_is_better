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

## Running Tests

To run unit tests for logic in this directory:
```bash
python3 test_nvfp4.py
```

## Results

This section is automatically updated by the evaluation script after each run. Do not edit manually.

<!-- RESULTS_START -->

| Model          |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:---------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|
| Llama 3.1 8B   |       6.24 |   1.74 |   1.43 |    1.29 |    1.22 |    1.23 |     1.35 |     1.53 |
| Granite 3.3 8B |       4.72 |   1.15 |   1.01 |    0.93 |    0.93 |    1.01 |     1.2  |     1.56 |
| Qwen 2.5 14B   |       5.36 |   1.39 |   1.22 |    1.12 |    1.16 |    1.26 |     1.49 |     2.06 |
| DeepSeek 7B    |      12.28 |   2.68 |   2.11 |    1.78 |    1.62 |    1.65 |     1.85 |     2.17 |

### Graphs
<table>
  <tr>
    <td><img src="plots/gap_llama_3.1_8b.png" alt="Llama 3.1 8B Gap" width="400">
<p align="center">Llama 3.1 8B Gap</p></td>
    <td><img src="plots/gap_granite_3.3_8b.png" alt="Granite 3.3 8B Gap" width="400">
<p align="center">Granite 3.3 8B Gap</p></td>
  </tr>
  <tr>
    <td><img src="plots/gap_qwen_2.5_14b.png" alt="Qwen 2.5 14B Gap" width="400">
<p align="center">Qwen 2.5 14B Gap</p></td>
    <td><img src="plots/gap_deepseek_7b.png" alt="DeepSeek 7B Gap" width="400">
<p align="center">DeepSeek 7B Gap</p></td>
  </tr>
</table>


<!-- RESULTS_END -->
