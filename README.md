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

| Model          |   Baseline |    BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:---------------|-----------:|--------:|-------:|--------:|--------:|--------:|---------:|---------:|
| Llama 3.1 8B   |     6.2438 | 1.7384  | 1.4256 |  1.2911 |  1.2159 |  1.2343 |   1.35   |   1.5325 |
| Granite 3.3 8B |     4.7191 | 1.1498  | 1.0061 |  0.9254 |  0.925  |  1.01   |   1.1999 |   1.5575 |
| Qwen 2.5 14B   |     5.3577 | 1.3907  | 1.2215 |  1.1188 |  1.1582 |  1.2571 |   1.494  |   2.0615 |
| DeepSeek 7B    |    12.2778 | 2.67717 | 2.1138 |  1.7812 |  1.6175 |  1.654  |   1.8505 |   2.1729 |

<!-- RESULTS_END -->
