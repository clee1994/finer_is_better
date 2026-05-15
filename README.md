# Finer is Better: NVFP4 Quantization Evaluation

## Abstract

Microscaling is a critical technique for preserving the quality of Large Language Models (LLMs) quantized to ultra-low precision formats. Intuitively, finer block sizes should yield lower quantization error; however, a counter-intuitive paradox recently identified in the literature demonstrates that standard abs-max scaling can actually degrade model quality as block sizes shrink [[Fasoli et al., 2026](https://arxiv.org/abs/2601.19026)]. In this work, we investigate the underlying mechanics of this phenomenon. We demonstrate that this degradation is not an inherent limitation of finer granularity, but is primarily driven by heavy-tailed tensor distributions interacting poorly with the coarse upper quantization bins of the FP4 element format. Specifically, we show that i) preventing the scaling factor from underflowing to zero mitigates localized errors, ii) targeted algorithmic interventions like the 4-over-6 methodology [[Cook et al., 2025](https://arxiv.org/abs/2502.04066)] effectively correct the quantization geometry for large elements, and iii) a brute-force search establishes an optimal baseline, confirming that the theoretical Mean Squared Error (MSE) strictly improves with finer block sizes. Ultimately, our findings reveal a valuable interchangeability: applying the correct algorithmic recipe allows standard, hardware-compliant formats (like OCP E4M3) to match the performance of custom, wider-exponent formats (like UE5M3). We validate these results across several large language models, fully resolving the scaling paradox and achieving robust downstream perplexity improvements.

*Our full paper findings are available on arXiv: [[Schaefer et al., 2026](https://arxiv.org/abs/2605.08565)].*

This directory contains our consolidated codebase for evaluating baseline perplexity and custom NVFP4 quantization with microscaling on large language models. This file is intended to be fully self-explanatory. Follow the steps below to set up the environment and run experiments.

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


## Running Tests

To run unit tests for logic in this directory:
```bash
python3 test_nvfp4.py
```

## Results

This section is automatically updated by the evaluation script after each run. Do not edit manually.

<!-- RESULTS_START -->

### Consolidated Experimental Figures

#### Publication Final 2x3 Consolidated Figure (Perplexity & Distributions)
![Publication Final Figure](plots/final_fig.png)

### Detailed Evaluation Perplexity Matrices

#### Granite

| Model_Config | Baseline | BS=4 | BS=8 | BS=16 | BS=32 | BS=64 | BS=128 | BS=256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e4m3 | +4.91 | +0.86 | +0.70 | +0.64 | +0.67 | +0.77 | +0.94 | +1.30 |
| e4m3 + PZ | +4.91 | +0.77 | +0.69 | +0.64 | +0.68 | +0.77 | +0.94 | +1.30 |
| e4m3 + 4o6 | +4.91 | +0.43 | +0.48 | +0.56 | +0.69 | +0.78 | +0.94 | +1.28 |
| e4m3 + 4o6 + PZ | +4.91 | +0.43 | +0.48 | +0.56 | +0.70 | +0.78 | +0.94 | +1.28 |
| ue5m3 | +4.91 | -0.03 | +0.04 | +0.11 | +0.19 | +0.38 | +0.60 | +1.00 |
| ue5m3 + PZ | +4.91 | -0.03 | +0.04 | +0.11 | +0.19 | +0.38 | +0.60 | +1.00 |
| ue5m3 + 4o6 | +4.91 | -0.08 | -0.02 | +0.06 | +0.17 | +0.38 | +0.58 | +1.04 |
| ue5m3 + 4o6 + PZ | +4.91 | -0.08 | -0.02 | +0.06 | +0.17 | +0.38 | +0.58 | +1.04 |
| e4m3 + H | +4.91 | +0.00 | +0.07 | +0.14 | +0.21 | +0.38 | +0.60 | +1.05 |
| e4m3 + PZ + H | +4.91 | -0.03 | +0.04 | +0.12 | +0.20 | +0.37 | +0.60 | +1.05 |
| e4m3 + 4o6 + H | +4.91 | -0.06 | -0.00 | +0.07 | +0.18 | +0.38 | +0.59 | +1.07 |
| e4m3 + 4o6 + PZ + H | +4.91 | -0.08 | -0.00 | +0.07 | +0.18 | +0.38 | +0.58 | +1.07 |
| ue5m3 + H | +4.91 | -0.03 | +0.04 | +0.13 | +0.22 | +0.37 | +0.58 | +1.05 |
| ue5m3 + PZ + H | +4.91 | -0.03 | +0.04 | +0.13 | +0.22 | +0.37 | +0.58 | +1.05 |
| ue5m3 + 4o6 + H | +4.91 | -0.07 | -0.02 | +0.07 | +0.21 | +0.35 | +0.57 | +1.05 |
| ue5m3 + 4o6 + PZ + H | +4.91 | -0.07 | -0.02 | +0.07 | +0.21 | +0.35 | +0.57 | +1.05 |

#### Llama

| Model_Config | Baseline | BS=4 | BS=8 | BS=16 | BS=32 | BS=64 | BS=128 | BS=256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e4m3 | +6.33 | +0.72 | +0.56 | +0.54 | +0.58 | +0.67 | +0.80 | +0.99 |
| e4m3 + PZ | +6.33 | +0.59 | +0.54 | +0.54 | +0.58 | +0.68 | +0.80 | +0.99 |
| e4m3 + 4o6 | +6.33 | +0.27 | +0.33 | +0.42 | +0.53 | +0.66 | +0.80 | +0.99 |
| e4m3 + 4o6 + PZ | +6.33 | +0.25 | +0.32 | +0.42 | +0.53 | +0.66 | +0.80 | +0.99 |
| ue5m3 | +6.33 | +0.20 | +0.31 | +0.44 | +0.54 | +0.66 | +0.78 | +1.01 |
| ue5m3 + PZ | +6.33 | +0.20 | +0.31 | +0.44 | +0.54 | +0.66 | +0.78 | +1.01 |
| ue5m3 + 4o6 | +6.33 | +0.11 | +0.23 | +0.38 | +0.50 | +0.64 | +0.80 | +1.00 |
| ue5m3 + 4o6 + PZ | +6.33 | +0.11 | +0.23 | +0.38 | +0.50 | +0.64 | +0.80 | +1.00 |
| e4m3 + H | +6.33 | +0.21 | +0.32 | +0.42 | +0.53 | +0.65 | +0.79 | +0.99 |
| e4m3 + PZ + H | +6.33 | +0.20 | +0.31 | +0.43 | +0.53 | +0.64 | +0.79 | +0.98 |
| e4m3 + 4o6 + H | +6.33 | +0.12 | +0.23 | +0.37 | +0.49 | +0.64 | +0.80 | +0.99 |
| e4m3 + 4o6 + PZ + H | +6.33 | +0.12 | +0.23 | +0.36 | +0.49 | +0.62 | +0.80 | +0.99 |
| ue5m3 + H | +6.33 | +0.19 | +0.30 | +0.43 | +0.54 | +0.66 | +0.81 | +1.00 |
| ue5m3 + PZ + H | +6.33 | +0.19 | +0.30 | +0.43 | +0.54 | +0.66 | +0.81 | +1.00 |
| ue5m3 + 4o6 + H | +6.33 | +0.11 | +0.23 | +0.36 | +0.50 | +0.64 | +0.81 | +0.99 |
| ue5m3 + 4o6 + PZ + H | +6.33 | +0.11 | +0.23 | +0.36 | +0.50 | +0.64 | +0.81 | +0.99 |

#### DeepSeek

| Model_Config | Baseline | BS=4 | BS=8 | BS=16 | BS=32 | BS=64 | BS=128 | BS=256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e4m3 | +12.39 | +0.29 | +0.41 | +0.54 | +0.71 | +0.88 | +1.08 | +1.40 |
| e4m3 + PZ | +12.39 | +0.29 | +0.41 | +0.55 | +0.71 | +0.87 | +1.08 | +1.40 |
| e4m3 + 4o6 | +12.39 | +0.14 | +0.27 | +0.43 | +0.61 | +0.82 | +1.08 | +1.38 |
| e4m3 + 4o6 + PZ | +12.39 | +0.14 | +0.27 | +0.42 | +0.62 | +0.83 | +1.08 | +1.38 |
| ue5m3 | +12.39 | +0.20 | +0.37 | +0.53 | +0.71 | +0.86 | +1.10 | +1.41 |
| ue5m3 + PZ | +12.39 | +0.20 | +0.37 | +0.53 | +0.71 | +0.86 | +1.10 | +1.41 |
| ue5m3 + 4o6 | +12.39 | +0.13 | +0.25 | +0.42 | +0.61 | +0.81 | +1.09 | +1.41 |
| ue5m3 + 4o6 + PZ | +12.39 | +0.12 | +0.25 | +0.42 | +0.61 | +0.81 | +1.09 | +1.41 |
| e4m3 + H | +12.39 | +0.21 | +0.38 | +0.54 | +0.70 | +0.84 | +1.08 | +1.39 |
| e4m3 + PZ + H | +12.39 | +0.20 | +0.37 | +0.54 | +0.70 | +0.83 | +1.08 | +1.39 |
| e4m3 + 4o6 + H | +12.39 | +0.11 | +0.27 | +0.45 | +0.64 | +0.81 | +1.06 | +1.39 |
| e4m3 + 4o6 + PZ + H | +12.39 | +0.11 | +0.26 | +0.45 | +0.62 | +0.81 | +1.06 | +1.39 |
| ue5m3 + H | +12.39 | +0.21 | +0.38 | +0.53 | +0.66 | +0.86 | +1.09 | +1.39 |
| ue5m3 + PZ + H | +12.39 | +0.21 | +0.38 | +0.53 | +0.66 | +0.86 | +1.09 | +1.39 |
| ue5m3 + 4o6 + H | +12.39 | +0.12 | +0.27 | +0.44 | +0.62 | +0.82 | +1.08 | +1.42 |
| ue5m3 + 4o6 + PZ + H | +12.39 | +0.12 | +0.27 | +0.44 | +0.62 | +0.82 | +1.08 | +1.42 |

#### Qwen

| Model_Config | Baseline | BS=4 | BS=8 | BS=16 | BS=32 | BS=64 | BS=128 | BS=256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e4m3 | +5.36 | +0.47 | +0.49 | +0.56 | +0.65 | +0.81 | +1.04 | +1.49 |
| e4m3 + PZ | +5.36 | +0.45 | +0.48 | +0.55 | +0.64 | +0.81 | +1.04 | +1.49 |
| e4m3 + 4o6 | +5.36 | +0.27 | +0.36 | +0.47 | +0.61 | +0.78 | +1.04 | +1.47 |
| e4m3 + 4o6 + PZ | +5.36 | +0.26 | +0.35 | +0.47 | +0.61 | +0.79 | +1.03 | +1.48 |
| ue5m3 | +5.36 | +0.28 | +0.40 | +0.51 | +0.64 | +0.79 | +1.06 | +1.51 |
| ue5m3 + PZ | +5.36 | +0.28 | +0.39 | +0.51 | +0.64 | +0.79 | +1.06 | +1.51 |
| ue5m3 + 4o6 | +5.36 |  |  |  |  | +0.78 | +1.06 | +1.50 |
| ue5m3 + 4o6 + PZ | +5.36 | +0.20 | +0.32 | +0.45 | +0.60 | +0.78 | +1.06 | +1.50 |
| e4m3 + H | +5.36 | +0.31 | +0.41 | +0.52 | +0.64 | +0.80 | +1.03 | +1.48 |
| e4m3 + PZ + H | +5.36 | +0.30 | +0.41 | +0.51 | +0.63 | +0.79 | +1.03 | +1.48 |
| e4m3 + 4o6 + H | +5.36 | +0.19 | +0.33 | +0.46 | +0.61 | +0.79 | +1.03 | +1.47 |
| e4m3 + 4o6 + PZ + H | +5.36 | +0.19 | +0.33 | +0.46 | +0.60 | +0.79 | +1.04 | +1.48 |
| ue5m3 + H | +5.36 | +0.27 | +0.39 | +0.50 | +0.64 | +0.80 | +1.05 | +1.47 |
| ue5m3 + PZ + H | +5.36 | +0.27 | +0.39 | +0.50 | +0.64 | +0.80 | +1.05 | +1.47 |
| ue5m3 + 4o6 + H | +5.36 | +0.19 | +0.32 | +0.45 | +0.61 | +0.78 | +1.03 | +1.49 |
| ue5m3 + 4o6 + PZ + H | +5.36 | +0.19 | +0.32 | +0.45 | +0.61 | +0.78 | +1.03 | +1.49 |



<!-- RESULTS_END -->

