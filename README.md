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

### Granite

|                                                            |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |   BS=512 |
|:-----------------------------------------------------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|---------:|
| e4m3 + PZ                                                  |       4.91 |    nan |    nan |    0.64 |  nan    |     nan |      nan |      nan |   nan    |
| e4m3 + PZ + RH16                                           |     nan    |    nan |    nan |    0.35 |  nan    |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH16                                        |     nan    |    nan |    nan |  nan    |    0.4  |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH32                                        |     nan    |    nan |    nan |  nan    |    0.38 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH64                                        |     nan    |    nan |    nan |  nan    |    0.39 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH128                                       |     nan    |    nan |    nan |  nan    |    0.39 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH16                                        |     nan    |    nan |    nan |  nan    |    0.65 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH32                                        |     nan    |    nan |    nan |  nan    |    0.61 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH64                                        |     nan    |    nan |    nan |  nan    |    0.67 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH128                                       |     nan    |    nan |    nan |  nan    |    0.67 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Ceil                                        |     nan    |    nan |    nan |  nan    |    0.59 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Floor                                       |     nan    |    nan |    nan |  nan    |  181.41 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Ceil                                        |     nan    |    nan |    nan |  nan    |    1.04 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Floor                                       |     nan    |    nan |    nan |  nan    |  126.83 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Hier+FoC+Ceil+RH32                          |     nan    |    nan |    nan |  nan    |    0.38 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Hier+FoC+Ceil+RH32                          |     nan    |    nan |    nan |  nan    |    0.57 |     nan |      nan |      nan |   nan    |
| int8 + PZ                                                  |       4.91 |    nan |    nan |  nan    |  nan    |     nan |      nan |      nan |    -0.16 |
| int8 + PZ + RH256                                          |       4.91 |    nan |    nan |  nan    |  nan    |     nan |      nan |      nan |    -0.19 |
| int4 + PZ                                                  |       4.91 |    nan |    nan |  nan    |  nan    |     nan |      nan |      nan |     5.78 |
| int4 + PZ + RH256                                          |       4.91 |    nan |    nan |  nan    |  nan    |     nan |      nan |      nan |     0.55 |
| mxfp4 (e2m1)                                               |       4.91 |    nan |    nan |  nan    |    0.59 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2)                                               |       4.91 |    nan |    nan |  nan    |    1.04 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + H                                           |     nan    |    nan |    nan |  nan    |    0.47 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + c90                                         |     nan    |    nan |    nan |  nan    |    0.54 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + c95                                         |     nan    |    nan |    nan |  nan    |    0.57 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + c99                                         |     nan    |    nan |    nan |  nan    |    0.59 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + c90                                         |     nan    |    nan |    nan |  nan    |    0.79 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + c95                                         |     nan    |    nan |    nan |  nan    |    0.89 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + c99                                         |     nan    |    nan |    nan |  nan    |    1.01 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + 4o6                                         |     nan    |    nan |    nan |  nan    |    0.73 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Tilted (2s)                                 |     nan    |    nan |    nan |  nan    |    0.49 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Tilted (3s)                                 |     nan    |    nan |    nan |  nan    |    0.53 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Tilted (5s)                                 |     nan    |    nan |    nan |  nan    |    0.59 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Rot (2s)                                    |     nan    |    nan |    nan |  nan    |    0.4  |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Rot (3s)                                    |     nan    |    nan |    nan |  nan    |    0.44 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Rot (5s)                                    |     nan    |    nan |    nan |  nan    |    0.47 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Tilted (2s)                                 |     nan    |    nan |    nan |  nan    |    0.68 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Tilted (3s)                                 |     nan    |    nan |    nan |  nan    |    1.42 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Tilted (5s)                                 |     nan    |    nan |    nan |  nan    |    0.72 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (2s)                                    |     nan    |    nan |    nan |  nan    |    0.71 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (3s)                                    |     nan    |    nan |    nan |  nan    |    0.7  |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (5s)                                    |     nan    |    nan |    nan |  nan    |    0.72 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + KS (320p)                                   |     nan    |    nan |    nan |  nan    |    0.77 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (80p)                                   |     nan    |    nan |    nan |  nan    |    0.66 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1                                         |     nan    |    nan |    nan |  nan    |    0.79 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1 + random hadamard                       |     nan    |    nan |    nan |  nan    |    0.53 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1 + hierarchical scales                   |     nan    |    nan |    nan |  nan    |    0.78 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1 + hierarchical scales + random hadamard |     nan    |    nan |    nan |  nan    |    0.55 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2                                         |     nan    |    nan |    nan |  nan    |    0.82 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2 + random hadamard                       |     nan    |    nan |    nan |  nan    |    0.45 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2 + hierarchical scales                   |     nan    |    nan |    nan |  nan    |    0.86 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2 + hierarchical scales + random hadamard |     nan    |    nan |    nan |  nan    |    0.42 |     nan |      nan |      nan |   nan    |

### Llama

|                                                            |   Baseline |   BS=4 |   BS=8 |   BS=16 |     BS=32 |   BS=64 |   BS=128 |   BS=256 |   BS=512 |
|:-----------------------------------------------------------|-----------:|-------:|-------:|--------:|----------:|--------:|---------:|---------:|---------:|
| e4m3 + PZ                                                  |       6.33 |    nan |    nan |    0.54 |    nan    |     nan |      nan |      nan |   nan    |
| int8 + PZ                                                  |       6.33 |    nan |    nan |  nan    |    nan    |     nan |      nan |      nan |    -0.06 |
| int8 + PZ + RH256                                          |       6.33 |    nan |    nan |  nan    |    nan    |     nan |      nan |      nan |    -0.08 |
| int4 + PZ                                                  |       6.33 |    nan |    nan |  nan    |    nan    |     nan |      nan |      nan |     3.32 |
| int4 + PZ + RH256                                          |       6.33 |    nan |    nan |  nan    |    nan    |     nan |      nan |      nan |     1.34 |
| mxfp4 (e2m1)                                               |       6.33 |    nan |    nan |  nan    |      0.94 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2)                                               |       6.33 |    nan |    nan |  nan    |      1.81 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Tilted (2s)                                 |     nan    |    nan |    nan |  nan    |      1.26 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Tilted (3s)                                 |     nan    |    nan |    nan |  nan    |      1.54 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Tilted (5s)                                 |     nan    |    nan |    nan |  nan    |      1.68 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Rot (2s)                                    |     nan    |    nan |    nan |  nan    |      1.16 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Rot (3s)                                    |     nan    |    nan |    nan |  nan    |      1.21 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Rot (5s)                                    |     nan    |    nan |    nan |  nan    |      1.25 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Tilted (2s)                                 |     nan    |    nan |    nan |  nan    |      1.77 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Tilted (3s)                                 |     nan    |    nan |    nan |  nan    |      1.74 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Tilted (5s)                                 |     nan    |    nan |    nan |  nan    |      2.01 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (2s)                                    |     nan    |    nan |    nan |  nan    |      1.74 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (3s)                                    |     nan    |    nan |    nan |  nan    |      1.7  |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (5s)                                    |     nan    |    nan |    nan |  nan    |      1.84 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + KS (320p)                                   |     nan    |    nan |    nan |  nan    |      1.89 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (80p)                                   |     nan    |    nan |    nan |  nan    |      1.86 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1                                         |     nan    |    nan |    nan |  nan    |      1.36 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1 + random hadamard                       |     nan    |    nan |    nan |  nan    |      1.45 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1 + hierarchical scales                   |     nan    |    nan |    nan |  nan    |      1.35 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1 + hierarchical scales + random hadamard |     nan    |    nan |    nan |  nan    |      1.34 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2                                         |     nan    |    nan |    nan |  nan    |      1.23 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2 + random hadamard                       |     nan    |    nan |    nan |  nan    |      1.12 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2 + hierarchical scales                   |     nan    |    nan |    nan |  nan    |      1.27 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2 + hierarchical scales + random hadamard |     nan    |    nan |    nan |  nan    |      1.06 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + H                                           |     nan    |    nan |    nan |  nan    |      1.81 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + 4o6                                         |     nan    |    nan |    nan |  nan    |      1.17 |     nan |      nan |      nan |   nan    |
| e4m3 + PZ + RH16                                           |     nan    |    nan |    nan |    0.82 |    nan    |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH16                                        |     nan    |    nan |    nan |  nan    |      1.01 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH32                                        |     nan    |    nan |    nan |  nan    |      1    |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH64                                        |     nan    |    nan |    nan |  nan    |      1.01 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH128                                       |     nan    |    nan |    nan |  nan    |      0.97 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH16                                        |     nan    |    nan |    nan |  nan    |      1.68 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH32                                        |     nan    |    nan |    nan |  nan    |      1.65 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH64                                        |     nan    |    nan |    nan |  nan    |      1.62 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH128                                       |     nan    |    nan |    nan |  nan    |      1.69 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + c90                                         |     nan    |    nan |    nan |  nan    | 439639    |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + c95                                         |     nan    |    nan |    nan |  nan    | 243914    |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + c99                                         |     nan    |    nan |    nan |  nan    |  34426.1  |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + c90                                         |     nan    |    nan |    nan |  nan    | 427842    |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + c95                                         |     nan    |    nan |    nan |  nan    | 186848    |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + c99                                         |     nan    |    nan |    nan |  nan    |  26889.8  |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Ceil                                        |     nan    |    nan |    nan |  nan    |      0.94 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Floor                                       |     nan    |    nan |    nan |  nan    |    738.42 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Ceil                                        |     nan    |    nan |    nan |  nan    |      1.81 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Floor                                       |     nan    |    nan |    nan |  nan    |    399.14 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Hier+FoC+Ceil+RH32                          |     nan    |    nan |    nan |  nan    |      0.95 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Hier+FoC+Ceil+RH32                          |     nan    |    nan |    nan |  nan    |      1.52 |     nan |      nan |      nan |   nan    |

### DeepSeek

|                                                            |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |   BS=512 |
|:-----------------------------------------------------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|---------:|
| e4m3 + PZ                                                  |      12.39 |    nan |    nan |    0.55 |  nan    |     nan |      nan |      nan |   nan    |
| e4m3 + 4o6 + PZ                                            |     nan    |    nan |    nan |    0.44 |  nan    |     nan |      nan |      nan |   nan    |
| e4m3 + PZ + H                                              |     nan    |    nan |    nan |    0.54 |  nan    |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + H                                           |     nan    |    nan |    nan |  nan    |    2.07 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + 4o6                                         |     nan    |    nan |    nan |  nan    |    1.63 |     nan |      nan |      nan |   nan    |
| e4m3 + PZ + RH16                                           |     nan    |    nan |    nan |    0.77 |  nan    |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH16                                        |     nan    |    nan |    nan |  nan    |    1.27 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH32                                        |     nan    |    nan |    nan |  nan    |    1.25 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH64                                        |     nan    |    nan |    nan |  nan    |    1.3  |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH128                                       |     nan    |    nan |    nan |  nan    |    1.34 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH16                                        |     nan    |    nan |    nan |  nan    |    1.88 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH32                                        |     nan    |    nan |    nan |  nan    |    1.8  |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH64                                        |     nan    |    nan |    nan |  nan    |    1.8  |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH128                                       |     nan    |    nan |    nan |  nan    |    1.86 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Ceil                                        |     nan    |    nan |    nan |  nan    |    1.35 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Floor                                       |     nan    |    nan |    nan |  nan    |  129.81 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Ceil                                        |     nan    |    nan |    nan |  nan    |    2    |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Floor                                       |     nan    |    nan |    nan |  nan    |   57.97 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Hier+FoC+Ceil+RH32                          |     nan    |    nan |    nan |  nan    |    1.29 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Hier+FoC+Ceil+RH32                          |     nan    |    nan |    nan |  nan    |    1.85 |     nan |      nan |      nan |   nan    |
| int8 + PZ                                                  |      12.39 |    nan |    nan |  nan    |  nan    |     nan |      nan |      nan |    -0.09 |
| int8 + PZ + RH256                                          |      12.39 |    nan |    nan |  nan    |  nan    |     nan |      nan |      nan |    -0.1  |
| int4 + PZ                                                  |      12.39 |    nan |    nan |  nan    |  nan    |     nan |      nan |      nan |     4.47 |
| int4 + PZ + RH256                                          |      12.39 |    nan |    nan |  nan    |  nan    |     nan |      nan |      nan |     1.64 |
| mxfp4 (e2m1)                                               |      12.39 |    nan |    nan |  nan    |    1.35 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2)                                               |      12.39 |    nan |    nan |  nan    |    2    |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + H                                           |     nan    |    nan |    nan |  nan    |    1.26 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + 4o6                                         |     nan    |    nan |    nan |  nan    |    1.23 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + c90                                         |     nan    |    nan |    nan |  nan    |    1.23 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + c95                                         |     nan    |    nan |    nan |  nan    |    1.27 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + c99                                         |     nan    |    nan |    nan |  nan    |    1.32 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + c90                                         |     nan    |    nan |    nan |  nan    |    1.66 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + c95                                         |     nan    |    nan |    nan |  nan    |    1.81 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + c99                                         |     nan    |    nan |    nan |  nan    |    1.98 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Tilted (2s)                                 |     nan    |    nan |    nan |  nan    |    1.76 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Tilted (3s)                                 |     nan    |    nan |    nan |  nan    |    1.99 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Tilted (5s)                                 |     nan    |    nan |    nan |  nan    |    2.28 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Rot (2s)                                    |     nan    |    nan |    nan |  nan    |    1.63 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Rot (3s)                                    |     nan    |    nan |    nan |  nan    |    1.74 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Rot (5s)                                    |     nan    |    nan |    nan |  nan    |    1.68 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Tilted (2s)                                 |     nan    |    nan |    nan |  nan    |    1.97 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Tilted (3s)                                 |     nan    |    nan |    nan |  nan    |    2.06 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Tilted (5s)                                 |     nan    |    nan |    nan |  nan    |    2.51 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (2s)                                    |     nan    |    nan |    nan |  nan    |    2.02 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (3s)                                    |     nan    |    nan |    nan |  nan    |    2    |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (5s)                                    |     nan    |    nan |    nan |  nan    |    2.09 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + KS (320p)                                   |     nan    |    nan |    nan |  nan    |    2.25 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (80p)                                   |     nan    |    nan |    nan |  nan    |    1.99 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1                                         |     nan    |    nan |    nan |  nan    |    1.63 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1 + random hadamard                       |     nan    |    nan |    nan |  nan    |    1.62 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1 + hierarchical scales                   |     nan    |    nan |    nan |  nan    |    1.62 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1 + hierarchical scales + random hadamard |     nan    |    nan |    nan |  nan    |    1.7  |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2                                         |     nan    |    nan |    nan |  nan    |    1.62 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2 + random hadamard                       |     nan    |    nan |    nan |  nan    |    1.32 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2 + hierarchical scales                   |     nan    |    nan |    nan |  nan    |    1.59 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2 + hierarchical scales + random hadamard |     nan    |    nan |    nan |  nan    |    1.35 |     nan |      nan |      nan |   nan    |

### Qwen

|                                                            |   Baseline |   BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |   BS=512 |
|:-----------------------------------------------------------|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|---------:|
| e4m3 + PZ                                                  |       5.36 |    nan |    nan |    0.55 |  nan    |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + 4o6                                         |     nan    |    nan |    nan |  nan    |    1.23 |     nan |      nan |      nan |   nan    |
| e4m3 + PZ + RH16                                           |     nan    |    nan |    nan |    0.68 |  nan    |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH16                                        |     nan    |    nan |    nan |  nan    |    0.88 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH32                                        |     nan    |    nan |    nan |  nan    |    0.88 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH64                                        |     nan    |    nan |    nan |  nan    |    0.89 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + RH128                                       |     nan    |    nan |    nan |  nan    |    0.88 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH16                                        |     nan    |    nan |    nan |  nan    |    1.29 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH32                                        |     nan    |    nan |    nan |  nan    |    1.29 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH64                                        |     nan    |    nan |    nan |  nan    |    1.28 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + RH128                                       |     nan    |    nan |    nan |  nan    |    1.34 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Ceil                                        |     nan    |    nan |    nan |  nan    |    1.1  |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Floor                                       |     nan    |    nan |    nan |  nan    |  577.15 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Ceil                                        |     nan    |    nan |    nan |  nan    |    1.79 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Floor                                       |     nan    |    nan |    nan |  nan    |  226.02 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Hier+FoC+Ceil+RH32                          |     nan    |    nan |    nan |  nan    |    0.87 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Hier+FoC+Ceil+RH32                          |     nan    |    nan |    nan |  nan    |    1.31 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + H                                           |     nan    |    nan |    nan |  nan    |    0.61 |     nan |      nan |      nan |   nan    |
| int8 + PZ                                                  |       5.36 |    nan |    nan |  nan    |  nan    |     nan |      nan |      nan |     0.02 |
| int8 + PZ + RH256                                          |       5.36 |    nan |    nan |  nan    |  nan    |     nan |      nan |      nan |    -0    |
| int4 + PZ                                                  |       5.36 |    nan |    nan |  nan    |  nan    |     nan |      nan |      nan |    16.08 |
| int4 + PZ + RH256                                          |       5.36 |    nan |    nan |  nan    |  nan    |     nan |      nan |      nan |     1.15 |
| mxfp4 (e2m1)                                               |       5.36 |    nan |    nan |  nan    |    1.1  |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2)                                               |       5.36 |    nan |    nan |  nan    |    1.79 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + H                                           |     nan    |    nan |    nan |  nan    |    1.09 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + 4o6                                         |     nan    |    nan |    nan |  nan    |    0.99 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + c90                                         |     nan    |    nan |    nan |  nan    |    1.01 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + c95                                         |     nan    |    nan |    nan |  nan    |    1.04 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + c99                                         |     nan    |    nan |    nan |  nan    |    1.08 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + c90                                         |     nan    |    nan |    nan |  nan    |    1.44 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + c95                                         |     nan    |    nan |    nan |  nan    |    1.59 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + c99                                         |     nan    |    nan |    nan |  nan    |    1.75 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Tilted (2s)                                 |     nan    |    nan |    nan |  nan    |    0.97 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Tilted (3s)                                 |     nan    |    nan |    nan |  nan    |    1.09 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Tilted (5s)                                 |     nan    |    nan |    nan |  nan    |    1.15 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Rot (2s)                                    |     nan    |    nan |    nan |  nan    |    0.96 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Rot (3s)                                    |     nan    |    nan |    nan |  nan    |    0.92 |     nan |      nan |      nan |   nan    |
| mxfp4 (e2m1) + Rot (5s)                                    |     nan    |    nan |    nan |  nan    |    0.94 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Tilted (2s)                                 |     nan    |    nan |    nan |  nan    |    1.31 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Tilted (3s)                                 |     nan    |    nan |    nan |  nan    |    1.35 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Tilted (5s)                                 |     nan    |    nan |    nan |  nan    |    1.47 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (2s)                                    |     nan    |    nan |    nan |  nan    |    1.28 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (3s)                                    |     nan    |    nan |    nan |  nan    |    1.27 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (5s)                                    |     nan    |    nan |    nan |  nan    |    1.28 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + KS (320p)                                   |     nan    |    nan |    nan |  nan    |    1.32 |     nan |      nan |      nan |   nan    |
| mxfp4 (e1m2) + Rot (80p)                                   |     nan    |    nan |    nan |  nan    |    1.3  |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1                                         |     nan    |    nan |    nan |  nan    |    1.4  |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1 + random hadamard                       |     nan    |    nan |    nan |  nan    |    1.15 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1 + hierarchical scales                   |     nan    |    nan |    nan |  nan    |    1.37 |     nan |      nan |      nan |   nan    |
| wgt e1m2, act e2m1 + hierarchical scales + random hadamard |     nan    |    nan |    nan |  nan    |    1.16 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2                                         |     nan    |    nan |    nan |  nan    |    1.43 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2 + random hadamard                       |     nan    |    nan |    nan |  nan    |    0.99 |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2 + hierarchical scales                   |     nan    |    nan |    nan |  nan    |    1.4  |     nan |      nan |      nan |   nan    |
| wgt e2m1, act e1m2 + hierarchical scales + random hadamard |     nan    |    nan |    nan |  nan    |    0.98 |     nan |      nan |      nan |   nan    |

### Perplexity Gap All Models
![Perplexity Gap All Models](plots/gap_all_models.png)



<!-- RESULTS_END -->

