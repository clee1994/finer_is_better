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

<!-- RESULTS_END -->

## Notes for Next LLM Agent

This repository evaluates the effect of **Hierarchical Scaling (+H)** on low-bit quantization (FP4/FP8) across four models: Granite-3.3-8B, Llama-3.1-8B, DeepSeek-LLM-7B-Base, and Qwen2.5-14B.

### Current Status:
- **Llama-3.1-8B** and **DeepSeek-LLM-7B-Base** are **100% complete** for all 16 configurations.
*   **Granite** and **Qwen** were interrupted by a server restart and might be missing the last few points of the heavy `4over6` configurations.
- All raw CSV results are stored in the `results/` directory.
- The combined results file is `results/results.csv`.

### Useful Scripts:
- `run_model_all_configs.sh`: Runs the evaluation sweep. It has been updated to run ONLY the new 8 `+ H` configurations to avoid re-running base configs.
- `cleanup_and_plot.py`: Aggregates all `results_*.csv` files from the `results/` directory into `results/results.csv`, cleans duplicates, and updates the README tables and plots.
- `create_final_figure.py`: Generates the final 2x3 publication figure combining perplexity gaps and weight/activation distributions for Granite and Qwen.

### Next Steps:
1.  Check `results/results.csv` for any remaining `NaN` values.
2.  Resume runs for missing points if necessary.
3.  Run `create_final_figure.py` to regenerate the consolidated figure once all data is in.

