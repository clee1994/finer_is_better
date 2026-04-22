# Finer is Better: NVFP4 Quantization Evaluation

This directory contains our consolidated codebase for evaluating baseline perplexity and custom NVFP4 quantization with microscaling on large language models, intended to reproduce findings from paper 2601.19026.

## Setup

To set up the environment on a new machine or VM, follow these steps:

1.  **Create and Activate Virtual Environment**:
    ```bash
    python3 -m venv ~/quant_eval_venv
    source ~/quant_eval_venv/bin/activate
    pip install --upgrade pip
    pip install torch transformers datasets tqdm pandas tabulate
    ```

2.  **Clone and Install Paper\s Repository**:

## Results
<!-- RESULTS_START -->

| Model          |   Baseline |    BS=4 |   BS=8 |   BS=16 |   BS=32 |   BS=64 |   BS=128 |   BS=256 |
|:---------------|-----------:|--------:|-------:|--------:|--------:|--------:|---------:|---------:|
| Llama 3.1 8B   |     6.2438 | 1.7384  | 1.4256 |  1.2911 |  1.2159 |  1.2343 |   1.35   |   1.5325 |
| Granite 3.3 8B |     4.7191 | 1.1498  | 1.0061 |  0.9254 |  0.925  |  1.01   |   1.1999 |   1.5575 |
| Qwen 2.5 14B   |     5.3577 | 1.3907  | 1.2215 |  1.1188 |  1.1582 |  1.2571 |   1.494  |   2.0615 |
| DeepSeek 7B    |    12.2778 | 2.67717 | 2.1138 |  1.7812 |  1.6175 |  1.654  |   1.8505 |   2.1729 |

<!-- RESULTS_END -->
