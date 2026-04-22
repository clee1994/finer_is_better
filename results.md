# Perplexity Gap Results

This document tracks the perplexity gap (Quantized Perplexity - Baseline Perplexity) for our four target models across different microscale block sizes.
Configuration: NVFP4 elements, FP8 scales (mode 152), and **no attention layers quantized**.

| Model | Baseline | BS=4 | BS=8 | BS=16 | BS=32 | BS=64 | BS=128 | BS=256 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3.1 8B** | 6.2438 | +1.7384 | +1.4256 | +1.2911 | +1.2159 | +1.2343 | +1.3500 | +1.5325 |
| **Granite 3.3 8B** | 4.7191 | +1.1498 | +1.0061 | +0.9254 | +0.9250 | +1.0100 | +1.1999 | +1.5575 |
| **Qwen 2.5 14B** | 5.3577 | +1.3907 | +1.2215 | +1.1188 | +1.1582 | +1.2571 | +1.4940 | +2.0615 |
| **DeepSeek 7B** | 12.2778 | +2.6772 | +2.1138 | +1.7812 | +1.6175 | +1.6540 | +1.8505 | +2.1729 |

### **Key Findings**
1.  **Universal Anomaly**: All four models exhibit the non-monotonic U-shaped curve (perplexity gap drops and then rises as block size increases).
2.  **Optimal Block Size**: For Qwen, the optimal block size is **16**. For Llama, Granite, and DeepSeek, it is **32** (though Granite is very flat between 16 and 32).
3.  **DeepSeek Baseline**: As noted, the DeepSeek baseline is relatively high, but the U-shaped trend is still clearly preserved.
