# Investigation Notes

## UE5M3 Minimum Scale in Prevent Zero

**Context**: 
In the current master sweep (launched ~20:41Z on 2026-04-23), the `prevent_zero` mode bounds the minimum scale at $2^{-9}$ for *both* standard FP8 and UE5M3 formats. This was done to keep the floor consistent.

However, the smallest non-zero subnormal representable in UE5M3 is actually $2^{-17}$. 

**Action Item**:
The code in `torch_eval.py` has been updated (commit `81a4ca6`) to use $2^{-17}$ as the floor for UE5M3 while keeping $2^{-9}$ for standard FP8. We need to run a sweep with this new logic to see if allowing smaller scales improves recovery for models like Qwen that might be sensitive to scale clipping.
