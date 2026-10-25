# LoopUno: Turning Agentic Looped LLMs into Diffusion LLMs in One Hour

Longxuan Yu (University of California, Riverside), Siheng Xiong (Georgia Institute of Technology)

A gated LoRA trained for 52 minutes on one 8×A100 node teaches the looped agentic model
Nanbeige4.2-3B (22 layers × 2 loops) to draft its own next tokens with
[Uno](https://arxiv.org/abs/2609.04010)-style discrete diffusion, then verify them losslessly.

| Checkpoint (lr 1e-4) | Training | TPF | Decode tok/s | vs AR | gsm8k acc |
|---|---|---|---|---|---|
| AR baseline | – | 1.00 | 1545 | 1.00× | 96.0% |
| step 16 | 14 min, 8M tokens | 1.62 | 1753 | 1.13× | 94.5% |
| **step 60** | **52 min, 31M tokens** | **2.04** | **2274** | **1.47×** | 93.0% |
| step 240 | 3.5 h, 126M tokens | 2.18 | 2234 | 1.45× | 96.0% |

gsm8k first 200 problems, temperature 1.0, block size 8, 8×A100-40GB. Verification is lossless,
so accuracy differences are sampling noise (±1.7% at n=200).

Open `index.html` for the full results page: method diagram, scaling with training time,
speedup, step cost and all runs. Agentic evaluations (BFCL function calling and SWE-agent
trajectory replay) are in progress.
