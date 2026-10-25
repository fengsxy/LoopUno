# LoopUno: Turning Agentic Looped LLMs into Diffusion LLMs in One Hour

Adapters: https://huggingface.co/liddlefish/LoopUno-Nanbeige4.2-3B · Results page: https://fengsxy.github.io/LoopUno/

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

The [results page](https://fengsxy.github.io/LoopUno/) (`docs/index.html`) has the method diagram,
scaling with training time, step cost and all runs.

## Agentic tasks

| Task | AR tok/s | Uno step 60 (52 min) | Uno step 240 (3.5 h) | Accuracy AR / Uno |
|---|---|---|---|---|
| BFCL v3 function calling (1000, greedy) | 1653 | 3273 (**1.98×**, TPF 2.34) | 3464 (2.10×, TPF 2.58) | 87.2% / 87.5% |
| SWE-agent trajectory replay (300, T=1.0) | 538 | 848 (**1.58×**, TPF 2.06) | 871 (1.62×, TPF 2.34) | speed only |

## Code

This repository is a fork of [ifm-ai/uno](https://github.com/ifm-ai/uno) at `fdfa90d`; the original README is
kept as [`UNO_README.md`](UNO_README.md). LoopUno adds:

- `nanbeige/`: builds the Uno training wrapper for Nanbeige4.2-3B (SDAR modeling without QK-norm, layers run
  twice) and parity tests against the plain looped model.
- `nano_vllm_uno/models/nanbeige.py`: the looped model for the Uno decoder, with one KV cache per loop.
- `training/`: `UNO_MODEL_PATH` for a local model folder, `--init-adapter` to continue from an adapter, and
  `UNO_SAVE_STEPS` for extra checkpoints.
- `loopuno/`: the scripts behind every number above.

## Reproduce

Install the environment as in [`UNO_README.md`](UNO_README.md#installation) (Python 3.10, FA2). Every script
sources `loopuno/env.sh`, which keeps caches, data, checkpoints and results under the repository root.

```bash
# 1. Assemble the Uno-ready model folder (weights are symlinked from the HF cache; 166107 is the mask token)
python nanbeige/build_uno_model.py models/nanbeige42_3b_uno 166107

# 2. Tokenize 60k OpenThoughts3 conversations
bash loopuno/prep_data.sh

# 3. Train for one hour on 8 GPUs: 60 steps, lr 1e-4, block 8, ZeRO-0 (the two loops share weights)
bash loopuno/train.sh
#    Optional continuation to step 240:
INIT=checkpoints/loopuno-b8-lr1e-4/checkpoint-60 CURRICULUM=loopuno/curriculum_b8_cont180.yaml \
  RUN=loopuno-b8-cont180 bash loopuno/train.sh

# 4. gsm8k: TPF, tok/s and accuracy per checkpoint, plus the AR reference
bash loopuno/eval_gsm8k.sh checkpoints/loopuno-b8-lr1e-4

# 5. Agentic: BFCL v3 and SWE-agent trajectory replay
python loopuno/agentic/prep.py
bash loopuno/agentic/run_all.sh checkpoints/loopuno-b8-lr1e-4/checkpoint-60

# 6. Per-step decode cost, AR vs Uno (1 GPU)
python loopuno/bench_step_cost.py nanbeige 8 graph
```

To skip training, download the adapters from
[liddlefish/LoopUno-Nanbeige4.2-3B](https://huggingface.co/liddlefish/LoopUno-Nanbeige4.2-3B) and pass a
`step60/` or `step240/` folder wherever a checkpoint is expected.

## License

Apache-2.0, as upstream; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Acknowledgements

LoopUno builds on Uno ([Sahoo et al., 2026](https://arxiv.org/abs/2609.04010)) and its code ([ifm-ai/uno](https://github.com/ifm-ai/uno)), applied to Nanbeige4.2-3B ([Nanbeige LLM Lab, 2026](https://arxiv.org/abs/2607.22083); [weights](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)).
