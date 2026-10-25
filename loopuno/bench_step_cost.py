"""Per-step decode cost on 1 GPU with 4 sequences: AR vs Uno, CUDA graph vs eager, Qwen3-8B vs Nanbeige4.2-3B.
Usage: python loopuno/bench_step_cost.py {qwen|nanbeige} BLOCK {graph|eager} [TEMPERATURE] [NANBEIGE_ADAPTER]"""
import json, os, sys, time, glob, statistics as st
from nano_vllm_uno import LLM, SamplingParams
from pathlib import Path
P = str(Path(__file__).resolve().parents[1])
name, block, eager = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "eager"
temp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
if name == "qwen":
    model = glob.glob(f"{P}/.cache/hf/hub/models--s-sahoo--uno-qwen3-8B/snapshots/*/")[0]
    lora, mask, stops = model + "adapter", 151669, [151645, 151643]
else:
    model = os.environ.get("UNO_MODEL_PATH", f"{P}/models/nanbeige42_3b_uno")
    lora, mask, stops = (sys.argv[5] if len(sys.argv) > 5 else f"{P}/checkpoints/loopuno-b8-lr1e-4/checkpoint-60"), 166107, [166101]
llm = LLM(model, gated_lora_path=lora, enforce_eager=eager, decode_timing=True, max_num_seqs=4,
          max_model_len=8192, max_num_batched_tokens=8192, max_diffusion_block_size=max(block, 2),
          cuda_graph_block_sizes=sorted({1, block}), attention_backend="fa2")
qs = [json.loads(l)["chat_input"] for l in open(f"{P}/data/eval/gsm8k_from_openai.jsonl")][:4]
prompts = [llm.tokenizer.apply_chat_template(q, add_generation_prompt=True, tokenize=True) for q in qs]
sp = SamplingParams(temperature=temp, top_p=0.95 if temp > 0 else None, max_tokens=4096 if temp > 0 else 768, ignore_eos=temp == 0, stop_token_ids=stops,
                    mask_token_id=mask, diffusion_block_size=block)
llm.generate(prompts[:1], sp, use_tqdm=False)  # warmup
llm.reset_decode_timing()
t = time.perf_counter(); outs = llm.generate(prompts, sp, use_tqdm=False); dt = time.perf_counter() - t
tim = llm.get_decode_timing()
toks = sum(len(o["token_ids"]) for o in outs)
fw = sum(o["stats"].get("forwards", 0) for o in outs); acc = sum(o["stats"].get("accepts", 0) for o in outs)
recs = tim["forward_records"]; nsteps = len(tim["step_wall_ms"])
mr = llm.model_runner
res = dict(model=name, block=block, mode="eager" if eager else "graph", tok_s=round(toks / dt), TPF=round(acc / max(fw, 1), 3),
           steps=nsteps, step_wall_ms=round(st.median(tim["step_wall_ms"]), 2), step_gpu_ms=round(st.median(tim["model_gpu_ms"]), 2),
           host_ms=round(st.median(tim["exposed_non_model_ms"]), 2), fwd_per_step=round(len(recs) / max(nsteps, 1), 2),
           graph_hits=getattr(mr, "cuda_graph_hits", None), graph_misses=getattr(mr, "cuda_graph_misses", None),
           rec_example={k: v for k, v in (recs[len(recs)//2] if recs else {}).items()})
print("RESULT", json.dumps(res)); os.makedirs(f"{P}/results/engine_bench", exist_ok=True); json.dump(res, open(f"{P}/results/engine_bench/{name}_b{block}_{res['mode']}_t{temp}.json", "w"))
llm.exit()
