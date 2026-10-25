"""Generate on an agentic eval set with AR (block 1) or Uno (block 8); record per-output text, length and decoder stats."""
import argparse, json, os, time
from pathlib import Path
from nano_vllm_uno import LLM, SamplingParams
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lora", required=True)
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--dp", type=int, default=8)
a = ap.parse_args()

P = str(Path(__file__).resolve().parents[2])
rows = [json.loads(l) for l in open(a.data)]
tok = AutoTokenizer.from_pretrained(os.environ.get("UNO_MODEL_PATH", f"{P}/models/nanbeige42_3b_uno"), trust_remote_code=True)
prompts = [tok.apply_chat_template(r["messages"], tools=r.get("tools"), add_generation_prompt=True, tokenize=True) for r in rows]
llm = LLM(os.environ.get("UNO_MODEL_PATH", f"{P}/models/nanbeige42_3b_uno"), gated_lora_path=a.lora, data_parallel_size=a.dp, max_num_seqs=4,
          max_model_len=32768, max_num_batched_tokens=32768, max_diffusion_block_size=max(a.block, 2),
          cuda_graph_block_sizes=sorted({1, a.block}), attention_backend="fa2")
sp = SamplingParams(temperature=a.temperature, top_p=0.95 if a.temperature > 0 else None, max_tokens=a.max_tokens,
                    stop_token_ids=[166101], mask_token_id=166107, diffusion_block_size=a.block)
t = time.perf_counter(); outs = llm.generate(prompts, sp, use_tqdm=False); dt = time.perf_counter() - t
fw = sum(o["stats"].get("forwards", 0) for o in outs); acc = sum(o["stats"].get("accepts", 0) for o in outs)
toks = sum(len(o["token_ids"]) for o in outs)
with open(a.out, "w") as f:
    for r, o in zip(rows, outs):
        f.write(json.dumps({"id": r["id"], "text": o["text"], "n_tokens": len(o["token_ids"]), "stats": o["stats"]}) + "\n")
summary = {"data": a.data, "block": a.block, "lora": a.lora, "temperature": a.temperature, "n": len(rows), "output_tokens": toks,
           "elapsed_s": round(dt, 1), "tok_s": round(toks / dt), "TPF": round(acc / max(fw, 1), 3)}
json.dump(summary, open(a.out.replace(".jsonl", ".summary.json"), "w"), indent=1)
print("SUMMARY", json.dumps(summary))
llm.exit()
