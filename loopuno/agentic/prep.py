"""Build agentic eval sets: BFCL v3 non-live AST (simple/multiple/parallel/parallel_multiple) and SWE-agent trajectory replay."""
import json, os, random
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

OUT = "data/agentic"
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(os.environ.get("UNO_MODEL_PATH", "models/nanbeige42_3b_uno"), trust_remote_code=True)


def fix_types(schema):
    # BFCL writes JSON-schema "dict"/"float"/"tuple"; normalise for the chat template.
    if isinstance(schema, dict):
        out = {k: fix_types(v) for k, v in schema.items()}
        if out.get("type") == "dict": out["type"] = "object"
        if out.get("type") == "float": out["type"] = "number"
        if out.get("type") == "tuple": out["type"] = "array"
        return out
    if isinstance(schema, list):
        return [fix_types(v) for v in schema]
    return schema


rows = []
for cat in ["simple", "multiple", "parallel", "parallel_multiple"]:
    q = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", f"BFCL_v3_{cat}.json", repo_type="dataset")
    a = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset")
    answers = {r["id"]: r["ground_truth"] for r in map(json.loads, open(a))}
    for r in map(json.loads, open(q)):
        tools = [{"type": "function", "function": fix_types(f)} for f in r["function"]]
        rows.append({"id": r["id"], "category": cat, "messages": r["question"][0], "tools": tools, "ground_truth": answers[r["id"]]})
with open(f"{OUT}/bfcl.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print("bfcl", len(rows))

rng = random.Random(20261025)
p = hf_hub_download("nebius/SWE-agent-trajectories", "data/train-00000-of-00012.parquet", repo_type="dataset")
df = pd.read_parquet(p)
rows, seen = [], 0
for i in rng.sample(range(len(df)), len(df)):
    traj = list(df.iloc[i]["trajectory"])
    msgs = []
    for m in traj:
        role, text = m["role"], m["text"]
        if role == "system": msgs.append({"role": "system", "content": m["system_prompt"]})
        elif role == "user": msgs.append({"role": "user", "content": text})
        elif role == "ai": msgs.append({"role": "assistant", "content": text})
    cuts = [k for k, m in enumerate(msgs) if m["role"] == "assistant" and k >= 3]
    if not cuts: continue
    k = rng.choice(cuts)
    prompt = msgs[:k]
    n = len(tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True))
    if n > 16000: continue
    rows.append({"id": f"{df.iloc[i]['instance_id']}@{k}", "messages": prompt, "prompt_tokens": n, "turn": k})
    if len(rows) == 300: break
with open(f"{OUT}/swe_replay.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
pt = sorted(r["prompt_tokens"] for r in rows)
print("swe_replay", len(rows), "prompt tokens median", pt[len(pt) // 2], "max", pt[-1])
