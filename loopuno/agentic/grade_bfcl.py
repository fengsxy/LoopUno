"""Simplified BFCL AST check for Nanbeige's <tool_call><function=..><parameter=..> output.
A call matches when the name matches, every required/given parameter takes one of the allowed values,
and no unexpected parameter appears; parallel categories need a one-to-one match of all calls."""
import json, re, sys
from collections import defaultdict

CALL = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.S)


def parse(text):
    text = text.split("</think>")[-1]
    calls = []
    for name, body in CALL.findall(text):
        args = {}
        for k, v in PARAM.findall(body):
            try: args[k] = json.loads(v)
            except Exception: args[k] = v
        calls.append((name.strip(), args))
    return calls


def norm(v):
    if isinstance(v, str): return v.strip().lower()
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, list): return [norm(x) for x in v]
    if isinstance(v, dict): return {k: norm(x) for k, x in v.items()}
    return v


def match(call, expected):
    (name, args), ((ename, eparams),) = call, expected.items()
    if name != ename: return False
    for k, allowed in eparams.items():
        if k not in args:
            if "" not in allowed: return False
            continue
        if not any(norm(args[k]) == norm(x) for x in allowed if x != ""): return False
    return all(k in eparams for k in args)


def correct(calls, truth):
    if len(calls) != len(truth): return False
    left = list(range(len(calls)))
    for exp in truth:
        hit = next((i for i in left if match(calls[i], exp)), None)
        if hit is None: return False
        left.remove(hit)
    return True


data = {r["id"]: r for r in map(json.loads, open("data/agentic/bfcl.jsonl"))}
res = defaultdict(lambda: [0, 0])
for o in map(json.loads, open(sys.argv[1])):
    r = data[o["id"]]; ok = correct(parse(o["text"]), r["ground_truth"])
    res[r["category"]][0] += ok; res[r["category"]][1] += 1
tot = [sum(v[0] for v in res.values()), sum(v[1] for v in res.values())]
out = {c: round(v[0] / v[1], 4) for c, v in res.items()} | {"overall": round(tot[0] / tot[1], 4), "n": tot[1]}
json.dump(out, open(sys.argv[1].replace(".jsonl", ".grade.json"), "w"), indent=1)
print("GRADE", json.dumps(out))
