"""Build a local Nanbeige4.2 looped-Llama checkpoint dir (symlinked weights, new config)."""
import json, os, shutil, sys
from huggingface_hub import snapshot_download

REPO, REV = "Nanbeige/Nanbeige4.2-3B", "b82e54bd609793562a75cbf9337970a93369eab5"
out = sys.argv[1]
src = snapshot_download(REPO, revision=REV)
os.makedirs(out, exist_ok=True)
for name in os.listdir(src):
    if name.endswith((".safetensors", ".json", ".model", ".txt", ".jinja")) and name != "config.json":
        dst = os.path.join(out, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(src, name)), dst)
ref = json.load(open(os.path.join(src, "config.json")))
keep = ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "rms_norm_eps", "rope_theta", "max_position_embeddings",
        "bos_token_id", "eos_token_id", "pad_token_id", "hidden_act", "num_loops", "tie_word_embeddings"]
cfg = {k: ref[k] for k in keep}
cfg.update(model_type="nanbeige_looped", architectures=["NanbeigeLoopedForCausalLM"],
           attention_bias=False, mlp_bias=False, torch_dtype="bfloat16",
           auto_map={"AutoConfig": "modeling_nanbeige_looped.NanbeigeLoopedConfig",
                     "AutoModelForCausalLM": "modeling_nanbeige_looped.NanbeigeLoopedForCausalLM"})
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "modeling_nanbeige_looped.py"), out)
print("built", out, sorted(os.listdir(out)))
