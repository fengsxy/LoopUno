"""Build models/nanbeige4.2-3b-uno: Uno's SDAR modeling adapted to Nanbeige4.2 (no qk-norm, 2 loops)."""
import json, os, re, shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download

out = Path(sys.argv[1])
nb = Path(snapshot_download("Nanbeige/Nanbeige4.2-3B", revision="b82e54bd609793562a75cbf9337970a93369eab5"))
sdar = Path(snapshot_download("s-sahoo/uno-qwen3-8B", revision="b8a7577b3223bdcf2b3af0f2fc6e95258b3bbc29"))
out.mkdir(parents=True, exist_ok=True)

for f in nb.iterdir():
    if f.suffix in (".safetensors", ".json", ".model", ".txt", ".jinja") and f.name != "config.json":
        dst = out / f.name
        if not dst.exists():
            dst.symlink_to(f.resolve())
for name in ("configuration_sdar.py", "fused_linear_diffusion_cross_entropy.py"):
    shutil.copy(sdar / name, out / name)

src = (sdar / "modeling_sdar.py").read_text()
# 1) Nanbeige is Llama-style: no per-head q/k RMSNorm.
for nm in ("q_norm", "k_norm"):
    old = f"        self.{nm} = SDARRMSNorm(self.head_dim, eps=config.rms_norm_eps)\n"
    assert old in src, nm
    src = src.replace(old, f"        self.{nm} = (SDARRMSNorm(self.head_dim, eps=config.rms_norm_eps)\n"
                           f"                        if getattr(config, \"use_qk_norm\", True) else nn.Identity())\n")
# 2) Looped stack: run the layers num_loops times, shared final norm after every loop.
head = "        for decoder_layer in self.layers[: self.config.num_hidden_layers]:\n"
tail = "        hidden_states = self.norm(hidden_states)\n\n        # add hidden states from the last decoder layer\n"
i, j = src.index(head), src.index(tail)
body = "".join("    " + ln if ln.strip() else ln for ln in src[i:j].splitlines(True))
loop = ('        num_loops = getattr(self.config, "num_loops", 1)\n'
        '        if num_loops > 1 and use_cache:\n'
        '            raise NotImplementedError("KV caching across loops is not supported (training-only).")\n'
        '        for loop_idx in range(num_loops):\n' + body +
        '            if loop_idx < num_loops - 1:\n'
        '                hidden_states = self.norm(hidden_states)\n\n')
src = src[:i] + loop + src[j:]
(out / "modeling_sdar.py").write_text(src)

ref = json.loads((nb / "config.json").read_text())
cfg = json.loads((sdar / "config.json").read_text())
for k in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
          "num_key_value_heads", "head_dim", "rms_norm_eps", "rope_theta", "max_position_embeddings",
          "bos_token_id", "eos_token_id", "pad_token_id", "tie_word_embeddings", "num_loops"):
    cfg[k] = ref[k]
cfg.update(use_qk_norm=False, max_window_layers=ref["num_hidden_layers"], mask_token_id=int(sys.argv[2]),
           block_size=8, use_cache=False, noise="uniform")
cfg.pop("rope_scaling", None)
(out / "config.json").write_text(json.dumps(cfg, indent=2))
print("built", out)
