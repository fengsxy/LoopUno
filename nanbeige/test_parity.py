"""Parity: local looped-Llama (sdpa + flex) vs official Nanbeige remote code; plus a flex backward pass."""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
local = sys.argv[1]
tok = AutoTokenizer.from_pretrained(local)
ref = AutoModelForCausalLM.from_pretrained("Nanbeige/Nanbeige4.2-3B", revision="b82e54bd609793562a75cbf9337970a93369eab5",
        trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager").cuda().eval()
text = tok.apply_chat_template([{"role": "user", "content": "Explain why the sky is blue in two sentences."}],
                               add_generation_prompt=True, tokenize=False) + "The sky looks blue because"
ids = tok(text, return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    a = ref(ids, use_cache=False).logits
del ref; torch.cuda.empty_cache()
for impl, dtype in [("sdpa", torch.float32), ("sdpa", torch.bfloat16), ("flex_attention", torch.bfloat16)]:
    m = AutoModelForCausalLM.from_pretrained(local, trust_remote_code=True, torch_dtype=dtype, attn_implementation=impl).cuda().eval()
    with torch.no_grad():
        b = m(ids, use_cache=False).logits.float()
    print(f"{impl:15s} {str(dtype):15s} max|d|={(a-b).abs().max().item():.4f} mean|d|={(a-b).abs().mean().item():.5f} top1={(a.argmax(-1)==b.argmax(-1)).float().mean().item():.4f}")
    if impl == "flex_attention":
        m.train(); m.gradient_checkpointing_enable()
        loss = m(ids, labels=ids, use_cache=False).loss; loss.backward()
        g = m.model.layers[0].self_attn.q_proj.weight.grad
        print("flex train loss", round(loss.item(), 4), "grad finite", bool(torch.isfinite(g).all()), "peak GB", round(torch.cuda.max_memory_allocated()/1e9, 1))
    del m; torch.cuda.empty_cache()
