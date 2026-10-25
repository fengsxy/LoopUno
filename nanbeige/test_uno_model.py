"""Check the Nanbeige-Uno model: (a) AR logits match the plain looped-Llama; (b) Uno bd-training forward runs."""
import sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
uno_dir, looped_dir = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(uno_dir)
text = tok.apply_chat_template([{"role": "user", "content": "Explain why the sky is blue in two sentences."}],
                               add_generation_prompt=True, tokenize=False) + "The sky looks blue because"
ids = tok(text, return_tensors="pt").input_ids.cuda()
ref = AutoModelForCausalLM.from_pretrained(looped_dir, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
with torch.no_grad():
    a = ref(ids, use_cache=False).logits.float()
del ref; torch.cuda.empty_cache()
m = AutoModelForCausalLM.from_pretrained(uno_dir, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
print("missing/unexpected check: q_norm present?", hasattr(m.model.layers[0].self_attn.q_norm, "weight"))
with torch.no_grad():
    L = ids.shape[1]; causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=ids.device))[None, None]
    b = m.model(input_ids=ids, attention_mask=causal, use_cache=False).last_hidden_state
    b = m.lm_head(b).float()
print(f"AR parity fp32: max|d|={(a-b).abs().max().item():.5f} top1={(a.argmax(-1)==b.argmax(-1)).float().mean().item():.4f}")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.modeling import validate_uno_model, prepare_uno_layout
validate_uno_model(m); print("validate_uno_model OK")
