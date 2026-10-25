"""Nanbeige4.2 as a looped Llama: the 22-layer stack runs ``num_loops`` times.

Matches the released checkpoint (no LoopSplit / mHC / n-gram): each loop ends with
the shared final RMSNorm, and the normalized state feeds the next loop. Built on
transformers' Llama layers so flex_attention BlockMasks pass straight through.
Training-only: KV caching across loops is not implemented.
"""

from typing import Optional

import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast


class NanbeigeLoopedConfig(LlamaConfig):
    model_type = "nanbeige_looped"

    def __init__(self, num_loops: int = 2, **kwargs):
        self.num_loops = num_loops
        super().__init__(**kwargs)


class NanbeigeLoopedModel(LlamaModel):
    config_class = NanbeigeLoopedConfig

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        num_loops: Optional[int] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if use_cache or past_key_values is not None:
            raise NotImplementedError("NanbeigeLoopedModel is training-only (no KV cache).")
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for _ in range(num_loops or self.config.num_loops):
            for decoder_layer in self.layers:
                # LlamaDecoderLayer is a GradientCheckpointingLayer: checkpointing is handled in __call__.
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
            hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class NanbeigeLoopedForCausalLM(LlamaForCausalLM):
    config_class = NanbeigeLoopedConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = NanbeigeLoopedModel(config)
        self.post_init()
