import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope_k(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # cos/sin shape: (B, S, head_dim) → unsqueeze head dim.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (k * cos) + (_rotate_half(k) * sin)


class ProjectKVCacheStrategy:
    """Fill the KV cache slot of a skipped layer by projecting that
    layer's *compensated* input hidden state through its own
    ``input_layernorm`` + ``self_attn.{k_proj, v_proj}`` + ``k_norm`` +
    RoPE. This keeps the cache "aligned" with what the layer would have
    produced, so subsequent (non-skipped) attentions on later positions
    can still attend to past positions through this slot.

    The iteration strategy is responsible for passing through ``kwargs``:

    * ``decoder_layer`` — the actual ``Qwen3DecoderLayer`` whose KV slot
      we are filling;
    * ``hidden_states`` — the **compensated** hidden state we want to
      project (i.e. ``h_in + EMA[i]``, the same tensor that the
      iteration strategy uses as the layer's "output");
    * ``position_embeddings`` — ``(cos, sin)`` pair from the rotary
      embedding for the current ``cache_position``.

    Length of the cache stays the same across all layers because we
    write exactly one (k, v) entry per skipped step, just like a normal
    layer would.
    """

    def __call__(self, llm, past_key_values, last_layer, cache_kwargs, *args, **kwargs):
        decoder_layer = kwargs.get("decoder_layer")
        hidden_states = kwargs.get("hidden_states")
        position_embeddings = kwargs.get("position_embeddings")
        layer_idx = kwargs.get("layer_idx")
        if decoder_layer is None or hidden_states is None or position_embeddings is None:
            raise ValueError(
                "ProjectKVCacheStrategy requires decoder_layer, hidden_states "
                "and position_embeddings in kwargs."
            )

        attn = decoder_layer.self_attn
        # Pre-attention layer norm (Qwen3 puts it on the decoder layer level).
        normed = decoder_layer.input_layernorm(hidden_states)

        # Linear projections.
        k = attn.k_proj(normed)
        v = attn.v_proj(normed)

        # Reshape to (B, num_kv_heads, S, head_dim).
        bsz, q_len, _ = k.shape
        head_dim = attn.head_dim
        num_kv_heads = k.shape[-1] // head_dim
        k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # Qwen3 applies an RMSNorm to k *before* RoPE.
        k_norm = getattr(attn, "k_norm", None)
        if k_norm is not None:
            k = k_norm(k)

        # RoPE on k.
        cos, sin = position_embeddings
        k = _apply_rope_k(k, cos, sin)

        idx = layer_idx if layer_idx is not None else last_layer + 1
        past_key_values.update(k, v, layer_idx=idx, cache_kwargs=cache_kwargs)
        return past_key_values
