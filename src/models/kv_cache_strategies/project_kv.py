import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope_k(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (k * cos) + (_rotate_half(k) * sin)


class ProjectKVCacheStrategy:
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
        normed = decoder_layer.input_layernorm(hidden_states)

        k = attn.k_proj(normed)
        v = attn.v_proj(normed)

        bsz, q_len, _ = k.shape
        head_dim = attn.head_dim
        num_kv_heads = k.shape[-1] // head_dim
        k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        k_norm = getattr(attn, "k_norm", None)
        if k_norm is not None:
            k = k_norm(k)

        cos, sin = position_embeddings
        k = _apply_rope_k(k, cos, sin)

        idx = layer_idx if layer_idx is not None else last_layer + 1
        past_key_values.update(k, v, layer_idx=idx, cache_kwargs=cache_kwargs)
        return past_key_values
