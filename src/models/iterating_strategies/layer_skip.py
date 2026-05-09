import torch

from src.models.llms.common import get_layer_attention_mask


class LayerSkipIterationStrategy:
    """Iterate decoder layers and optionally substitute one (training) or
    several (inference) of them with the learnable aligner.

    Training: a single random layer is replaced by the aligner. The output
    of the aligner becomes the input of the following decoder layers, so
    gradients flow through the aligner via the rest of the network and the
    final cross-entropy loss.

    Inference (single-token decoding only): the layer skipper decides per
    layer whether to substitute that layer with the aligner. KV cache for
    skipped layers is filled by the configured KV cache strategy so that
    subsequent attention layers can still attend to past tokens.
    """

    def __init__(self, kv_cache_strategy):
        self.kv_cache_strategy = kv_cache_strategy

    def __call__(
        self,
        llm,
        layers,
        hidden_states,
        causal_mask_mapping,
        position_ids,
        past_key_values,
        use_cache,
        cache_position,
        position_embeddings,
        **kwargs,
    ):
        if llm.layer_skipper is None:
            raise ValueError("LayerSkipIterationStrategy requires a layer_skipper to be defined")

        last_calculated_layer = None
        skip_vector = []

        skip_layer = None
        if llm.training:
            skip_layer = torch.randint(0, len(layers), (1,)).item()

        for i, decoder_layer in enumerate(layers):
            if llm.training:
                is_skipping = i == skip_layer
            else:
                # KV propagation requires single-token decoding *and* at least
                # one previous layer that produced cache entries to copy from.
                is_skipping = (
                    hidden_states.size(1) == 1
                    and last_calculated_layer is not None
                    and llm.layer_skipper.should_skip(hidden_states, i)
                )
            skip_vector.append(int(is_skipping))

            if is_skipping:
                raw_hidden_states = hidden_states
                aligned_hidden_states = llm.aligner(raw_hidden_states, i)
                if not llm.training:
                    cos, sin = position_embeddings
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    self.kv_cache_strategy(
                        llm,
                        past_key_values,
                        last_calculated_layer,
                        cache_kwargs,
                        decoder_layer=decoder_layer,
                        hidden_states=aligned_hidden_states,
                        aligned_hidden_states=aligned_hidden_states,
                        raw_hidden_states=raw_hidden_states,
                        position_embeddings=position_embeddings, 
                        layer_idx=i,
                        start=i,
                        until=i + 1,
                    )
                hidden_states = aligned_hidden_states
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=get_layer_attention_mask(causal_mask_mapping, decoder_layer),
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                last_calculated_layer = i

        skip_tensor = torch.tensor(skip_vector, device=hidden_states.device)
        return {
            "hidden_states": hidden_states,
            "past_key_values": past_key_values,
            "skip_tensor": skip_tensor,
        }
