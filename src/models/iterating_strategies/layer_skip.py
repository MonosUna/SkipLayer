import torch

from src.models.llms.common import get_layer_attention_mask


class LayerSkipIterationStrategy:
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
        skip_vector: list[int] = []

        if llm.training:
            decisions: list[bool] = [
                bool(llm.layer_skipper.should_skip(hidden_states, i))
                for i in range(len(layers))
            ]
            aligner_start = int(getattr(llm.aligner, "start_layer", 0))
            trainable_indices = [
                i for i in range(aligner_start, len(layers))
                if any(p.requires_grad for p in llm.aligner.parameters())
            ]
            if trainable_indices and not any(decisions[i] for i in trainable_indices):
                forced = int(torch.randint(0, len(trainable_indices), (1,)).item())
                decisions[trainable_indices[forced]] = True
        else:
            decisions = None

        for i, decoder_layer in enumerate(layers):
            if llm.training:
                is_skipping = decisions[i]
            else:
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
