import torch

from src.models.llms.common import get_layer_attention_mask


class EmaLayerSkipIterationStrategy:
    def __init__(self, kv_cache_strategy):
        self.kv_cache_strategy = kv_cache_strategy

    def _full_forward(
        self,
        layers,
        hidden_states,
        causal_mask_mapping,
        position_ids,
        past_key_values,
        use_cache,
        cache_position,
        position_embeddings,
        record_io,
        **kwargs,
    ):
        layer_io: list[tuple[torch.Tensor, torch.Tensor]] = []
        for decoder_layer in layers:
            h_in = hidden_states
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
            if record_io:
                layer_io.append((h_in.detach(), hidden_states.detach()))
        return hidden_states, layer_io

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
            raise ValueError("EmaLayerSkipIterationStrategy requires a layer_skipper")

        skipper = llm.layer_skipper
        has_ema_api = (
            hasattr(skipper, "reset")
            and hasattr(skipper, "record_prefill")
            and hasattr(skipper, "update_ema")
            and hasattr(skipper, "compensate")
            and hasattr(skipper, "end_step")
        )

        def _should_skip(i: int) -> bool:
            fn = getattr(skipper, "should_skip", None)
            if fn is None:
                return False
            try:
                return bool(fn(i))
            except TypeError:
                return bool(fn(hidden_states, i))

        is_single_token = hidden_states.size(1) == 1
        is_inference = not llm.training

        if not (is_inference and is_single_token):
            record_io = is_inference and not is_single_token and has_ema_api
            if record_io:
                skipper.reset()
            hidden_states, layer_io = self._full_forward(
                layers,
                hidden_states,
                causal_mask_mapping,
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                position_embeddings,
                record_io=record_io,
                **kwargs,
            )
            if record_io:
                skipper.record_prefill(layer_io)
            skip_vector = [0] * len(layers)
            skip_tensor = torch.tensor(skip_vector, device=hidden_states.device)
            return {
                "hidden_states": hidden_states,
                "past_key_values": past_key_values,
                "skip_tensor": skip_tensor,
            }

        skip_vector: list[int] = []
        last_calculated_layer = None
        cos, sin = position_embeddings
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        num_skipped = 0

        for i, decoder_layer in enumerate(layers):
            can_skip = last_calculated_layer is not None and has_ema_api
            is_skipping = can_skip and _should_skip(i)
            skip_vector.append(int(is_skipping))

            h_in = hidden_states
            if is_skipping:
                h_out = skipper.compensate(i, h_in)
                self.kv_cache_strategy(
                    llm,
                    past_key_values,
                    last_calculated_layer,
                    cache_kwargs,
                    decoder_layer=decoder_layer,
                    hidden_states=h_out,
                    position_embeddings=position_embeddings,
                    layer_idx=i,
                    start=i,
                    until=i + 1,
                )
                hidden_states = h_out
                num_skipped += 1
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=get_layer_attention_mask(
                        causal_mask_mapping, decoder_layer
                    ),
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                if has_ema_api:
                    skipper.update_ema(i, h_in, hidden_states)
                last_calculated_layer = i

        if has_ema_api:
            skipper.end_step(num_skipped)
        skip_tensor = torch.tensor(skip_vector, device=hidden_states.device)
        return {
            "hidden_states": hidden_states,
            "past_key_values": past_key_values,
            "skip_tensor": skip_tensor,
        }
