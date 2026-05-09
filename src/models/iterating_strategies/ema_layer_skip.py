import torch

from src.models.llms.common import get_layer_attention_mask


class EmaLayerSkipIterationStrategy:
    """Iteration strategy implementing the EMA-compensation method (see
    TODO.md and ``EmaSkipper``).

    Behaviour:

    * **Training** — no layer is skipped at all. This method is
      inference-only; the strategy degrades to a normal forward so that
      hooking it up does not interfere with any auxiliary training.
    * **Prefill** (multi-token forward) — runs every layer fully,
      collects ``(h_in, h_out)`` per layer and hands them to
      ``layer_skipper.record_prefill``, which initializes EMA buffers
      and picks the K candidate layers.
    * **Single-token generation** — for every layer:
        - if ``layer_skipper.should_skip(i)`` returns ``True``, the
          layer's output is taken to be ``h_in + EMA[i]`` (compensated),
          and the layer's KV cache slot is filled by the configured
          ``kv_cache_strategy`` (which for this method is
          ``ProjectKVCacheStrategy`` — projects the compensated hidden
          state through the layer's k_proj/v_proj).
        - otherwise the layer is executed normally and its
          ``(h_in, h_out)`` updates the EMA estimate.
      After the loop, ``layer_skipper.end_step(num_skipped)`` is called
      to update the adaptive ``p_skip`` and tick the step counter.
    """

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

        is_single_token = hidden_states.size(1) == 1
        is_inference = not llm.training

        # Training or prefill at inference time — no skipping. On prefill
        # we additionally record per-layer (h_in, h_out) and hand them to
        # the skipper so it can pick candidates and seed EMA buffers.
        if not (is_inference and is_single_token):
            record_io = is_inference and not is_single_token
            if record_io:
                llm.layer_skipper.reset()
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
                llm.layer_skipper.record_prefill(layer_io)
            skip_vector = [0] * len(layers)
            skip_tensor = torch.tensor(skip_vector, device=hidden_states.device)
            return {
                "hidden_states": hidden_states,
                "past_key_values": past_key_values,
                "skip_tensor": skip_tensor,
            }

        # Single-token inference path.
        skipper = llm.layer_skipper
        skip_vector: list[int] = []
        last_calculated_layer = None
        cos, sin = position_embeddings
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        num_skipped = 0

        for i, decoder_layer in enumerate(layers):
            # KV propagation needs at least one already-computed layer to
            # source from (only matters for ``SimpleKVCachePropagate``;
            # ``ProjectKVCacheStrategy`` projects from the layer itself
            # so the guard does not strictly apply, but skipping the very
            # first layer remains questionable).
            can_skip = last_calculated_layer is not None
            is_skipping = can_skip and skipper.should_skip(i)
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
                skipper.update_ema(i, h_in, hidden_states)
                last_calculated_layer = i

        skipper.end_step(num_skipped)
        skip_tensor = torch.tensor(skip_vector, device=hidden_states.device)
        return {
            "hidden_states": hidden_states,
            "past_key_values": past_key_values,
            "skip_tensor": skip_tensor,
        }
