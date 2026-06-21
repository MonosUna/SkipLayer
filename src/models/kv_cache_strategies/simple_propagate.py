class SimpleKVCachePropagate:
    def __call__(self, llm, past_key_values, last_layer, cache_kwargs, *args, **kwargs):
        start = kwargs.get("start")
        until = kwargs.get("until")

        layers = past_key_values.layers
        src = layers[last_layer]

        k_last = src.keys[:, :, -1:, :]
        v_last = src.values[:, :, -1:, :]

        if start is None:
            start = last_layer + 1
        if until is None:
            until = len(layers)

        for j in range(start, until):
            past_key_values.update(
                k_last,
                v_last,
                layer_idx=j,
                cache_kwargs=cache_kwargs,
            )

        return past_key_values
