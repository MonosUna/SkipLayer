from typing import Any


def get_layer_attention_mask(causal_mask_mapping: Any, decoder_layer: Any) -> Any:
    if not isinstance(causal_mask_mapping, dict):
        return causal_mask_mapping

    attention_type = getattr(decoder_layer, "attention_type", None)
    if attention_type is not None:
        return causal_mask_mapping[attention_type]

    if "full_attention" in causal_mask_mapping:
        return causal_mask_mapping["full_attention"]

    if len(causal_mask_mapping) == 1:
        return next(iter(causal_mask_mapping.values()))

    raise KeyError(
        "Unable to resolve attention mask for decoder layer without an attention_type attribute"
    )
