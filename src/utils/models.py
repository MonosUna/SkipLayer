import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import AutoConfig


def inject_model_config(cfg: DictConfig) -> None:
    model_cfg = cfg.llm.model_loading
    config = AutoConfig.from_pretrained(
        model_cfg.pretrained_model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    injected_config = {
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
    }
    if hasattr(config, "num_attention_heads"):
        num_attention_heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // num_attention_heads)
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
        injected_config["num_attention_heads"] = num_attention_heads
        injected_config["num_key_value_heads"] = num_key_value_heads
        injected_config["head_dim"] = head_dim
        injected_config["kv_hidden_size"] = head_dim * num_key_value_heads
    if hasattr(config, "intermediate_size"):
        injected_config["intermediate_size"] = config.intermediate_size

    with open_dict(cfg):
        cfg.model = OmegaConf.create(injected_config)


def create_model(cfg: DictConfig):
    model_cfg = cfg.llm.model_loading
    print(f"\nLoading pretrained model: {model_cfg.pretrained_model_name_or_path}")

    model_class = hydra.utils.get_class(cfg.llm._target_)
    dtype = hydra.utils.get_object(model_cfg.dtype)

    model = model_class.from_pretrained(
        model_cfg.pretrained_model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
        dtype=dtype,
        device_map=model_cfg.device_map,
    )
    print(f"Model loaded: {model.__class__.__name__}")

    dtype = next(model.parameters()).dtype

    aligner = hydra.utils.instantiate(cfg.aligner).to(device=cfg.device, dtype=dtype)
    layer_skipper = hydra.utils.instantiate(cfg.layer_skipper).to(cfg.device, dtype=dtype)
    iteration_strategy = hydra.utils.instantiate(cfg.iterating_strategy)

    model.freeze_llm_parameters()
    for p in aligner.parameters():
        p.requires_grad = True

    model.model.aligner = aligner
    model.model.layer_skipper = layer_skipper
    model.model.iteration_strategy = iteration_strategy

    print(f"All components initialized on device: {cfg.device}")
    return model


def iter_trainable_parameters(model):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def get_parameter_statistics(model) -> dict[str, float]:
    total_params = 0
    trainable_params = 0
    total_bytes = 0
    trainable_bytes = 0

    for parameter in model.parameters():
        param_count = parameter.numel()
        param_bytes = param_count * parameter.element_size()
        total_params += param_count
        total_bytes += param_bytes
        if parameter.requires_grad:
            trainable_params += param_count
            trainable_bytes += param_bytes

    gib = 1024**3
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_gib": total_bytes / gib,
        "trainable_gib": trainable_bytes / gib,
    }
