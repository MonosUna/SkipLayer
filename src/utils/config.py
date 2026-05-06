from omegaconf import OmegaConf


def _coerce_number(value):
    if isinstance(value, bool):
        raise TypeError("Boolean values are not supported in the `mul` resolver.")
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        value = value.strip()
        try:
            return int(value)
        except ValueError:
            return float(value)
    raise TypeError(f"Unsupported value for `mul` resolver: {type(value).__name__}")


def _mul_resolver(*values):
    result = 1
    for value in values:
        result *= _coerce_number(value)
    return result


def register_resolvers() -> None:
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", _mul_resolver)
