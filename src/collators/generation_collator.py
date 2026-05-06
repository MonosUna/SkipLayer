from typing import Any


class GenerationCollator:
    def __init__(self, **kwargs):
        pass

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, list]:
        messages_list = [[{"role": "user", "content": item["text"]}] for item in batch]
        return {"messages": messages_list}
