import torch
import torch.nn.functional as F


class CELoss:
    """Cross-entropy loss between final logits and shifted labels.

    The aligner participates in this loss when the iteration strategy
    substitutes one of the decoder layers with it: gradients flow through
    the rest of the network and into the aligner parameters.
    """

    model_forward_kwargs = {
        "compute_logits": True,
        "use_cache": False,
    }

    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch["labels"].long()
        attention_mask = batch["attention_mask"]
        logits = batch["logits"]

        log_probs = F.log_softmax(logits.float(), dim=-1)
        nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        nll = nll * attention_mask

        loss = nll.sum() / attention_mask.sum().clamp(min=1)
        return loss
