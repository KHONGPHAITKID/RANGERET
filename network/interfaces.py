from typing import Any, Dict

import torch
from torch import nn


class SegmentationModel(nn.Module):
    """Base class for segmentation models returning a standardized output."""


def normalize_model_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, dict):
        if "logits" not in output:
            raise KeyError("Model output dict must contain a 'logits' entry.")
        return output

    if torch.is_tensor(output):
        return {"logits": output, "aux": {}}

    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def get_logits(output: Any) -> torch.Tensor:
    normalized = normalize_model_output(output)
    logits = normalized["logits"]
    if not torch.is_tensor(logits):
        raise TypeError("Model logits must be a torch.Tensor.")
    if logits.dim() != 4:
        raise ValueError(f"Model logits must have shape (B, C, H, W), got {tuple(logits.shape)}")
    return logits
