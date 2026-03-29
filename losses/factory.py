import torch
import torch.nn as nn

from losses.bundle import SegmentationLossBundle
from utils.boundary_loss import BoundaryLoss
from utils.lovasz_loss import Lovasz_loss


def _legacy_terms():
    return [
        {"name": "cross_entropy", "weight": 1.0},
        {"name": "boundary", "weight": 1.0},
        {"name": "lovasz", "weight": 1.5},
    ]


def _build_term(name, arch, class_weights):
    ignore_index = arch["dataset"]["ignore_label"]

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
    if name == "lovasz":
        return Lovasz_loss(ignore=ignore_index)
    if name == "boundary":
        return BoundaryLoss()

    raise ValueError(f"Unsupported loss term: {name}")


def build_loss_bundle(arch, class_weights, device):
    if not torch.is_tensor(class_weights):
        class_weights = torch.tensor(class_weights, dtype=torch.float)

    class_weights = class_weights.to(device)
    loss_cfg = arch.get("loss", {})
    terms_cfg = loss_cfg.get("terms", _legacy_terms())

    terms = []
    for term_cfg in terms_cfg:
        name = term_cfg["name"].lower()
        terms.append(
            {
                "name": name,
                "weight": term_cfg.get("weight", 1.0),
                "module": _build_term(name, arch, class_weights).to(device),
            }
        )

    return SegmentationLossBundle(terms).to(device)
