import torch
import torch.nn.functional as F
from torch import nn


class SegmentationLossBundle(nn.Module):
    def __init__(self, terms):
        super().__init__()
        self.term_names = []
        self.term_weights = {}
        modules = {}

        for index, term in enumerate(terms):
            key = f"term_{index}_{term['name']}"
            modules[key] = term["module"]
            self.term_names.append((key, term["name"]))
            self.term_weights[key] = float(term["weight"])

        self.modules_map = nn.ModuleDict(modules)

    def forward(self, logits, labels, outputs=None):
        total = logits.new_tensor(0.0)
        loss_items = {}
        probabilities = None

        for key, name in self.term_names:
            module = self.modules_map[key]
            if name == "cross_entropy":
                value = module(logits, labels)
            else:
                if probabilities is None:
                    probabilities = F.softmax(logits, dim=1)
                value = module(probabilities, labels)

            total = total + self.term_weights[key] * value
            loss_items[name] = value

        loss_items["total"] = total
        return total, loss_items
