import torch


def _load_state(path, map_location=None):
    return torch.load(path, map_location=map_location)


def _extract_state_dict(state):
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def load_model_weights(model, path, strict=True, map_location=None):
    state = _extract_state_dict(_load_state(path, map_location=map_location))
    model.load_state_dict(state, strict=strict)


def load_pretrained_weights(model, path, component=None, strict=True, map_location=None):
    if component in (None, "", "model", "full"):
        load_model_weights(model, path, strict=strict, map_location=map_location)
        return

    target = getattr(model, component, None)
    if target is None:
        raise AttributeError(f"Model does not have a '{component}' component for pretrained loading.")

    state = _extract_state_dict(_load_state(path, map_location=map_location))
    target.load_state_dict(state, strict=strict)
