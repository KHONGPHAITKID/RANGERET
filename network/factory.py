def _legacy_model_config(arch):
    params = arch["model_params"]
    return {
        "name": params.get("model_architecture", "rangeret"),
        "params": params,
    }


def get_model_config(arch):
    if "model" in arch:
        config = arch["model"]
        return {
            "name": config["name"],
            "params": config.get("params", {}),
        }
    return _legacy_model_config(arch)


def get_model_name(arch):
    return get_model_config(arch)["name"]


def build_model(arch, resolution, num_classes, activate_recurrent=False):
    model_cfg = get_model_config(arch)
    name = model_cfg["name"].lower()
    params = model_cfg["params"]

    if name in {"rangeret", "range_ret"}:
        from network.rangeret import RangeRet

        return RangeRet(params, resolution, num_classes, activate_recurrent=activate_recurrent)
    if name == "mambarv":
        from network.mambarv import MambaRV

        return MambaRV(params, resolution, num_classes, activate_recurrent=activate_recurrent)

    raise ValueError(f"Unsupported model architecture: {model_cfg['name']}")
