from utils.knn import KNN


def _legacy_postprocess_config(arch):
    knn_cfg = arch["model_params"]["post"]["KNN"]
    if not knn_cfg["use"]:
        return {"name": "none", "params": {}}
    return {"name": "knn", "params": knn_cfg["params"]}


def get_postprocess_config(arch):
    if "postprocess" in arch:
        config = arch["postprocess"]
        if not config.get("enabled", True):
            return {"name": "none", "params": {}}
        return {
            "name": config.get("name", "none"),
            "params": config.get("params", {}),
        }
    return _legacy_postprocess_config(arch)


def build_postprocess(arch, num_classes, dataset_type):
    post_cfg = get_postprocess_config(arch)
    name = post_cfg["name"].lower()

    if name in {"none", "identity"}:
        return None
    if name == "knn":
        return KNN(post_cfg["params"], num_classes, dataset_type)

    raise ValueError(f"Unsupported postprocess method: {post_cfg['name']}")
