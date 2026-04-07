import argparse
import time
from contextlib import nullcontext

import torch
import yaml

from network.factory import build_model, get_model_config, get_model_name
from network.interfaces import get_logits
from utils.checkpoint import load_model_weights


def parse_args():
    parser = argparse.ArgumentParser("./benchmark.py")
    parser.add_argument(
        "--config",
        type=str,
        default="config/RangeRet-semantickitti.yaml",
        help="Architecture yaml cfg file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional model checkpoint to load before benchmarking.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Device to use for benchmarking.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Synthetic batch size used for the benchmark.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional input height override. Defaults to config sensor height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional input width override. Defaults to config sensor width.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations before measuring latency.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use automatic mixed precision during benchmarking on CUDA.",
    )
    parser.add_argument(
        "--no-flops",
        action="store_true",
        default=False,
        help="Skip FLOPs estimation.",
    )
    return parser.parse_args()


def load_arch_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_input_shape(arch, batch_size, height_override=None, width_override=None):
    model_cfg = get_model_config(arch)
    input_dim = model_cfg["params"].get("input_dim", 5)
    sensor_cfg = arch["dataset"]["sensor"]["img_prop"]
    height = height_override or sensor_cfg["height"]
    width = width_override or sensor_cfg["width"]
    return batch_size, input_dim, height, width


def maybe_autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def profile_flops(model, sample, device, use_fp16=False):
    # PyTorch profiler only provides FLOPs for selected operators. Treat this as an approximation.
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.inference_mode():
        with maybe_autocast(device, use_fp16):
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                _ = model(sample)

    total_flops = 0
    for event in prof.key_averages():
        event_flops = getattr(event, "flops", 0)
        if event_flops is not None:
            total_flops += int(event_flops)
    return total_flops


def benchmark_latency(model, sample, device, warmup, iters, use_fp16=False):
    model.eval()

    with torch.inference_mode():
        for _ in range(warmup):
            with maybe_autocast(device, use_fp16):
                outputs = model(sample)
                _ = get_logits(outputs)
            synchronize_if_needed(device)

        synchronize_if_needed(device)
        start = time.perf_counter()
        for _ in range(iters):
            with maybe_autocast(device, use_fp16):
                outputs = model(sample)
                _ = get_logits(outputs)
            synchronize_if_needed(device)
        total_time = time.perf_counter() - start

    avg_latency = total_time / iters
    throughput = sample.shape[0] * iters / total_time
    return avg_latency, throughput


def main():
    args = parse_args()
    arch = load_arch_config(args.config)
    device = resolve_device(args.device)

    model_name = get_model_name(arch)
    model_cfg = get_model_config(arch)
    input_shape = get_input_shape(arch, args.batch_size, args.height, args.width)
    resolution = input_shape[2:]
    num_classes = arch["dataset"]["num_classes"]

    print("----------")
    print("BENCHMARK")
    print("config", args.config)
    print("model", model_name)
    print("device", device)
    print("input_shape", input_shape)
    print("num_classes", num_classes)
    print("warmup", args.warmup)
    print("iters", args.iters)
    print("fp16", args.fp16)
    print("----------")

    if args.fp16 and device.type != "cuda":
        raise ValueError("--fp16 is only supported for CUDA benchmarking.")
    if device.type == "cpu" and model_cfg["params"].get("backbone", "").lower() == "retnet":
        raise ValueError("The current RetNet implementation assumes CUDA during initialization. Use --device cuda for RangeRet/RetNet benchmarks.")

    sample = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        model = build_model(arch, resolution, num_classes)

    model = model.to(device)
    model.eval()

    if args.checkpoint is not None:
        load_model_weights(model, args.checkpoint, strict=True, map_location=device)
        print(f"Loaded checkpoint from {args.checkpoint}")

    total_params, trainable_params = count_parameters(model)

    avg_latency, fps = benchmark_latency(
        model=model,
        sample=sample,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        use_fp16=args.fp16,
    )

    flops = None
    if not args.no_flops:
        try:
            flops = profile_flops(model, sample, device=device, use_fp16=args.fp16)
        except Exception as exc:
            print(f"FLOPs profiling failed: {exc}")

    logits = get_logits(model(sample))

    print()
    print("Results")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Latency: {avg_latency * 1000.0:.3f} ms / batch")
    print(f"FPS: {fps:.2f}")
    if flops is None:
        print("Approx. FLOPs: unavailable")
    else:
        print(f"Approx. FLOPs: {flops:,}")
        print(f"Approx. GFLOPs: {flops / 1e9:.3f}")


if __name__ == "__main__":
    main()
