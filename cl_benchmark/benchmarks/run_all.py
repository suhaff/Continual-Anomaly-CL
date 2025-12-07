# cl_benchmark/benchmarks/run_all.py
"""
Run multiple anomaly-detection configs (different backbones/settings)
using the same memory-bank pipeline from cl_benchmark.cl_benchmark.

This does NOT change your main training code – it just calls run(cfg)
several times with different overrides and prints a summary.
"""

import argparse
import copy
import yaml

from cl_benchmark.cl_benchmark import (
    run as run_anomaly,
    DEFAULT_CFG,
    parse_overrides,
    normalize_device,
)


def load_base_cfg(config_path: str | None, extra_set_args: list[str] | None):
    """Load DEFAULT_CFG + YAML + --set overrides."""
    cfg = DEFAULT_CFG.copy()

    # 1) YAML config (same as cl_benchmark.py)
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)
            if yaml_cfg:
                cfg.update(yaml_cfg)

    # 2) CLI overrides: --set KEY=VALUE ...
    overrides = parse_overrides(extra_set_args)
    cfg.update(overrides)

    # 3) Normalize device
    cfg["DEVICE"] = normalize_device(cfg.get("DEVICE", cfg["DEVICE"]))
    return cfg


def build_method_list(base_outdir: str):
    """
    Define the benchmark variants.

    You can add/remove entries here to test different backbones or
    different memory settings.
    """
    methods = [
        {
            "name": "ResNet18_pretrained",
            "overrides": {
                "BACKBONE": "ResNet18_pretrained",
                "OUTDIR": f"{base_outdir}_resnet18_pretrained",
            },
        },
        {
            "name": "ResNet18_scratch",
            "overrides": {
                "BACKBONE": "ResNet18",
                "OUTDIR": f"{base_outdir}_resnet18_scratch",
            },
        },
        {
            "name": "SimpleCNN",
            "overrides": {
                "BACKBONE": "SimpleCNN",
                "OUTDIR": f"{base_outdir}_simplecnn",
            },
        },
    ]
    return methods


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple anomaly-detection configs (multi-backbone benchmark)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file (same as for cl_benchmark.cl_benchmark)",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        help="Override config values, e.g. DEVICE=cuda MEMORY_SAMPLES_PER_TASK=1500",
    )
    args = parser.parse_args()

    # Base configuration (shared across methods)
    base_cfg = load_base_cfg(args.config, args.set)

    base_outdir = base_cfg.get("OUTDIR", "results/mvtec+loco/Anomaly")
    methods = build_method_list(base_outdir)

    benchmark_summary = []

    for method in methods:
        name = method["name"]
        print("\n\n" + "#" * 60)
        print(f"### Running benchmark method: {name}")
        print("#" * 60 + "\n")

        # Deep copy to avoid in-place modifications
        cfg_m = copy.deepcopy(base_cfg)
        cfg_m.update(method["overrides"])
        cfg_m["DEVICE"] = normalize_device(cfg_m.get("DEVICE", base_cfg["DEVICE"]))

        # Call your existing anomaly pipeline
        auc_list = run_anomaly(cfg_m)

        # Mean AUC over all tasks
        mean_auc = float(sum(auc_list)) / max(len(auc_list), 1)
        benchmark_summary.append((name, mean_auc))

    # Final summary
    print("\n======== BENCHMARK SUMMARY (mean AUC over all tasks) ========")
    for name, mean_auc in benchmark_summary:
        print(f"{name:30s}: {mean_auc:.4f}")


if __name__ == "__main__":
    main()
