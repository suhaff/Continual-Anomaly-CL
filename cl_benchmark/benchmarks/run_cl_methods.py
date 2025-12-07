# cl_benchmark/benchmarks/run_cl_methods.py

import copy
import argparse
import yaml

from cl_benchmark.cl_train import run as run_cl


CL_METHODS = ["Finetune", "Replay", "EWC", "LwF", "GPM"]


def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    summary = {}

    for method in CL_METHODS:
        print("\n" + "#" * 60)
        print(f"### Running CL method: {method}")
        print("#" * 60 + "\n")

        cfg = copy.deepcopy(base_cfg)
        cfg["CL_METHOD"] = method

        acc, auc = run_cl(cfg)
        summary[method] = (acc, auc)

    print("\n======== CL BENCHMARK SUMMARY (final mean ACC / AUC) ========")
    for m, (acc, auc) in summary.items():
        print(f"{m:10s}: ACC={acc:.2f}%  |  AUC={auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
