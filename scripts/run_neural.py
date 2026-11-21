"""第二轮神经网络实验入口，保留首轮冻结结果。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["benchmark", "train", "blend", "finalize", "verify", "report"])
    parser.add_argument("--model", choices=["mlp", "ft_transformer", "tabm", "all"], default="all")
    args = parser.parse_args()
    from fraud import neural_experiments as experiments
    if args.stage == "benchmark":
        experiments.benchmark()
    elif args.stage == "train":
        names = list(experiments.neural_config()["models"]) if args.model == "all" else [args.model]
        for kind in names:
            experiments.run_cv(kind)
    else:
        from fraud import neural_finalize
        getattr(neural_finalize, args.stage)()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
