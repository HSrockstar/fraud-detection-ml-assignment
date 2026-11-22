"""项目入口。所有路径相对于项目，不依赖调用时的工作目录。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["eda", "baseline", "optimize", "finalize", "extensions", "report", "verify", "all"])
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()
    from fraud import eda, experiments

    frozen = (experiments.ROOT / "reports/selection.json").exists()
    if args.stage == "eda" or (args.stage == "all" and not frozen):
        eda.run()
    if args.stage == "baseline" or (args.stage == "all" and not frozen):
        experiments.baseline()
    if args.stage == "optimize" or (args.stage == "all" and not frozen):
        experiments.optimize(args.trials)
    if args.stage in {"finalize", "all"}:
        from fraud.finalize import run
        run()
    if args.stage in {"extensions", "all"}:
        from fraud.extensions import run
        run()
    if args.stage in {"report", "all"}:
        from fraud.reporting import run
        run()
    if args.stage in {"verify", "all"}:
        from fraud.verify import run
        run()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
