"""执行用户已确认的 256 条 GPT-6.2 Sol xhigh 小样本实验，或离线复算结果。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud.llm_pilot import evaluate, execute, load_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--execute", action="store_true", help="实际调用已确认的 API 请求，已完成批次不会再次调用")
    action.add_argument("--evaluate-only", action="store_true", help="仅用已保存的 API 响应离线复算")
    action.add_argument("--check-only", action="store_true", help="仅检查固定请求，不读密钥，不调用 API")
    parser.add_argument("--limit-batches", type=int, choices=range(1, 9), default=8, help="执行到第几批，包含该批")
    parser.add_argument("--start-batch", type=int, choices=range(1, 9), default=1, help="从第几批开始，用于不同批次独立运行")
    parser.add_argument("--retry-token-limit", action="store_true", help="仅补跑一个因输出 token 上限截断的批次，保留失败记录并提高生成预算")
    args = parser.parse_args()
    if args.check_only:
        _, _, batches = load_protocol()
        print(f"固定请求核验通过：{len(batches)} 批，256 条样本；未调用 API。")
    elif args.evaluate_only:
        evaluate()
    else:
        execute(args.limit_batches, args.start_batch, args.retry_token_limit)
        if args.limit_batches == 8 and args.start_batch == 1:
            evaluate()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
