"""载入本项目训练好的最终模型，对具有相同特征的新 CSV 预测。"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from fraud.models import predict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--round",choices=["round1","round2"],default="round1")
    args=parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("输入与输出路径必须不同")
    frame=pd.read_csv(args.input)
    if args.round=="round2":
        from fraud.neural_finalize import predict_submission
        probability=predict_submission(frame)
        if not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():
            raise ValueError("预测概率无效")
        if not frame["id"].is_unique or frame["id"].isna().any():
            raise ValueError("输入 id 必须无缺失且唯一")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({"id":frame["id"],"is_fraud":probability}).to_csv(args.output,index=False)
        print(f"已生成第二轮 {len(frame):,} 条预测：{args.output.resolve()}")
        return
    folder=ROOT / "artifacts/final"
    metadata=json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    required=[metadata["id_column"]]+metadata["feature_columns"]
    missing=set(required)-set(frame.columns)
    if missing:
        raise ValueError(f"缺少输入列: {sorted(missing)}")
    if frame[metadata["id_column"]].isna().any() or not frame[metadata["id_column"]].is_unique:
        raise ValueError("输入 id 必须无缺失且唯一")
    probability=np.zeros(len(frame))
    for name,weight in metadata["weights"].items():
        bundle=joblib.load(folder / f"{name}.joblib")
        probability+=weight*predict(bundle,frame[metadata["feature_columns"]])
    if not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():
        raise ValueError("预测概率无效")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({metadata["id_column"]:frame[metadata["id_column"]],metadata["target_column"]:probability}).to_csv(args.output,index=False)
    print(f"已生成 {len(frame):,} 条预测：{args.output.resolve()}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
