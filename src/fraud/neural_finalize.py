"""第二轮的开发集融合、全量重训、预测与证据核验。"""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .common import ID, ROOT, TARGET, timestamp, write_json
from .evaluation import f1_threshold, metrics
from .neural import fit_network, predict_frame, restore_network
from .neural_experiments import ARTIFACTS, REPORTS, data, load_bundle, neural_config, save_bundle


def aligned_oof():
    dev, folds, _, _ = data()
    base = pd.read_parquet(ROOT / "artifacts/selected_oof.parquet")
    if not np.array_equal(base[ID], dev[ID]) or not np.array_equal(base[TARGET], dev[TARGET]):
        raise ValueError("首轮融合 OOF 与第二轮数据不一致")
    values = {"tree_round1": base.probability.to_numpy()}
    for kind in neural_config()["models"]:
        frame = pd.read_parquet(ARTIFACTS / kind / "oof.parquet")
        if not np.array_equal(frame[ID], dev[ID]) or not np.array_equal(frame.fold, folds) or not np.array_equal(frame[TARGET], dev[TARGET]):
            raise ValueError(f"{kind} OOF 样本、标签或折顺序错误")
        values[kind] = frame.probability.to_numpy()
    return dev, folds, values


def blend():
    path = REPORTS / "selection.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    cfg = neural_config()
    dev, folds, values = aligned_oof()
    y = dev[TARGET].to_numpy()
    names = list(values)
    base_score = float(roc_auc_score(y, values["tree_round1"]))
    candidates = []
    best = {"weights": {"tree_round1": 1.0}, "auc": base_score}
    best_probability = values["tree_round1"].copy()
    # 四路非负权重按 0.1 的固定网格比较，搜索和阈值都仅使用开发集 OOF。
    for first, second, third in itertools.product(range(11), repeat=3):
        fourth = 10-first-second-third
        if fourth < 0:
            continue
        weights = {name: integer/10 for name,integer in zip(names, [first,second,third,fourth]) if integer}
        probability = sum(values[name]*weight for name,weight in weights.items())
        auc = float(roc_auc_score(y, probability))
        candidates.append({"weights": weights, "auc": auc})
        if auc > best["auc"]:
            best = {"weights": weights, "auc": auc}
            best_probability = probability
    changes = [float(roc_auc_score(y[folds==fold], best_probability[folds==fold]) - roc_auc_score(y[folds==fold], values["tree_round1"][folds==fold])) for fold in range(3)]
    if best["auc"] - base_score < 0.00005 or sum(change > 0 for change in changes) < 2:
        best = {"weights": {"tree_round1": 1.0}, "auc": base_score}
        best_probability = values["tree_round1"].copy()
        changes = [0.0]*3
    threshold = f1_threshold(y, best_probability)
    result = {"frozen_at": timestamp(), "weights": best["weights"], "oof_metrics": metrics(y, best_probability, threshold),
              "baseline_auc": base_score, "auc_improvement": best["auc"]-base_score, "fold_auc_changes": changes,
              "candidate_count": len(candidates), "candidates": candidates, "f1_threshold": threshold,
              "criterion": "开发集 OOF AUC 比首轮融合至少提升 0.00005，且至少两折提升；否则保留首轮方案。",
              "evaluation_scope": "开发集选型分数，存在选择偏差；本轮未使用旧独立测试集重复选参或声称新的独立测试成绩。"}
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_json(path, result)
    pd.DataFrame({ID: dev[ID], TARGET: y, "fold": folds, **values, "selected": best_probability}).to_parquet(ARTIFACTS / "comparison_oof.parquet", index=False)
    rows = []
    for name, probability in {**values, "round2_selected": best_probability}.items():
        threshold = f1_threshold(y, probability)
        entry = {"name": name, **{key:value for key,value in metrics(y, probability, threshold).items() if key != "confusion_matrix"}}
        entry["kind"] = name if name in cfg["models"] else "ensemble"
        entry["feature_engineering"] = False
        if name in cfg["models"]:
            model_result = json.loads((REPORTS / f"{name}.json").read_text(encoding="utf-8"))
            entry["training_seconds"] = model_result["total_seconds"]
            entry["parameters"] = model_result["folds"][0]["parameter_count"]
            entry["peak_memory_mib"] = max(row["peak_memory_mib"] for row in model_result["folds"])
            entry["mean_fold_auc"] = model_result["mean_fold_auc"]
            entry["std_fold_auc"] = model_result["std_fold_auc"]
        rows.append(entry)
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False, encoding="utf-8-sig")
    original = pd.read_csv(ROOT / "reports/model_comparison.csv").rename(columns={"seconds":"training_seconds"})
    original["round"] = "round1"
    comparison["round"] = np.where(comparison.name=="tree_round1", "round1", "round2")
    pd.concat([original,comparison],ignore_index=True).sort_values("roc_auc",ascending=False).to_csv(REPORTS/"all_models_comparison.csv",index=False,encoding="utf-8-sig")
    print(json.dumps({key:value for key,value in result.items() if key != "candidates"}, ensure_ascii=False, indent=2), flush=True)
    return result


def finalize():
    selection = blend()
    complete = REPORTS / "finalization.json"
    if complete.exists():
        print("第二轮全量模型和提交文件已存在。", flush=True)
        return
    cfg = neural_config()
    _, _, train, test = data()
    features = train.columns.drop([ID,TARGET]).tolist()
    folder = ARTIFACTS / "final"
    folder.mkdir(parents=True, exist_ok=True)
    test_probabilities = {}
    rounds, training = {}, {}
    for kind, params in cfg["models"].items():
        result = json.loads((REPORTS / f"{kind}.json").read_text(encoding="utf-8"))
        epochs = max(1, int(np.median([row["selected_epochs"] for row in result["folds"]])))
        rounds[kind] = epochs
        model_path = folder / f"{kind}.pt"
        if model_path.exists():
            bundle = load_bundle(model_path)
            model = restore_network(bundle)
        else:
            model, bundle = fit_network(kind, params, train[features], train[TARGET], cfg["seed"], epochs,
                                        log_path=REPORTS / f"{kind}_full_history.json")
            save_bundle(model_path, bundle)
        test_probabilities[kind] = predict_frame(model, bundle["preprocessor"], test[features], params["batch_size"]*2)
        training[kind] = {key:bundle[key] for key in ["training_seconds","peak_memory_mib","parameter_count","training_device"]}
        pd.DataFrame({ID:test[ID],TARGET:test_probabilities[kind]}).to_csv(ROOT / "submissions" / f"round2_{kind}.csv",index=False)
        del model,bundle
        gc.collect();torch.cuda.empty_cache()
    base = pd.read_csv(ROOT / "submissions/round1_best.csv")
    if not np.array_equal(base[ID],test[ID]):
        raise ValueError("首轮提交的测试样本顺序不同")
    test_probabilities["tree_round1"] = base[TARGET].to_numpy()
    probability = sum(test_probabilities[name]*weight for name,weight in selection["weights"].items())
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    submission = pd.DataFrame({ID:test[ID],TARGET:probability})
    if not np.array_equal(submission[ID],sample[ID]) or submission.columns.tolist()!=sample.columns.tolist() or not np.isfinite(probability).all() or not submission[TARGET].between(0,1).all():
        raise ValueError("第二轮提交格式或概率无效")
    submission.to_csv(ROOT / "submissions/round2_best.csv",index=False)
    pd.DataFrame({ID:test[ID],**test_probabilities}).to_parquet(folder / "test_predictions.parquet",index=False)
    write_json(folder / "manifest.json",{"weights":selection["weights"],"feature_columns":features,"id_column":ID,"target_column":TARGET,"base_tree_manifest":"artifacts/final/manifest.json"})
    write_json(complete,{"completed_at":timestamp(),"weights":selection["weights"],"epochs":rounds,"training":training,"submission":"submissions/round2_best.csv","rows":len(submission),"kaggle_status":"仅生成文件，未上传 Kaggle。","evaluation_scope":selection["evaluation_scope"]})


def predict_submission(frame):
    metadata = json.loads((ARTIFACTS / "final/manifest.json").read_text(encoding="utf-8"))
    missing=set([ID]+metadata["feature_columns"])-set(frame.columns)
    if missing:
        raise ValueError(f"缺少输入列: {sorted(missing)}")
    if frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError("输入 id 必须无缺失且唯一")
    output = np.zeros(len(frame))
    for name,weight in metadata["weights"].items():
        if name=="tree_round1":
            import joblib
            from .models import predict
            base = json.loads((ROOT / "artifacts/final/manifest.json").read_text(encoding="utf-8"))
            base_probability = np.zeros(len(frame))
            for member,base_weight in base["weights"].items():
                base_probability += base_weight*predict(joblib.load(ROOT / "artifacts/final" / f"{member}.joblib"),frame[base["feature_columns"]])
            output += weight*base_probability
        else:
            bundle = load_bundle(ARTIFACTS / "final" / f"{name}.pt")
            model = restore_network(bundle)
            output += weight*predict_frame(model,bundle["preprocessor"],frame[metadata["feature_columns"]],bundle["parameters"]["batch_size"]*2)
            del model,bundle
            gc.collect();torch.cuda.empty_cache()
    return output


def verify():
    dev,folds,train,test = data()
    features=train.columns.drop([ID,TARGET]).tolist()
    checks=[]
    protocol=json.loads((REPORTS/"protocol.json").read_text(encoding="utf-8"))
    for name,record in protocol.items():
        if isinstance(record,dict) and "sha256" in record:
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==record["sha256"],f"首轮冻结文件发生变化: {name}"
    checks.append("首轮数据划分、OOF、留出集成绩和提交文件校验值保持一致")
    selection=json.loads((REPORTS/"selection.json").read_text(encoding="utf-8"))
    comparison=pd.read_parquet(ARTIFACTS/"comparison_oof.parquet")
    assert np.array_equal(comparison[ID],dev[ID]) and np.array_equal(comparison.fold,folds)
    assert np.array_equal(comparison[TARGET],dev[TARGET])
    assert np.isclose(sum(selection["weights"].values()),1) and all(weight>=0 for weight in selection["weights"].values())
    combined=sum(comparison[name].to_numpy()*weight for name,weight in selection["weights"].items())
    assert np.allclose(combined,comparison.selected,atol=1e-12)
    replay_metrics=metrics(dev[TARGET],combined,selection["f1_threshold"])
    for key in ["roc_auc","average_precision","precision","recall","f1"]:
        assert np.isclose(replay_metrics[key],selection["oof_metrics"][key],atol=1e-12)
    manifest=json.loads((ARTIFACTS/"final/manifest.json").read_text(encoding="utf-8"))
    assert manifest["weights"]==selection["weights"]
    checks.append("融合权重、开发集 OOF 对齐、选型指标和最终模型清单一致")
    max_errors={}
    for kind in neural_config()["models"]:
        result=json.loads((REPORTS / f"{kind}.json").read_text(encoding="utf-8"))
        assert result["signature"]["configuration"]==neural_config(),f"{kind} 当前配置与训练记录不同"
        assert result["signature"]["source_sha256"]==hashlib.sha256((ROOT/"src/fraud/neural.py").read_bytes()).hexdigest(),f"{kind} 训练实现已改变"
        oof=pd.read_parquet(ARTIFACTS / kind / "oof.parquet")
        assert np.array_equal(oof[ID],dev[ID]) and np.array_equal(oof.fold,folds)
        assert np.array_equal(oof[TARGET],dev[TARGET]) and np.isfinite(oof.probability).all()
        assert np.isclose(metrics(oof[TARGET],oof.probability)["roc_auc"],result["oof_default"]["roc_auc"],atol=1e-12)
        bundle=load_bundle(ARTIFACTS / kind / "fold_0.pt")
        model=restore_network(bundle)
        valid=np.flatnonzero(folds==0)
        replay=predict_frame(model,bundle["preprocessor"],dev.iloc[valid][features],bundle["parameters"]["batch_size"]*2)
        delta=float(np.max(np.abs(replay-oof.iloc[valid].probability.to_numpy())))
        assert np.allclose(replay,oof.iloc[valid].probability,atol=2e-4,rtol=1e-3)
        max_errors[kind]=delta
        assert all("RTX" in row["training_device"] for row in result["folds"])
        checks.append(f"{kind}: GPU 训练、OOF覆盖、指标复算、首折完整重载推理通过")
        del model,bundle
        gc.collect();torch.cuda.empty_cache()
    submission=pd.read_csv(ROOT / "submissions/round2_best.csv")
    sample=pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    assert submission.shape==sample.shape and submission.columns.tolist()==sample.columns.tolist()
    assert np.array_equal(submission[ID],test[ID]) and submission[TARGET].between(0,1).all()
    replay=predict_submission(test)
    delta=float(np.max(np.abs(replay-submission[TARGET].to_numpy())))
    assert np.allclose(replay,submission[TARGET],atol=2e-4,rtol=1e-3)
    checks.append("全部比赛测试样本的提交格式、概率和重载融合推理通过")
    write_json(REPORTS / "verification.json",{"checked_at":timestamp(),"passed":True,"checks":checks,"fold_reload_max_errors":max_errors,"submission_reload_max_error":delta,"rows":len(submission)})
    print("\n".join(checks),flush=True)


def report():
    from .neural_reporting import run
    run()
