"""Standalone S6E11 classifier. Compare raw, decimal keys, and GPT-2 numeric token keys."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from compare_tree_member import bootstrap_delta
from train_rate_glm import rates
from train_joint_rate_glm import context_keys


def numeric_token_keys(text):
    vocab = Path("references/gpt2_124m_vocab.bpe").read_bytes()
    assert sha256(vocab).hexdigest() == "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"
    pairs = {tuple(line.split()): i for i,line in enumerate(vocab.decode().splitlines()[1:])}

    @lru_cache(maxsize=None)
    def tokenize(piece):
        tokens = list(piece)
        while len(tokens) > 1:
            pair = min(zip(tokens, tokens[1:]), key=lambda item: pairs.get(item, 1000000))
            if pair not in pairs:
                break
            merged = []
            i = 0
            while i < len(tokens):
                if i+1 < len(tokens) and (tokens[i],tokens[i+1]) == pair:
                    merged.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
        return tokens

    out = []
    for field in text:
        code, values = pd.factorize(text[field])
        assert code.min() >= 0
        records = []
        for s in values:
            integer, _, frac = s.partition(".")
            assert integer.isdecimal() and (not frac or frac.isdecimal())
            # GPT-2 separates digit runs and the decimal point before BPE.
            head = tokenize("Ġ"+integer)
            tail = tokenize(frac) if frac else ["<empty>"]
            records.append((head[0],head[-1],tail[0],tail[-1],(head[-1],tail[0]),(len(head),len(tail))))
        for token_values in zip(*records):
            groups = pd.factorize(pd.Series(token_values, dtype=object))[0]
            out.append(groups[code].astype(np.int32))
    return out


def fit_model(x, y, raw_categories):
    model = lgb.LGBMClassifier(n_estimators=850, learning_rate=.075, num_leaves=4,
                               min_child_samples=100, reg_lambda=2, max_bin=63,
                               colsample_bytree=.8, random_state=42, n_jobs=4,
                               force_col_wise=True, verbosity=-1)
    model.fit(x,y,categorical_feature=raw_categories)
    return model


def main():
    root = Path(".kaggle-cache/competitions/playground-series-s6e11")
    frames = [pd.read_csv(root/n) for n in ("train.csv","test.csv")]
    train,test = frames
    ids = train.id
    y = train.is_fraud.to_numpy(dtype=int)
    fold = pd.read_csv("splits/seed42.csv").set_index("id").loc[ids,"fold"].to_numpy()
    table = pd.concat([train[test.columns],test],ignore_index=True).drop(columns="id")
    digits = context_keys(table)
    floats = list(table.select_dtypes(include="float").columns)
    text = pd.concat([pd.read_csv(root/n,usecols=floats,dtype=str)
                      for n in ("train.csv","test.csv")],ignore_index=True)
    token_keys = numeric_token_keys(text)
    del text
    all_keys = digits + token_keys
    raw_cats = []
    for j,col in enumerate(table):
        if not pd.api.types.is_numeric_dtype(table[col]):
            raw_cats.append(j)
            table[col] = pd.factorize(table[col])[0]
    raw = table.to_numpy(dtype=np.float32)
    stops = {"raw":raw.shape[1], "digits":raw.shape[1]+len(digits),
             "bpe":raw.shape[1]+len(all_keys)}
    work = Path("runs/bpe_tree")
    work.mkdir(exist_ok=True)
    for f in range(6):
        final = f == 5
        fit = np.arange(len(y)) if final else np.flatnonzero((fold>=0)&(fold!=f))
        score = np.arange(len(y),len(table)) if final else np.flatnonzero(fold==f)
        names = list(stops)
        if final:
            scores = {}
            for name in stops:
                valid_predictions = np.full(len(y), np.nan)
                for j in range(5):
                    with np.load(work/f"fold{j}.npz") as saved:
                        assert np.array_equal(ids[fold==j], saved["id"])
                        valid_predictions[fold==j] = saved[name]
                scores[name] = roc_auc_score(y[fold>=0], valid_predictions[fold>=0])
            best = max(scores, key=scores.get)
            names = [best]
        path = work / (f"full_{best}.npz" if final else f"fold{f}.npz")
        if path.exists():
            continue
        x = np.empty((len(fit),stops["bpe"]),dtype=np.float32)
        x[:,:raw.shape[1]] = raw[fit]
        for a,b in StratifiedKFold(5,shuffle=True,random_state=42).split(fit,y[fit]):
            x[b,raw.shape[1]:] = rates(all_keys,fit[a],fit[b],y)
        v = np.column_stack((raw[score], rates(all_keys,fit,score,y)))
        preds = {}
        for name in names:
            end = stops[name]
            model = fit_model(x[:,:end],y[fit],raw_cats)
            pred = model.booster_.predict(v[:,:end],num_threads=4)
            assert np.isfinite(pred).all()
            preds[name] = pred
            print("full" if final else f, name, "done" if final else roc_auc_score(y[score],pred),flush=True)
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp,id=test.id if final else ids.iloc[score],**preds)
        temp.replace(path)
        del x,v,model
        gc.collect()
    oof = {}
    for name in stops:
        rows = []
        for f in range(5):
            with np.load(work/f"fold{f}.npz") as z:
                rows.append(pd.Series(z[name],index=z["id"]))
        oof[name] = pd.concat(rows,verify_integrity=True).loc[ids[fold>=0]].to_numpy()
    labels = y[fold>=0].astype(bool)
    print("pooled OOF", {name:roc_auc_score(labels,p) for name,p in oof.items()},flush=True)
    print("paired 95% interval BPE minus digits",bootstrap_delta(oof["digits"],oof["bpe"],labels),flush=True)
    template = pd.read_csv(root/"sample_submission.csv")
    with np.load(work/f"full_{best}.npz") as z:
        assert np.array_equal(template.id,z["id"]) and template.id.is_unique
        template.is_fraud = z[best]
    assert template.is_fraud.between(0,1).all()
    template.to_csv(work/f"single_{best}_tree.csv",index=False)


if __name__=="__main__":
    main()
