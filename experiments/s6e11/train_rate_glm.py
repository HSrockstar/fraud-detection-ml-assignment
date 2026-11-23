"""Self-trained fold-local category rate model. Uses only the official features and labels."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from compare_tree_member import rank, bootstrap_delta, load_comparison_data


def keys(df):
    # Codes identify equal values only. All counts and label rates are fitted per split.
    columns = []
    def add(s):
        columns.append(pd.factorize(s, use_na_sentinel=False)[0].astype(np.int32))
    for _, s in df.items():
        add(s)
        if pd.api.types.is_float_dtype(s) and (s % 1 != 0).any():
            cents = np.rint(100*s).astype(int)
            for v in (cents//100, cents%100, cents//1000, cents//10%10,
                      cents%10, np.floor(10*np.log1p(s)).astype(int)):
                add(v)
    return columns


def rates(columns, fit, score, y):
    result = np.empty((len(score),len(columns)),dtype=np.float32)
    prior = y[fit].mean()
    for j,col in enumerate(columns):
        n = int(col.max())+1
        count = np.bincount(col[fit], minlength=n)
        fraud = np.bincount(col[fit],weights=y[fit],minlength=n)
        lookup = np.log((fraud+30*prior)/(count-fraud+30*(1-prior)))
        result[:,j] = lookup[col[score]]
    return result


def main():
    t = pd.read_csv(".kaggle-cache/competitions/playground-series-s6e11/train.csv")
    fold = pd.read_csv("splits/seed42.csv").set_index("id").loc[t.id,"fold"].to_numpy()
    t = t.loc[fold>=0].reset_index(drop=True)
    fold = fold[fold>=0]
    y = t.is_fraud.to_numpy(dtype=int)
    cols = keys(t.drop(columns=["id","is_fraud"]))
    pred = np.empty(len(y))
    for f in range(5):
        a,b = np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
        x = np.empty((len(a),len(cols)),dtype=np.float32)
        for inner_fit,inner_score in StratifiedKFold(5,shuffle=True,random_state=42).split(a,y[a]):
            x[inner_score] = rates(cols,a[inner_fit],a[inner_score],y)
        model = LogisticRegression(C=.1,max_iter=600)
        model.fit(x,y[a])
        pred[b] = model.decision_function(rates(cols,a,b,y))
        print("fold",f,roc_auc_score(y[b],pred[b]),flush=True)
    np.savez_compressed("runs/self_rate_glm_oof.npz",id=t.id,validation=pred)
    seeds=[]
    for s in (42,43):
        series=[]
        for f in range(5):
            with np.load(f"runs/realmlp-e4-s{s}-f{f}.npz") as z:
                series.append(pd.Series(z["validation"],index=z["validation_id"]))
        seeds.append(rank(pd.concat(series,verify_integrity=True).loc[t.id]))
    base,tree,_,_,_,_,_=load_comparison_data()
    print("single",roc_auc_score(y,pred),flush=True)
    for label,b,w in [("self",rank(np.mean(seeds,axis=0)),.5),("public proxy",.9*base+.1*tree,.1)]:
        x=(1-w)*b+w*rank(pred)
        print(label,roc_auc_score(y,b),roc_auc_score(y,x),
              [round(roc_auc_score(y[fold==f],x[fold==f])-roc_auc_score(y[fold==f],b[fold==f]),7) for f in range(5)],
              bootstrap_delta(b,x,y.astype(bool)),flush=True)


if __name__ == "__main__":
    main()
