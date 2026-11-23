"""Train a matched case-control ranker and audit fixed baseline errors."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold

from compare_tree_member import bootstrap_delta, placements, rank


ANCHORS = ["amount_usd", "merchant_risk_score", "velocity_score"]
OUT = Path("runs/hard_case_pairs")


def features(table):
    x = table.drop(columns=["id", "is_fraud"] + ANCHORS).copy()
    for name in x.select_dtypes("object"):
        x[name] = x[name].astype("category")
    for name in ["hours_since_last_txn", "distance_from_home_km", "account_balance_usd"]:
        cents = np.rint(100 * x[name]).astype(int)
        x[name + "_fraction"] = (cents % 100).astype("category")
    return x


def coordinates(table, fit):
    # Each coordinate is in percentile units, so no dollar-vs-score unit mismatch.
    return np.column_stack([np.searchsorted(np.sort(table[c].to_numpy()[fit]),table[c]) / len(fit)
                            for c in ANCHORS])


def main():
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(".kaggle-cache/competitions/playground-series-s6e11/train.csv")
    y = d.is_fraud.to_numpy(int)
    assert d.id.is_unique
    with np.load("runs/lm_error_audit/aligned.npz") as z:
        ids = pd.Index(d.id).get_indexer(z["id"])
        assert (ids >= 0).all()
        base, lm = np.full(len(y), np.nan), np.full(len(y), np.nan)
        base[ids], lm[ids] = z["additive"], z["lm"]
    pos = ids[y[ids] == 1]
    n = len(pos) // 10
    hard = np.intersect1d(pos[np.argsort(base[pos],kind="stable")[:n]],pos[np.argsort(lm[pos],kind="stable")[:n]])
    assert len(hard) == 790
    rows, allmatched = [], []
    x = features(d)
    aux = np.full(len(y), np.nan)
    for fold,(fit,val) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(d.id,y)):
        val = val[np.isfinite(base[val])]
        c = coordinates(d, fit)
        fraud, clean = fit[y[fit]==1], fit[y[fit]==0]
        rng = np.random.default_rng(2026)
        tree = cKDTree(c[clean])
        _, neighbors = tree.query(c[fraud],k=40)
        # Draw eight nearby normal controls per training case. Never use either public score in fitting.
        chosen = np.argsort(rng.random(neighbors.shape),axis=1)[:,:8]
        groups = np.column_stack([fraud,clean[np.take_along_axis(neighbors,chosen,axis=1)]])
        ix = groups.ravel()
        model = lgb.LGBMRanker(objective="lambdarank",n_estimators=140,learning_rate=.04,
                               num_leaves=12,min_child_samples=80,verbosity=-1,n_jobs=4,random_state=42)
        model.fit(x.iloc[ix],y[ix],group=np.full(len(groups),9))
        aux[val] = model.predict(x.iloc[val])
        clean = val[y[val]==0]
        tree = cKDTree(c[clean])
        for group, cases in [("hard",np.intersect1d(hard,val)),("all",val[y[val]==1])]:
            dist, near = tree.query(c[cases],k=256)
            for i, case in enumerate(cases):
                candidates = clean[near[i]]
                ok = dist[i] <= .05
                if group == "hard":
                    ok &= (base[candidates]>base[case]) & (lm[candidates]>lm[case])
                selected = np.flatnonzero(ok)[:3]
                target = rows if group=="hard" else allmatched
                for j in selected:
                    target.append((fold,case,candidates[j],dist[i,j]))
        print("fold",fold,"completed",flush=True)
    results = pd.DataFrame({"id":d.id.iloc[ids].to_numpy(),"fold_local_ranker":aux[ids]})
    results.to_csv(OUT/"ranker_oof.csv",index=False)
    candidate = np.full(len(y),np.nan)
    candidate[ids] = .95 * rank(base[ids]) + .05 * rank(aux[ids])
    lab = y[ids].astype(bool)
    bp = placements(base[ids],lab)[0]
    cp = placements(candidate[ids],lab)[0]
    print("AUC",bp.mean(),cp.mean(),"delta CI",bootstrap_delta(base[ids],candidate[ids],lab),flush=True)
    onlyhard = np.isin(pos,hard)
    print("placement deltas hard/other", (cp-bp)[onlyhard].mean(),(cp-bp)[~onlyhard].mean())
    matched_cases = {r[1] for r in rows}
    blind = [r for r in allmatched if r[1] in matched_cases]
    for name, data in [("hard",rows),("all",allmatched),("hard_score_blind",blind)]:
        pairs = pd.DataFrame(data,columns=["fold","case","control","distance"])
        a,b = pairs.case.to_numpy(),pairs.control.to_numpy()
        for tag, score in [("aux",aux),("base",base),("new",candidate)]:
            pairs[tag] = (score[a]>score[b]).astype(float)+.5*(score[a]==score[b])
        pairs["case_id"] = d.id.to_numpy()[a]
        pairs["control_id"] = d.id.to_numpy()[b]
        pairs.to_csv(OUT/f"{name}_pairs.csv",index=False)
        print(name,"cases/pairs/dist",pairs.case.nunique(),len(a),pairs.distance.median(),
              "wins",pairs.groupby("case")[["aux","base","new"]].mean().mean().to_dict())
        if name!="all":
            print("per fold",pairs.groupby(["fold","case"]).aux.mean().groupby(level=0).mean().to_dict())
            for col in ["cvv_retry_count","prior_disputes","ip_country_mismatch","is_foreign_transaction",
                        "used_vpn","billing_shipping_mismatch","is_new_merchant","is_ai_generated_scam_attempt"]:
                present = d[col].to_numpy()>0
                print(col, "case/control",present[pairs.case.unique()].mean(),
                      pd.Series(present[b]).groupby(a).mean().mean())
            print("median raw absolute differences",d[ANCHORS].iloc[a].reset_index(drop=True).sub(d[ANCHORS].iloc[b].reset_index(drop=True)).abs().median().to_dict())
    pd.DataFrame({"id":d.id.iloc[hard],"got_match":np.isin(hard,[r[1] for r in rows])}).to_csv(OUT/"hard_cases.csv",index=False)


if __name__ == "__main__":
    main()
