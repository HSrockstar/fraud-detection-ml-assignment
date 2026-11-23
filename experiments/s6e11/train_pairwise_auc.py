"""Same additive baseline and development splits, with a pairwise ranking loss."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from compare_tree_member import bootstrap_delta, placements
from train_context_network import blocks, encoded_inputs, ContextNet


def main():
    frame = pd.read_csv(".kaggle-cache/competitions/playground-series-s6e11/train.csv")
    folds = pd.read_csv("splits/seed42.csv").set_index("id").loc[frame.id, "fold"].to_numpy()
    y = frame.is_fraud.to_numpy(int)
    table = frame.drop(columns=["id", "is_fraud"])
    money, context = blocks(table)
    baseline, candidate = np.zeros(len(y)), np.zeros(len(y))
    out = Path("runs/pairwise_auc")
    out.mkdir(exist_ok=True)
    for f in range(5):
        fit = np.flatnonzero((folds >= 0) & (folds != f))
        valid = np.flatnonzero(folds == f)
        x, v, boundary = encoded_inputs(table, y, fit, valid, money, context)
        torch.manual_seed(42)
        model = ContextNet(x.size(1), boundary, False, y[fit].mean()).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.0012, weight_decay=.02)
        labels = torch.tensor(y[fit], device="cuda", dtype=torch.float32)
        gen = torch.Generator(device="cuda").manual_seed(2026)
        for epoch in range(12):
            opt.param_groups[0]["lr"] = .00015 + .00105 * (1+np.cos(np.pi*epoch/11))/2
            for rows in torch.randperm(len(x), generator=gen, device="cuda").split(2048):
                z, target = model(x[rows]), labels[rows]
                loss = F.binary_cross_entropy_with_logits(z, target)
                pos, neg = z[target==1], z[target==0]
                if pos.numel() and neg.numel():
                    loss = loss + .15 * F.softplus(neg[None, :] - pos[:, None]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            candidate[valid] = torch.cat([model(b).sigmoid() for b in v.split(4096)]).cpu().numpy()
        with np.load(f"runs/context_network/fold{f}.npz") as ref:
            assert np.array_equal(ref["id"], frame.id.iloc[valid])
            baseline[valid] = ref["additive"]
        print(f, roc_auc_score(y[valid],candidate[valid])-roc_auc_score(y[valid],baseline[valid]), flush=True)
        np.savez_compressed(out/f"fold{f}.npz", id=frame.id.iloc[valid], prediction=candidate[valid])
    mask = folds >= 0
    label = y[mask].astype(bool)
    old = placements(baseline[mask], label)[0]
    new = placements(candidate[mask], label)[0]
    print("OOF",new.mean(),"delta",new.mean()-old.mean(),"paired CI",
          bootstrap_delta(baseline[mask],candidate[mask],label),flush=True)


if __name__ == "__main__":
    main()
