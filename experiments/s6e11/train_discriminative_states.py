"""A conditional latent-state classifier with shared spline main effects."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from compare_tree_member import bootstrap_delta
from train_context_network import blocks, encoded_inputs


class ConditionalStates(nn.Module):
    def __init__(self, width, prior):
        super().__init__()
        self.register_buffer("knots", torch.tensor([-2., -1., 0., 1., 2.]))
        self.main = nn.Linear(width * 6, 1)
        self.states = nn.Linear(width * 6, 8, bias=False)
        nn.init.zeros_(self.main.weight)
        nn.init.constant_(self.main.bias, float(np.log(prior / (1 - prior))))
        nn.init.normal_(self.states.weight, std=.008)

    def forward(self, x):
        field = x.unsqueeze(-1)
        basis = torch.cat((field, torch.relu(field - self.knots)), dim=-1).flatten(1)
        # Bound each latent-state adjustment to prevent one state from dominating.
        scores = 2 * torch.tanh(self.states(basis).reshape(-1, 2, 4) / 2)
        partitions = torch.logsumexp(scores, dim=2)
        return self.main(basis).flatten() + partitions[:, 1] - partitions[:, 0]


def main():
    data = pd.read_csv(".kaggle-cache/competitions/playground-series-s6e11/train.csv")
    folds = pd.read_csv("splits/seed42.csv").set_index("id").loc[data.id, "fold"].to_numpy()
    y = data.is_fraud.to_numpy(int)
    table = data.drop(columns=["id", "is_fraud"])
    money, context = blocks(table)
    pred, base = np.zeros(len(y)), np.zeros(len(y))
    dest = Path("runs/discriminative_states")
    dest.mkdir(exist_ok=True)
    for f in range(5):
        fit = np.flatnonzero((folds >= 0) & (folds != f))
        valid = np.flatnonzero(folds == f)
        x, v, _ = encoded_inputs(table, y, fit, valid, money, context)
        torch.manual_seed(42)
        model = ConditionalStates(x.size(1), y[fit].mean()).cuda()
        target = torch.tensor(y[fit], dtype=torch.float32, device="cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=.0012, weight_decay=.02)
        gen = torch.Generator(device="cuda").manual_seed(2026)
        for epoch in range(12):
            opt.param_groups[0]["lr"] = .00015 + .00105 * (1 + np.cos(np.pi * epoch / 11)) / 2
            for batch in torch.randperm(len(x), device="cuda", generator=gen).split(2048):
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(model(x[batch]), target[batch])
                loss.backward()
                opt.step()
        with torch.no_grad():
            pred[valid] = torch.cat([model(b).sigmoid() for b in v.split(4096)]).cpu().numpy()
        with np.load(f"runs/context_network/fold{f}.npz") as ref:
            assert np.array_equal(ref["id"], data.id.iloc[valid])
            base[valid] = ref["additive"]
        print(f, roc_auc_score(y[valid], pred[valid]),roc_auc_score(y[valid], pred[valid])-roc_auc_score(y[valid],base[valid]), flush=True)
        np.savez_compressed(dest / f"fold{f}.npz", id=data.id.iloc[valid], prediction=pred[valid])
    dev = folds >= 0
    print("OOF", roc_auc_score(y[dev], pred[dev]), "paired interval",
          bootstrap_delta(base[dev], pred[dev], y[dev].astype(bool)), flush=True)


if __name__ == "__main__":
    main()
