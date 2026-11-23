"""Fold-local masked field reconstruction followed by binary classification."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from compare_tree_member import bootstrap_delta
from train_context_network import AMOUNTS


def tokenize(table, fit):
    columns, parents, sizes = [], [], []
    for field, (name, values) in enumerate(table.items()):
        if name in AMOUNTS:
            edges = np.unique(np.quantile(values.iloc[fit], np.arange(1, 64) / 64))
            cents = np.rint(values.to_numpy() * 100).astype("int64")
            features = [np.searchsorted(edges, values), cents % 100, cents // 100 % 100]
        else:
            vocab = pd.Index(values.iloc[fit].unique())
            features = [vocab.get_indexer(values) + 1]
        for codes in features:
            columns.append(codes)
            parents.append(field)
            sizes.append(100 if name in AMOUNTS else len(vocab) + 1)
    return (torch.tensor(np.column_stack(columns), device="cuda", dtype=torch.int32),
            torch.tensor(parents, device="cuda"), sizes)


class Network(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.register_buffer("offset", torch.tensor(np.cumsum([0] + sizes[:-1])))
        self.embedding = nn.Embedding(sum(sizes), 8)
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(8 * len(sizes), 128),
                                     nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, 96), nn.SiLU())
        self.decoder = nn.Linear(96, sum(sizes))
        self.classifier = nn.Linear(96, 1)

    def encode(self, x, hidden=None):
        tokens = self.embedding(x + self.offset)
        if hidden is not None:
            tokens = tokens.masked_fill(hidden.unsqueeze(-1), 0)
        return self.encoder(tokens)


def run(tokens, parents, sizes, fit, valid, y, pretrain):
    torch.manual_seed(42)
    model = Network(sizes).cuda()
    x, v = tokens[fit], tokens[valid]
    if pretrain:
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        gen = torch.Generator(device="cuda").manual_seed(320)
        n_fields = int(parents.max()) + 1
        for _ in range(4):
            for rows in torch.randperm(len(x), generator=gen, device="cuda").split(1024):
                batch = x[rows]
                # Predict one original field. Hide all digits derived from that field.
                field = int(torch.randint(n_fields, (), generator=gen, device="cuda"))
                hidden = parents == field
                logits = model.decoder(model.encode(batch, hidden)).split(sizes, dim=1)
                indices = torch.where(hidden)[0].tolist()
                loss = sum(F.cross_entropy(logits[j], batch[:, j].long()) for j in indices) / len(indices)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
    target = torch.tensor(y[fit], dtype=torch.float32, device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    gen = torch.Generator(device="cuda").manual_seed(2026)
    for epoch in range(10):
        opt.param_groups[0]["lr"] = .0001 + .0009 * (1 + np.cos(np.pi * epoch / 9)) / 2
        for rows in torch.randperm(len(x), generator=gen, device="cuda").split(1024):
            loss = F.binary_cross_entropy_with_logits(model.classifier(model.encode(x[rows])).flatten(), target[rows])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        return torch.cat([model.classifier(model.encode(b)).flatten().sigmoid() for b in v.split(4096)]).cpu().numpy()


def main():
    directory = Path("runs/masked_fields")
    directory.mkdir(exist_ok=True)
    data = pd.read_csv(".kaggle-cache/competitions/playground-series-s6e11/train.csv")
    folds = pd.read_csv("splits/seed42.csv").set_index("id").loc[data.id, "fold"].to_numpy()
    y = data.is_fraud.to_numpy()
    table = data.drop(columns=["id", "is_fraud"])
    predictions = {name: np.zeros(len(y)) for name in ["scratch", "masked", "additive"]}
    for fold in range(5):
        fit = np.flatnonzero((folds >= 0) & (folds != fold))
        valid = np.flatnonzero(folds == fold)
        tokens, parents, sizes = tokenize(table, fit)
        result = {}
        for name in ["scratch", "masked"]:
            result[name] = run(tokens, parents, sizes, fit, valid, y, name == "masked")
            print(fold, name, roc_auc_score(y[valid], result[name]), flush=True)
        with np.load(f"runs/context_network/fold{fold}.npz") as prev:
            assert np.array_equal(prev["id"], data.id.iloc[valid])
            result["additive"] = prev["additive"]
        for name, p in result.items():
            predictions[name][valid] = p
        np.savez_compressed(directory / f"fold{fold}.npz", id=data.id.iloc[valid], **result)
    dev = folds >= 0
    for name, p in predictions.items():
        print("OOF", name, roc_auc_score(y[dev], p[dev]), flush=True)
    for name in ["scratch", "additive"]:
        print("masked minus", name, bootstrap_delta(predictions[name][dev], predictions["masked"][dev], y[dev].astype(bool)), flush=True)


if __name__ == "__main__":
    main()
