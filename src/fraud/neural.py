"""表格神经网络、仅在训练数据拟合的预处理以及 GPU 训练循环。"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from torch import nn


class NeuralPreprocessor:
    """字符串类别预留未知值 0；布尔列直接使用 0/1，连续列分位数变换。"""

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, frame):
        self.columns = frame.columns.tolist()
        self.categorical = frame.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        self.boolean = frame.select_dtypes(include="bool").columns.tolist()
        self.continuous = [c for c in self.columns if c not in self.categorical + self.boolean]
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        clean = self.imputer.fit_transform(frame[self.continuous])
        self.quantile = QuantileTransformer(n_quantiles=min(1024, len(frame)), output_distribution="normal", subsample=min(100000, len(frame)), random_state=self.seed)
        self.quantile.fit(clean)
        self.vocabularies = {c: {v: i + 1 for i, v in enumerate(sorted(frame[c].fillna("__MISSING__").astype(str).unique()))} for c in self.categorical}
        self.cardinalities = [len(self.vocabularies[c]) + 1 for c in self.categorical]
        self.n_cont = len(self.continuous) + len(self.boolean)
        return self

    def transform(self, frame):
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError(f"缺少输入列: {sorted(missing)}")
        continuous = self.quantile.transform(self.imputer.transform(frame[self.continuous]))
        boolean = frame[self.boolean].fillna(False).astype("float32").to_numpy()
        numeric = np.column_stack([continuous, boolean]).astype("float32")
        categorical = (np.column_stack([frame[c].fillna("__MISSING__").astype(str).map(self.vocabularies[c]).fillna(0).to_numpy(dtype="int64") for c in self.categorical])
                       if self.categorical else np.empty((len(frame), 0), dtype="int64"))
        if not np.isfinite(numeric).all():
            raise ValueError("数值预处理产生无效值")
        return numeric, categorical


class SmallMLP(nn.Module):
    def __init__(self, n_cont, cardinalities, params):
        super().__init__()
        dimension = params["embedding_dim"]
        self.embeddings = nn.ModuleList([nn.Embedding(cardinality, dimension) for cardinality in cardinalities])
        width = n_cont + len(cardinalities) * dimension
        layers = []
        for next_width in params["hidden"]:
            layers += [nn.Linear(width, next_width), nn.LayerNorm(next_width), nn.ReLU(), nn.Dropout(params["dropout"])]
            width = next_width
        self.backbone = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, numeric, categorical):
        embedding = [layer(categorical[:, i]) for i, layer in enumerate(self.embeddings)]
        return self.backbone(torch.cat([numeric] + embedding, dim=-1))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("high")


def make_network(kind, prep, params, numeric_training, seed):
    seed_everything(seed)
    extra = {}
    if kind == "mlp":
        model = SmallMLP(prep.n_cont, prep.cardinalities, params)
    elif kind == "ft_transformer":
        from rtdl_revisiting_models import FTTransformer
        model = FTTransformer(n_cont_features=prep.n_cont, cat_cardinalities=prep.cardinalities, d_out=1,
                              n_blocks=params["n_blocks"], d_block=params["d_block"], attention_n_heads=params["attention_n_heads"],
                              attention_dropout=params["attention_dropout"], ffn_d_hidden=None, ffn_d_hidden_multiplier=4 / 3,
                              ffn_dropout=params["ffn_dropout"], residual_dropout=0.0)
    elif kind == "tabm":
        import rtdl_num_embeddings
        from tabm import TabM
        # 分箱只由本次实际训练的数据确定，不使用外层验证或比赛测试样本。
        bins = rtdl_num_embeddings.compute_bins(torch.from_numpy(numeric_training), n_bins=params["n_bins"])
        extra["bins"] = bins
        embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins, d_embedding=params["num_embedding_dim"], activation=False, version="B")
        model = TabM.make(n_num_features=prep.n_cont, cat_cardinalities=prep.cardinalities, d_out=1,
                          n_blocks=params["n_blocks"], d_block=params["d_block"], dropout=params["dropout"], k=params["k"], num_embeddings=embeddings)
    else:
        raise ValueError(f"未知网络: {kind}")
    return model, extra


def tensor_data(encoded, labels=None):
    numeric, categorical = encoded
    items = [torch.as_tensor(numeric, device="cuda"), torch.as_tensor(categorical, device="cuda")]
    if labels is not None:
        items.append(torch.as_tensor(np.asarray(labels, dtype="float32"), device="cuda"))
    return items


def batch_probabilities(model, tensors, batch_size):
    model.eval()
    chunks = []
    # 分类排序使用 FP32 推理，避免 BF16 输出量化引入过多并列分数。
    with torch.inference_mode():
        for start in range(0, len(tensors[0]), batch_size):
            logits = model(tensors[0][start:start+batch_size], tensors[1][start:start+batch_size]).float()
            probability = logits.sigmoid()
            # TabM 的各成员先分别转换为概率，再取算术平均。
            if probability.ndim == 3:
                probability = probability.mean(dim=1)
            chunks.append(probability.reshape(-1))
    # 整批预测结束后只传输一次，避免每个小批次等待 CPU/GPU 同步。
    return torch.cat(chunks).cpu().numpy()


def fit_network(kind, params, x, y, seed, epochs, validation=None, patience=None, min_epochs=1, log_path=None):
    if not torch.cuda.is_available():
        raise RuntimeError("神经网络实验强制使用 CUDA，不自动回退 CPU")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    prep = NeuralPreprocessor(seed).fit(x)
    encoded = prep.transform(x)
    model, extra = make_network(kind, prep, params, encoded[0], seed)
    model = model.to("cuda")
    train_tensors = tensor_data(encoded, y)
    valid_tensors = tensor_data(prep.transform(validation[0])) if validation is not None else None
    groups = model.make_parameter_groups() if kind == "ft_transformer" else model.parameters()
    optimizer = torch.optim.AdamW(groups, lr=params["lr"], weight_decay=params["weight_decay"])
    history, best_auc, best_epoch, best_state = [], -np.inf, 0, None
    batch_size = params["batch_size"]
    n = len(y)
    setup_seconds = time.perf_counter() - start
    for epoch in range(1, epochs + 1):
        model.train()
        torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        order = torch.randperm(n, device="cuda")
        accumulated = torch.zeros((), device="cuda")
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(train_tensors[0][indices], train_tensors[1][indices]).float()
                target = train_tensors[2][indices]
                if logits.ndim == 3:
                    target = target[:, None, None].expand_as(logits)
                else:
                    target = target[:, None]
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            accumulated += loss.detach() * len(indices)
        torch.cuda.synchronize()
        record = {"epoch": epoch, "train_loss": float(accumulated.item()/n), "train_seconds": time.perf_counter()-epoch_start}
        if not np.isfinite(record["train_loss"]):
            raise RuntimeError(f"{kind} loss 出现 NaN/Inf，epoch={epoch}")
        if validation is not None:
            probabilities = batch_probabilities(model, valid_tensors, batch_size*2)
            auc = float(roc_auc_score(validation[1], probabilities))
            record.update(valid_auc=auc, valid_ap=float(average_precision_score(validation[1], probabilities)))
            if auc > best_auc + 1e-6:
                best_auc, best_epoch = auc, epoch
                best_state = {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}
        record["epoch_total_seconds"] = time.perf_counter() - epoch_start
        history.append(record)
        if log_path is not None:
            from .common import write_json
            write_json(Path(log_path), history)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{kind} epoch={epoch} loss={record['train_loss']:.5f} seconds={record['train_seconds']:.2f}" + (f" inner_auc={record['valid_auc']:.6f}" if validation is not None else ""), flush=True)
        if validation is not None and patience is not None and epoch >= min_epochs and epoch-best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    result = {"kind": kind, "parameters": copy.deepcopy(params), "preprocessor": prep, "extra": extra,
              "state_dict": {key: value.detach().cpu().clone() for key,value in model.state_dict().items()},
              "selected_epochs": best_epoch, "history": history, "training_device": torch.cuda.get_device_name(),
              "peak_memory_mib": torch.cuda.max_memory_allocated()/1024**2, "training_seconds": time.perf_counter()-start,
              "setup_seconds": setup_seconds,
              "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "seed": seed}
    del train_tensors, valid_tensors, optimizer
    return model, result


def predict_frame(model, prep, frame, batch_size):
    tensors = tensor_data(prep.transform(frame))
    probabilities = batch_probabilities(model, tensors, batch_size)
    del tensors
    return probabilities


def restore_network(bundle):
    prep, params, kind = bundle["preprocessor"], bundle["parameters"], bundle["kind"]
    if kind == "tabm":
        import rtdl_num_embeddings
        from tabm import TabM
        embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(bundle["extra"]["bins"], d_embedding=params["num_embedding_dim"], activation=False, version="B")
        model = TabM.make(n_num_features=prep.n_cont, cat_cardinalities=prep.cardinalities, d_out=1, n_blocks=params["n_blocks"], d_block=params["d_block"], dropout=params["dropout"], k=params["k"], num_embeddings=embeddings)
    else:
        dummy = np.zeros((2, prep.n_cont), dtype="float32")
        model, _ = make_network(kind, prep, params, dummy, bundle["seed"])
    model.load_state_dict(bundle["state_dict"])
    return model.to("cuda").eval()
