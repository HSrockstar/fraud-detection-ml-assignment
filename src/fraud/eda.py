from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .common import ID, ROOT, TARGET, load_data, plot_setup, splits, write_json


def run():
    train, test = load_data()
    split = splits(train)
    dev = train.loc[split.partition == "dev"].copy()
    columns = train.columns.drop([ID, TARGET]).tolist()
    numeric = dev[columns].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in columns if c not in numeric]
    summary = {
        "competition": "playground-series-s6e11", "source": "https://www.kaggle.com/competitions/playground-series-s6e11/data",
        "train_rows": len(train), "competition_test_rows": len(test), "feature_count": len(columns),
        "dev_rows": len(dev), "holdout_rows": int((split.partition == "holdout").sum()),
        "numeric_features": numeric, "categorical_features": categorical,
        "dev_class_counts": {str(k): int(v) for k, v in dev[TARGET].value_counts().sort_index().items()},
        "dev_fraud_fraction": float(dev[TARGET].mean()),
        "dev_duplicate_features": int(dev.duplicated(subset=columns).sum()),
        "all_normal_accuracy_on_dev": float(1 - dev[TARGET].mean()),
        "scope": "有监督分析仅使用开发集；独立测试集不用于特征选择和调参。",
    }
    table = pd.DataFrame({"dtype": dev[columns].dtypes.astype(str), "dev_missing": dev[columns].isna().sum(),
                          "competition_test_missing": test[columns].isna().sum(), "dev_unique": dev[columns].nunique()})
    table.to_csv(ROOT / "reports/feature_profile.csv", encoding="utf-8-sig")
    dev[numeric].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.to_csv(ROOT / "reports/numeric_summary.csv", encoding="utf-8-sig")
    drift = []
    for column in numeric:
        a, b = dev[column].dropna(), test[column].dropna()
        stat = ks_2samp(a, b)
        drift.append({"feature": column, "ks_statistic": float(stat.statistic), "p_value": float(stat.pvalue)})
    pd.DataFrame(drift).sort_values("ks_statistic", ascending=False).to_csv(ROOT / "reports/train_test_drift.csv", index=False)
    correlations = dev[numeric + [TARGET]].corr(method="spearman")
    ranked = correlations[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    summary["top_numeric_associations"] = ranked.head(8).to_dict()
    summary["total_dev_missing"] = int(dev[columns].isna().sum().sum())
    summary["max_numeric_ks"] = max(drift, key=lambda r: r["ks_statistic"])
    write_json(ROOT / "reports/eda_summary.json", summary)
    plt = plot_setup()
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6.5, 4))
    counts = dev[TARGET].value_counts().reindex([0, 1])
    ax.bar(["正常交易", "欺诈交易"], counts, color=["#4276AA", "#D45B55"])
    for i, value in enumerate(counts):
        ax.text(i, value + len(dev) * 0.012, f"{value:,}\n({value / len(dev):.2%})", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set(title="开发集类别分布", ylabel="样本数")
    fig.tight_layout(); fig.savefig(ROOT / "reports/figures/class_balance.png"); plt.close(fig)
    top = ranked.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, column in zip(axes.flat, top):
        low, high = dev[column].quantile([0.005, 0.995])
        bins = np.linspace(low, high, 35) if high > low else 10
        for label, color in [(0, "#4276AA"), (1, "#D45B55")]:
            values = dev.loc[dev[TARGET] == label, column].clip(low, high)
            ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5, color=color, label="欺诈" if label else "正常")
        ax.set_title(column, fontsize=10); ax.legend(fontsize=8)
    fig.suptitle("关联较强的数值特征分布（开发集，两端截尾仅用于绘图）")
    fig.tight_layout(); fig.savefig(ROOT / "reports/figures/feature_distributions.png"); plt.close(fig)
    heat_columns = ranked.head(10).index.tolist() + [TARGET]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(correlations.loc[heat_columns, heat_columns], cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("开发集 Spearman 相关性（关联不代表因果）")
    fig.tight_layout(); fig.savefig(ROOT / "reports/figures/correlation.png"); plt.close(fig)
    if categorical:
        rows = (len(categorical) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 3.2), squeeze=False)
        for ax, column in zip(axes.flat, categorical):
            rates = dev.groupby(column, dropna=False)[TARGET].mean().sort_values()
            rates.plot.barh(ax=ax, color="#4276AA")
            ax.set(title=column, xlabel="欺诈比例", ylabel="")
            ax.set_title(column, fontsize=10)
        for ax in list(axes.flat)[len(categorical):]:
            ax.set_visible(False)
        fig.tight_layout(); fig.savefig(ROOT / "reports/figures/categorical_fraud_rates.png"); plt.close(fig)
    print(summary, flush=True)
