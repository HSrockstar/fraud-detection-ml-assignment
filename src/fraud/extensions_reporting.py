"""把选做实验的真实指标加入课程实验报告。"""

import json

from .common import ROOT, plot_setup


def append_blocks(blocks, result, first_section, figure_number):
    oof = result["oof"]

    def heading(text):
        blocks.append({"type": "heading", "level": 2, "text": text})

    def para(text):
        blocks.append({"type": "paragraph", "text": text})

    heading(f"{first_section} 从零实现逻辑回归")
    para("我们用 NumPy 实现 Sigmoid、负对数似然、梯度和 Hessian，求解牛顿方向后用回溯线搜索更新参数。目标函数为平均二元交叉熵加权重平方和除以 2Cn，其中 C=1，n 为训练样本数，截距不参与惩罚。手写版和 sklearn 的 L-BFGS 版共用折内预处理和开发集三折，使用相同的目标函数。")
    rows = []
    for name, label in [("scratch", "NumPy 牛顿法"), ("sklearn", "sklearn L-BFGS")]:
        elapsed = sum(fold[f"{name}_fit_seconds"] for fold in result["folds"])
        rows.append([
            label, f"{oof[name]['roc_auc']:.6f}", f"{oof[name]['average_precision']:.6f}",
            f"{oof[name]['log_loss']:.6f}", f"{elapsed:.2f}",
        ])
    blocks.append({"type": "table", "headers": ["算法", "OOF AUC", "OOF AP", "对数损失", "拟合 秒"],
                   "rows": rows, "keep_together": True})
    max_diff = max(fold["max_absolute_probability_difference"] for fold in result["folds"])
    iterations = [fold["scratch_iterations"] for fold in result["folds"]]
    scratch_steps = "、".join(map(str, iterations))
    skl_steps = "、".join(str(fold["sklearn_iterations"]) for fold in result["folds"])
    para(f"两种实现的折外概率平均绝对差为 {result['mean_absolute_probability_difference']:.2e}，最大差为 {max_diff:.2e}。手写版各折迭代 {scratch_steps} 次，库版本为 {skl_steps} 次。牛顿法每步构造并求解 Hessian，迭代次数和单步成本均与 L-BFGS 不同，表中耗时为三折拟合之和。")
    heading(f"{first_section + 1} 样本量与输入扰动的鲁棒性")
    para("对逻辑回归和 XGBoost 做单因素对照。样本量实验从每折训练数据分层构建 10%、25%、50%、100% 的嵌套子集，两种模型在相同子集重新拟合，在完整、干净的验证折评估。扰动实验固定全样本训练的模型，用相同随机数扰动验证折的六列连续特征。噪声标准差相当于当折训练 IQR 的 0.10、0.25、0.50 倍。异常值实验随机选取 1%、5%、10% 的验证样本，每行将一个随机连续特征替换为当折训练 Q3 加 10 倍 IQR。")
    size_10 = [oof[f"size_0.10_{model}"]["roc_auc"] for model in ("logistic", "xgb")]
    clean = [oof[f"size_1.00_{model}"] for model in ("logistic", "xgb")]
    noise = [oof[f"noise_0.50_{model}"] for model in ("logistic", "xgb")]
    outliers = [oof[f"outliers_0.10_{model}"] for model in ("logistic", "xgb")]
    auc_drop = [(base["roc_auc"] - noisy["roc_auc"], base["roc_auc"] - extreme["roc_auc"])
                for base, noisy, extreme in zip(clean, noise, outliers)]
    para(f"样本量从 10% 增至 100% 时，逻辑回归的 OOF AUC 从 {size_10[0]:.4f} 变为 {clean[0]['roc_auc']:.4f}，XGBoost 从 {size_10[1]:.4f} 变为 {clean[1]['roc_auc']:.4f}。相对干净输入，最强噪声下两者 AUC 分别下降 {auc_drop[0][0]:.4f} 和 {auc_drop[1][0]:.4f}。10% 异常值下，分别下降 {auc_drop[0][1]:.4f} 和 {auc_drop[1][1]:.4f}。")
    para(f"在 10% 异常值条件下，逻辑回归的 AP 从 {clean[0]['average_precision']:.3f} 降至 {outliers[0]['average_precision']:.3f}，XGBoost 从 {clean[1]['average_precision']:.3f} 降至 {outliers[1]['average_precision']:.3f}。极端输入可沿逻辑回归的线性得分继续放大，树模型在超出最外侧分裂阈值后仍走同一分支。这一结构差异与本实验中两者受异常值影响的大小一致。逐折指标见实验附件。")
    figure(result)
    blocks.append({
        "type": "figure", "path": "figures/robustness.png",
        "caption": f"图 {figure_number} 样本量、噪声和异常值对开发集 AUC 的影响",
    })


def figure(result):
    plt = plot_setup()
    oof = result["oof"]
    protocol = result["protocol"]
    settings = [
        ("size", protocol["sample_fractions"], "训练样本比例", True),
        ("noise", [0.0, *protocol["noise_sigmas_in_training_iqr"]], "噪声标准差 / IQR", False),
        ("outliers", [0.0, *protocol["outlier_row_fractions"]], "异常样本比例", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.25), layout="constrained")
    for ax, (prefix, levels, xlabel, percent) in zip(axes, settings):
        for model, label, color, marker in (
            ("logistic", "逻辑回归", "#356694", "o"),
            ("xgb", "XGBoost", "#BB563B", "s"),
        ):
            keys = [f"{prefix}_{level:.2f}" if level else "size_1.00" for level in levels]
            ax.plot(levels, [oof[f"{key}_{model}"]["roc_auc"] for key in keys],
                    marker=marker, color=color, linewidth=2, label=label)
        ticks = [0.0, 0.05, 0.10] if prefix == "outliers" else levels
        ax.set_xticks(ticks, [f"{level:.0%}" if percent else f"{level:.2f}" for level in ticks])
        ax.set_xlabel(xlabel, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.locator_params(axis="y", nbins=5)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("OOF ROC-AUC", fontsize=14)
    axes[0].legend(fontsize=11, loc="lower right")
    fig.savefig(ROOT / "reports/figures/robustness.png")
    plt.close(fig)


def append_latest(blocks):
    """引用已经保存的后续实验和实际比赛记录。"""
    def optional(name):
        path = ROOT / "reports" / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    blend = optional("blend_submission.json")
    capacity = optional("leaf_capacity.json")
    shallow = optional("shallow_blend_submission.json")
    llm = optional("llm_pilot_v1/metrics.json")
    if blend and capacity and shallow:
        public = blend["kaggle_submission"]["public_score"]
        shallow_public = shallow["kaggle_submission"]["public_score"]
        paragraphs = [
            f"后续将部分连续数值拆为整数档和小数档，XGBoost 使用交叉拟合目标编码，CatBoost 使用原生类别处理。按 7∶3 融合的开发集 OOF AUC 为 {blend['oof_auc']:.6f}，提交记录中的公开榜 AUC 为 {public:.5f}。",
            f"固定学习率、正则化和编码后，三叶 XGBoost 的 OOF AUC 为 {capacity['best']['oof_auc']:.6f}，相对深度为 3 的参照提升 {capacity['best']['delta_vs_depth3']:.6f}。极浅树融合进一步得到 OOF AUC {shallow['oof_auc']:.6f}，该次公开榜 AUC 为 {shallow_public:.5f}。开发集选型与比赛评估的排序出现差异，公开榜得分仍以整数档和小数档的两模型融合为本项目已记录最高值。",
        ]
        blocks.extend({"type": "paragraph", "text": text} for text in paragraphs)
    if llm:
        pilot_scores = llm["metrics_at_common_threshold_0_5"]
        label = f"{llm['model']}_{llm['reasoning_effort']}"
        blocks.append({
            "type": "paragraph",
            "text": f"GPT 小样本对照从既有留出集抽取 {llm['evaluation_rows']} 笔交易。GPT 的样本 AUC 为 {pilot_scores[label]['roc_auc']:.4f}，原树融合在同一批样本上的 AUC 为 {pilot_scores['selected']['roc_auc']:.4f}。分层配对区间跨零，结果保存在补充附件。",
        })
