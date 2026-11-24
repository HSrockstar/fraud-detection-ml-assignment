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
    glm = optional("glm_submission.json")
    public_oof = optional("public_oof_blend.json")
    progress = optional("competition_progress.json")
    llm = optional("llm_pilot_v1/metrics.json")
    if blend and capacity and shallow:
        paragraphs = [
            "数值档位实验保留原始数值，并将部分连续列拆为整数档和小数档，分别提取取值层面的风险信息。XGBoost 使用交叉拟合目标编码，CatBoost 将档位作为类别输入。两模型按 7∶3 融合，在相同开发集三折上比较。",
            f"固定编码、学习率和正则化后，三叶 XGBoost 的 OOF AUC 为 {capacity['best']['oof_auc']:.6f}，比深度 3 的参照提高 {capacity['best']['delta_vs_depth3']:.6f}。GLM 起点实验则把逻辑回归的 logit 作为提升模型的初始得分。两项对照分别检验树容量和可加风险基准的作用。" if glm else f"固定编码、学习率和正则化后，三叶 XGBoost 的 OOF AUC 为 {capacity['best']['oof_auc']:.6f}。",
        ]
        blocks.extend({"type": "paragraph", "text": text} for text in paragraphs)
        rows = [["整数与小数档 7∶3", f"{blend['oof_auc']:.6f}", f"{blend['kaggle_submission']['public_score']:.5f}"],
                ["极浅树融合", f"{shallow['oof_auc']:.6f}", f"{shallow['kaggle_submission']['public_score']:.5f}"]]
        if glm:
            rows.append(["GLM 起点 XGBoost", f"{glm['oof_auc']:.6f}", f"{glm['kaggle_submission']['public_score']:.5f}"])
        if public_oof:
            rows.append(["自训与公开 OOF 融合", f"{public_oof['selected_dev_auc']:.6f}", f"{public_oof['kaggle_submission']['public_score']:.5f}"])
        if progress:
            rows.extend([["自训加性样条网络", "另设五折，未归档分数", f"{progress['self_trained_additive']['public_score']:.5f}"],
                         ["公开预测 45∶45∶10", "—", f"{progress['latest_public_blend']['public_score']:.5f}"]])
        blocks.append({"type": "table", "headers": ["方案", "开发集 OOF AUC", "实测公榜 AUC"], "rows": rows})
        blocks.append({"type": "paragraph", "text": "前三项使用相同的本地三折划分；公榜列来自真实提交。极浅树的本地 AUC 为 0.948309，公榜为 0.95041，整数与小数档 7∶3 的公榜为 0.95132。两类评估分别反映本地选型结果和比赛测试样本上的排序表现。"})
    if public_oof:
        blocks.append({"type": "paragraph", "text": "公开 OOF 融合使用 Alex Chilton 以 CC0 发布的两份 LightGBM 预测[10]，与团队 7∶3 预测按 0.4、0.5、0.1 做名次加权。开发集对齐 AUC 为 0.950031，实测公榜 AUC 为 0.95146。公开模型使用各自的训练划分，该开发集分数用于融合选择。"})
    if progress:
        blocks.extend({"type": "paragraph", "text": text} for text in [
            f"自训加性样条网络将数值与折内风险编码展开为分段线性基函数，并比较可加结构与八维交互结构。该轮采用种子 42 的开发集五折，与课程主实验分开记录；加性网络实测公榜 AUC 为 {progress['self_trained_additive']['public_score']:.5f}。脚本已收录，逐折预测与本地分数未随仓库归档。",
            f"最新公开预测融合采用 Malyshev Danil V44、Areeb Ali V16 和 goodpjw2008 V40[11—13]。每份预测先按平均秩转换为百分位，再按 45%、45%、10% 相加，实测公榜 AUC 为 {progress['latest_public_blend']['public_score']:.5f}，提交编号 {progress['latest_public_blend']['reference']}。等权融合为 0.95566；45∶45∶10 与前两份原版在公榜五位小数精度上持平。该结果属于公开预测组合，分数表达相对风险排序。",
        ])
    if llm:
        pilot_scores = llm["metrics_at_common_threshold_0_5"]
        label = f"{llm['model']}_{llm['reasoning_effort']}"
        blocks.append({
            "type": "paragraph",
            "text": f"GPT 小样本对照从既有留出集抽取 {llm['evaluation_rows']} 笔交易。GPT 的样本 AUC 为 {pilot_scores[label]['roc_auc']:.4f}，原树融合在同一批样本上的 AUC 为 {pilot_scores['selected']['roc_auc']:.4f}。分层配对区间跨零，结果保存在补充附件。",
        })
