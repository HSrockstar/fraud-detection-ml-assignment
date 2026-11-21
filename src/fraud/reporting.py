from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from .common import ROOT, TARGET, config, plot_setup, write_json

LABELS = {"logistic_baseline": "逻辑回归", "tree_baseline": "决策树", "xgb_baseline": "XGBoost 基线", "cat_baseline": "CatBoost 基线", "xgb_tuned": "XGBoost 调参", "xgb_engineered": "XGBoost 特征工程", "xgb_target_encoded": "XGBoost 目标编码", "selected": "最终选择"}


def read_json(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def plots(comparison, holdout, selection):
    plt = plot_setup()
    import seaborn as sns
    data = pd.read_parquet(ROOT / "artifacts/evaluation/holdout_predictions.parquet")
    y = data[TARGET]
    names = comparison.name.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    position = np.arange(len(names))
    for ax, metric, title in zip(axes, ["roc_auc", "average_precision"], ["ROC-AUC", "Average Precision"]):
        ax.barh(position-0.18, [comparison.set_index("name").loc[n, metric] for n in names], height=0.35, label="开发集 OOF", color="#4276AA")
        ax.barh(position+0.18, [holdout[n]["oof_threshold"][metric] for n in names], height=0.35, label="独立测试集", color="#E6A34A")
        ax.set_yticks(position, [LABELS.get(n, n) for n in names]); ax.set_title(title); ax.legend()
        ax.grid(axis="x", alpha=0.2); ax.set_axisbelow(True)
        if metric == "roc_auc": ax.set_xlim(0.85, 0.97)
    fig.tight_layout(); fig.savefig(ROOT / "reports/figures/model_comparison.png"); plt.close(fig)
    for kind in ["roc", "pr"]:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for name in names + ["selected"]:
            label = LABELS.get(name, name)
            if kind == "roc":
                RocCurveDisplay.from_predictions(y, data[name], name=label, ax=ax)
            else:
                PrecisionRecallDisplay.from_predictions(y, data[name], name=label, ax=ax)
        ax.set_title("独立测试集 ROC 曲线" if kind == "roc" else "独立测试集 Precision Recall 曲线")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(ROOT / f"reports/figures/{kind}_curves.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, threshold in zip(axes, [0.5, selection["f1_threshold"]]):
        matrix = confusion_matrix(y, data.selected >= threshold)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["正常", "欺诈"], yticklabels=["正常", "欺诈"])
        ax.set(title=f"阈值 {threshold:.3f}", xlabel="预测类别", ylabel="真实类别")
    fig.suptitle("真实类别与预测类别对比（最终选择模型）")
    fig.tight_layout(); fig.savefig(ROOT / "reports/figures/confusion_matrix.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for target, color in [(0, "#4276AA"), (1, "#D45B55")]:
        values = data.loc[y == target, "selected"].clip(1e-6, 1)
        ax.hist(values, bins=np.geomspace(1e-6, 1, 60), histtype="step", density=True, color=color, label="真实欺诈" if target else "真实正常", linewidth=1.5)
    ax.axvline(selection["f1_threshold"], color="black", linestyle="--", label="开发集选择的 F1 阈值")
    ax.set(xscale="log", yscale="log", title="独立测试集预测概率分布", xlabel="预测欺诈概率", ylabel="密度（各类别分别归一化）"); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "reports/figures/prediction_distribution.png"); plt.close(fig)
    best = next(iter(selection["weights"]))
    importance = read_json(f"../artifacts/{best}/feature_importance.json")
    importance = pd.Series(importance).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    importance.plot.barh(ax=ax, color="#4276AA")
    ax.set(title=f"{LABELS.get(best,best)} 第一折特征重要性", xlabel="模型重要性（用于解释关联）")
    fig.tight_layout(); fig.savefig(ROOT / "reports/figures/feature_importance.png"); plt.close(fig)
    trials = pd.read_csv(ROOT / "reports/optimization_trials.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trials.number, trials.value, "o-", color="#4276AA", label="本次搜索分数")
    ax.plot(trials.number, trials.value.cummax(), "--", color="#D45B55", label="累计最优")
    ax.set(title="XGBoost 超参数搜索（开发集第 0 折）", xlabel="试验序号", ylabel="ROC-AUC"); ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(ROOT / "reports/figures/optimization.png"); plt.close(fig)


def run():
    eda, env, selection, holdout, optimization = [read_json(n) for n in ["eda_summary.json", "environment.json", "selection.json", "holdout_metrics.json", "optimization.json"]]
    comparison = pd.read_csv(ROOT / "reports/model_comparison.csv")
    cfg = config()
    team = json.loads((ROOT / "configs/team.json").read_text(encoding="utf-8"))
    plots(comparison, holdout, selection)
    blocks = []
    def heading(text, level=1): blocks.append({"type": "heading", "level": level, "text": text})
    def para(text): blocks.append({"type": "paragraph", "text": text})
    def table(headers, rows): blocks.append({"type": "table", "headers": headers, "rows": rows})
    def figure(name, caption): blocks.append({"type": "figure", "path": "figures/" + name + ".png", "caption": caption})
    blocks.append({"type": "title", "text": "信用卡欺诈预测分类实验报告"})
    table(["姓名 / 学号", "具体任务及贡献"], [[f"{m['name']} / {m['student_id']}", m["contribution"]] for m in team])
    chosen = " + ".join(f"{LABELS.get(n,n)} × {w:.1f}" for n,w in selection["weights"].items())
    final = holdout["selected"]["oof_threshold"]
    heading("一 课题综述")
    heading("1 课题简介", 2)
    para(f"本项目参加 Kaggle Playground Series S6E11 信用卡欺诈预测比赛，并完成机器学习课程项目 2 的分类实验。模型根据交易金额、账户状态、商户风险、验证方式及行为标记等特征，预测交易为欺诈的概率。我们在统一数据划分下比较逻辑回归、决策树、XGBoost 与 CatBoost，再通过参数优化、特征工程及融合选择提升预测表现。首轮按开发集选择的方案为 {chosen}，独立测试集 ROC-AUC 为 {final['roc_auc']:.6f}。")
    round2_path = ROOT / "reports/round2/selection.json"
    if round2_path.exists():
        round2 = json.loads(round2_path.read_text(encoding="utf-8"))
        para(f"第二轮进一步加入 MLP、FT-Transformer 和 TabM，比较神经网络与树模型的互补性。本轮所选方案的开发集 OOF AUC 为 {round2['oof_metrics']['roc_auc']:.6f}；本轮沿用开发集选型，首轮独立测试成绩保留。")
    heading("2 课题目标", 2)
    para("比赛以 ROC-AUC 为主要指标；课程实验同时考察类别不平衡下的精确率、召回率和 F1，分析不同分类方法的适用条件。实验目标包括可复现的数据预处理、公平的模型比较、至少一种模型的优化，以及真实类别与预测类别的可视化分析。")
    heading("3 课题数据集", 2)
    para(f"数据来自 Kaggle 官方比赛，共 {eda['train_rows']:,} 条有标签样本和 {eda['competition_test_rows']:,} 条比赛测试样本。去除 id 与 is_fraud 后有 {eda['feature_count']} 个输入特征，包括 13 个数值特征、5 个字符串类别特征和 6 个布尔标记。id 只用于对应样本，不参与训练。比赛测试标签不公开，所有本地评估均来自官方训练数据的内部划分。")
    para(f"开发集欺诈比例为 {eda['dev_fraud_fraction']:.4%}，约每 60 笔交易出现一笔欺诈；全部预测为正常即可取得 {eda['all_normal_accuracy_on_dev']:.4%} 的准确率，却完全无法发现欺诈。因此准确率作为辅助指标，不能单独用于选择模型。")
    figure("class_balance", "图 1 开发集类别分布")
    heading("二 实验设计与结果")
    heading("1 数据准备与预处理", 2)
    para(f"以随机种子 {cfg['seed']} 按标签分层预留 20% 独立测试数据，即 {eda['holdout_rows']:,} 条；其余 {eda['dev_rows']:,} 条作为开发集。开发集固定为 {cfg['n_splits']} 折，所有模型共用划分。每次用两折训练、一折预测，汇总得到每条开发样本的折外预测。划分与数据指纹保存在 artifacts/splits.parquet 及配套 JSON 中。")
    para(f"开发集共有 {eda['total_dev_missing']} 个缺失值，未发现完全相同的重复特征行。逻辑回归使用数值列中位数填补、标准化和类别独热编码；决策树及 XGBoost 保留数值尺度，对字符串类别进行独热编码；CatBoost 原生处理类别特征。填补、缩放和类别词表均只在对应训练数据上拟合，验证集的新类别采用未知值处理。金额等长尾变量保留原值，避免把可能有用的欺诈信号当作异常删除。")
    para("数值分布比较与类别欺诈比例仅用于开发集探索。布尔标记在模型中按 0/1 使用。训练开发集与比赛测试集的数值 KS 统计量较小，本轮未发现明显的单变量分布偏移；这一检查不等同于证明联合分布相同。")
    figure("feature_distributions", "图 2 开发集中关联较强的数值特征分布")
    para("CVV 重试次数、商户风险和交易速度与标签的关联较明显；单变量相关性反映边际关系，不能解释全部交互效应，也不构成因果结论。类别特征中不同验证方式和商户类型对应的欺诈比例也存在差异。")
    heading("2 模型搭建", 2)
    table(["方法", "基本原理", "关键设置"], [
        ["逻辑回归", "线性特征组合经 Sigmoid 映射为概率，使用 L2 正则化", "C=1，最多 1000 次迭代"],
        ["决策树", "递归划分特征空间，在叶节点形成分类概率", "最大深度 8，叶节点至少 100 样本，类别平衡权重"],
        ["XGBoost", "逐轮拟合损失梯度，组合多棵树以刻画非线性和交互", "GPU hist，基线深度 5，学习率 0.04，L2=10"],
        ["CatBoost", "梯度提升树，原生支持类别特征并施加正则化", "GPU，深度 6，学习率 0.04，L2=8"],
    ])
    para("逻辑回归与决策树代表不同的建模思路，满足课程对至少两类传统分类方法的要求；两种树提升方法用于进一步争取比赛成绩。CPU 用于轻量对照模型，主要树提升训练均强制使用 GPU。")
    heading("3 模型训练与评估", 2)
    para(f"运行环境为 Windows、Python {env['python']}、{env['gpu']}。主要依赖版本为 scikit-learn {env['packages']['scikit-learn']}、XGBoost {env['packages']['xgboost']} 和 CatBoost {env['packages']['catboost']}。GPU 自检实际完成两类模型的训练与预测，并核对训练设备；完整版本锁定在 requirements-lock.txt。")
    para("树提升训练在每个外层训练折内部再分出 10% 用于早停，随后按选定轮数重训整个外层训练折。外层验证折不参与早停。主指标 ROC-AUC 衡量风险排序；Average Precision 总结精确率召回率曲线，对少数欺诈类别更敏感；同时记录准确率、Precision、Recall、F1 和对数损失。分类阈值在开发集折外预测上按 F1 最大化选择，并原样用于独立测试集。")
    table(["实验", "OOF AUC", "折间标准差", "OOF AP", "耗时 秒"], [[LABELS.get(r['name'],r['name']), f"{r['roc_auc']:.6f}", f"{r['std_fold_auc']:.6f}", f"{r['average_precision']:.6f}", f"{r['seconds']:.1f}"] for _,r in comparison.iterrows()])
    para("表中 OOF 成绩用于模型开发，参数搜索和融合权重也在开发集上选择，因此最优方案的 OOF 成绩存在选择偏差。独立测试集在方案冻结后用于最终评估，结果如下；测试结果不反向修改本轮选型、权重或阈值。")
    para("OOF 每折模型使用开发集的三分之二训练，最终测试模型使用完整开发集训练，轮数固定为交叉验证所选轮数的中位数。因此两阶段分数的差异同时包含训练样本量和抽样差异，应在相同阶段内比较不同方法。")
    table(["实验", "测试 AUC", "测试 AP", "Precision", "Recall", "F1"], [[LABELS.get(n,n), *[f"{holdout[n]['oof_threshold'][m]:.4f}" for m in ['roc_auc','average_precision','precision','recall','f1']]] for n in comparison.name.tolist()+["selected"]])
    heading("4 结果可视化与模型对比", 2)
    figure("model_comparison", "图 3 开发集与独立测试集性能对比")
    para("逻辑回归提供了较强的线性基准，说明部分风险因素可以通过近似可加的关系解释。单棵决策树受到分段划分和有限复杂度的约束，排序信息较粗。树提升模型通过多轮弱学习器组合表达细粒度非线性关系，其收益需要结合 AUC、AP 和训练成本共同判断。")
    para("决策树使用类别平衡权重，使少数类在训练损失中获得更高权重，也会改变输出概率的尺度。因此各模型使用各自在开发集选择的阈值计算分类指标，同时保留默认 0.5 阈值结果；AUC 直接使用连续预测，不依赖阈值。")
    figure("pr_curves", "图 4 独立测试集精确率召回率曲线")
    figure("confusion_matrix", "图 5 真实类别与预测类别对比")
    para(f"最终方案的开发集 F1 阈值为 {selection['f1_threshold']:.6f}。在独立测试集上，该阈值对应 Precision={final['precision']:.4f}、Recall={final['recall']:.4f}、F1={final['f1']:.4f}。默认 0.5 阈值与该阈值的混淆矩阵并列展示，可观察误报与漏报之间的变化。比赛提交保留连续概率，阈值变化本身不改变 ROC-AUC。")
    heading("5 模型优化与分析", 2)
    para(f"首先对 XGBoost 进行 {optimization['completed_trials']} 次 Optuna 搜索，范围包括深度 3 至 7、学习率 0.025 至 0.09、子样本比例、列采样比例、叶节点约束及 L1/L2 正则化。搜索只使用开发集第 0 折，候选参数随后在完整三折上评估。搜索折参与选参，其最高成绩不视为独立性能估计。")
    table(["参数", "搜索选定值"], [[key, f"{value:.6g}" if isinstance(value,float) else str(value)] for key,value in optimization["best_params"].items()])
    para("进一步增加交易金额与账户余额比值、金额和距离对数、时刻正余弦、夜间交易标记、风险标记计数，以及商户风险与交易速度等交互项。特征均由当前样本的输入计算，不使用标签。参数优化模型与特征工程模型使用相同参数和数据划分，从而单独观察特征构造的效果。")
    para("另设目标编码实验，在保留原始数值与独热特征的基础上，为每个原始特征增加平滑欺诈率编码。scikit-learn TargetEncoder 使用 5 折内部交叉拟合和平滑强度 20，训练编码通过 fit_transform(X, y) 生成，外层验证、早停子集与最终测试样本均仅调用 transform。编码在每一次实际训练内重新拟合，不使用外层验证或独立测试标签。")
    base = comparison.set_index("name")
    delta = base.loc["xgb_tuned", "roc_auc"] - base.loc["xgb_baseline", "roc_auc"]
    feature_delta = base.loc["xgb_engineered", "roc_auc"] - base.loc["xgb_tuned", "roc_auc"]
    para(f"调参相对 XGBoost 基线的 OOF AUC 变化为 {delta:+.6f}；同参数下增加特征后的变化为 {feature_delta:+.6f}。正值表示本次开发集比较有所改善，负值表示该尝试未带来提升。模型选择保留实际效果较好的方案，而不是默认特征越多越好。")
    te_delta = base.loc["xgb_target_encoded", "roc_auc"] - base.loc["xgb_tuned", "roc_auc"]
    para(f"交叉拟合目标编码相对同参数调参模型的 OOF AUC 变化为 {te_delta:+.6f}。这一处理尝试从重复出现的特征取值中提取非线性风险信息，不能使用整份训练数据先计算目标均值再做交叉验证，否则会发生标签泄漏。")
    para(f"目标编码模型的折间 AUC 标准差为 {base.loc['xgb_target_encoded','std_fold_auc']:.6f}，原始 XGBoost 基线为 {base.loc['xgb_baseline','std_fold_auc']:.6f}。编码依赖重复取值的样本频率，各折少数类和取值覆盖的变化会影响估计；平滑与内部交叉拟合用于控制这种波动。")
    figure("optimization", "图 6 参数搜索过程")
    para(f"最后在开发集折外预测上比较单模型与 XGBoost/CatBoost 的概率加权平均，权重步长为 0.1。只有融合 AUC 比最佳单模型至少提高 0.00005 时才增加模型组合。冻结结果为 {chosen}，OOF AUC={selection['oof_metrics']['roc_auc']:.6f}。")
    heading("6 拓展探索", 2)
    para("本轮拓展包括超参数分析、无标签组合特征、交叉拟合目标编码、模型融合及特征重要性分析。后续可在开发集研究交互目标编码和原始数据源辅助特征，通过消融实验检查收益；新一轮选择须与本轮独立测试结果的使用历史一起记录。")
    figure("feature_importance", "图 7 所选方案成员模型的特征重要性")
    if round2_path.exists():
        from .neural_reporting import append_blocks
        append_blocks(blocks)
    heading("三 实验总结")
    para(f"本项目完成了信用卡欺诈二分类的数据分析、四类模型基线、GPU 树提升训练、超参数优化、特征工程、融合选择和独立测试。首轮方案测试 ROC-AUC={final['roc_auc']:.6f}，AP={final['average_precision']:.6f}。类别严重不平衡时，应结合风险排序和少数类识别指标评价模型，避免高准确率掩盖漏检。")
    para("完成独立测试后，按开发集交叉验证得到的训练轮数，在全部官方有标签数据上重训所选方案，并生成与官方样例列名及 id 顺序一致的提交文件。当前材料记录本地实验结果；Kaggle 上传与线上评分状态单独保存，不能以本地 AUC 代替线上成绩。")
    if round2_path.exists():
        para(f"加入三种神经网络后，第二轮开发集 OOF AUC 为 {round2['oof_metrics']['roc_auc']:.6f}，相对首轮融合变化 {round2['auc_improvement']:+.6f}。各网络及融合的预测、训练耗时与持久化模型均保留，可据此继续比较精度和计算成本。")
    heading("参考资料")
    for text in [
        "[1] Kaggle. Predicting Credit Card Fraud. https://www.kaggle.com/competitions/playground-series-s6e11",
        "[2] scikit-learn. Model evaluation and cross-validation documentation. https://scikit-learn.org/stable/modules/model_evaluation.html",
        "[3] XGBoost. GPU support documentation. https://xgboost.readthedocs.io/en/stable/gpu/index.html",
        "[4] CatBoost. Training on GPU. https://catboost.ai/docs/en/features/training-on-gpu",
        "[5] Optuna. Documentation. https://optuna.readthedocs.io/",
        "[6] scikit-learn. TargetEncoder and internal cross fitting. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html",
        "[7] JamesAyCheese. Credit Card Fraud Detection Kaggle Playground S6E11. https://github.com/jamesaycheese/kaggle-credit-card-fraud",
        *(["[8] Gorishniy et al. Revisiting Deep Learning Models for Tabular Data. https://github.com/yandex-research/rtdl-revisiting-models", "[9] Gorishniy et al. TabM Advancing Tabular Deep Learning With Parameter Efficient Ensembling. https://github.com/yandex-research/tabm"] if round2_path.exists() else []),
    ]: para(text)
    write_json(ROOT / "reports/report_content.json", blocks)
    markdown = []
    for b in blocks:
        if b["type"] == "title": markdown.append("# " + b["text"])
        elif b["type"] == "heading": markdown.append("#" * (b["level"]+1) + " " + b["text"])
        elif b["type"] == "paragraph": markdown.append(b["text"])
        elif b["type"] == "figure": markdown.append(f"![{b['caption']}]({b['path']})\n\n{b['caption']}")
        elif b["type"] == "table":
            markdown.append("\n".join(["| " + " | ".join(b["headers"]) + " |", "| " + " | ".join(["---"] * len(b["headers"])) + " |"] + ["| " + " | ".join(str(c) for c in row) + " |" for row in b["rows"]]))
    (ROOT / "reports/实验报告.md").write_text("\n\n".join(markdown)+"\n", encoding="utf-8")
    print("报告正文和图表已按实际实验结果生成。", flush=True)
