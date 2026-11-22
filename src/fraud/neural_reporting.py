"""将第二轮真实结果加入现有课程报告，明确区分开发集和首轮留出集。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import ROOT, plot_setup, write_json
from .neural_experiments import REPORTS, neural_config

LABELS={"mlp":"MLP","ft_transformer":"FT-Transformer","tabm":"TabM","tree_round1":"首轮树融合","round2_selected":"第二轮选择"}


def load(name):
    return json.loads((REPORTS/name).read_text(encoding="utf-8"))


def figures():
    plt=plot_setup()
    comparison=pd.read_csv(REPORTS/"model_comparison.csv")
    names=comparison.name.tolist()
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,metric,title in zip(axes,["roc_auc","average_precision"],["开发集 OOF ROC-AUC","开发集 OOF AP"]):
        ax.barh([LABELS[name] for name in names],comparison[metric],color=["#D45B55" if name=="round2_selected" else "#4276AA" for name in names])
        ax.set_title(title);ax.grid(axis="x",alpha=0.2);ax.set_axisbelow(True)
        if metric=="roc_auc":ax.set_xlim(min(comparison.roc_auc)-0.005,max(comparison.roc_auc)+0.002)
    fig.tight_layout();fig.savefig(ROOT/"reports/figures/neural_comparison.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for kind in neural_config()["models"]:
        history=load(f"{kind}_fold0_inner_history.json")
        ax.plot([row["epoch"] for row in history],[row["valid_auc"] for row in history],label=LABELS[kind],linewidth=1.7)
    ax.set(title="神经网络内部早停验证 AUC（第 0 折）",xlabel="训练轮数",ylabel="内部验证 ROC-AUC")
    ax.legend();ax.grid(alpha=0.2)
    fig.tight_layout();fig.savefig(ROOT/"reports/figures/neural_learning_curves.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for kind in neural_config()["models"]:
        history=load(f"{kind}_fold0_inner_history.json")
        ax.plot([row["epoch"] for row in history],[row["train_loss"] for row in history],label=LABELS[kind])
    ax.set(title="神经网络训练损失（第 0 折）",xlabel="训练轮数",ylabel="BCE loss");ax.legend();ax.grid(alpha=0.2)
    fig.tight_layout();fig.savefig(ROOT/"reports/figures/neural_training_loss.png");plt.close(fig)


def append_blocks(blocks):
    cfg=neural_config()
    selection=load("selection.json")
    comparison=pd.read_csv(REPORTS/"model_comparison.csv")
    results={kind:load(f"{kind}.json") for kind in cfg["models"]}
    env=load("environment.json")
    def para(text):blocks.append({"type":"paragraph","text":text})
    def table(headers,rows):blocks.append({"type":"table","headers":headers,"rows":rows})
    blocks.append({"type":"heading","level":2,"text":"7 深度学习分类与第二轮融合"})
    actual_gpu=results["mlp"]["folds"][0]["training_device"]
    para(f"第二轮在 {actual_gpu} 上比较 MLP、FT-Transformer 和 TabM，显存峰值及每轮耗时另见硬件实测记录。")
    para("连续数值在每次实际训练数据内拟合中位数填补和正态分位数变换，布尔标记保留 0/1，字符串类别建立训练词表并为未知值预留编号。MLP 使用类别嵌入与 256、128、64 单元的全连接层。FT-Transformer 将各字段编码为 token，以 3 层、64 维、4 个注意力头学习字段交互。TabM 使用官方实现，设置 16 个共享参数成员、3 层 128 维网络和分段线性数值嵌入。")
    para(f"三种方法均采用 AdamW、二元交叉熵和 BF16 混合精度。最多训练 {cfg['max_epochs']} 轮，根据外层训练折内部 10% 样本的 AUC 早停，再以选定轮数重训完整外层训练折。外层验证样本只用于生成 OOF 预测；TabM 对成员概率取平均。完整配置、逐轮损失、早停 AUC、GPU 型号和显存峰值均保存。")
    para(f"神经网络环境为 PyTorch {env['torch']}、CUDA {env['torch_cuda']}、tabm {env['packages']['tabm']} 与 rtdl-revisiting-models {env['packages']['rtdl-revisiting-models']}。训练使用 BF16，评估与提交预测使用 FP32，以减少低精度输出造成的排序并列。")
    table(["方案","OOF AUC","OOF AP","OOF F1","CV耗时 秒"],[[LABELS[row['name']],f"{row['roc_auc']:.6f}",f"{row['average_precision']:.6f}",f"{row['f1']:.4f}",f"{row['training_seconds']:.1f}" if pd.notna(row.get('training_seconds',np.nan)) else ("沿用首轮" if row['name']=="tree_round1" else "组合预测")] for _,row in comparison.iterrows()])
    network_comparison=comparison[comparison.name.isin(cfg["models"])]
    best_network=network_comparison.loc[network_comparison.roc_auc.idxmax()]
    para(f"本轮单个网络中，{LABELS[best_network['name']]} 的开发集 OOF AUC 最高，为 {best_network['roc_auc']:.6f}，相对首轮树融合变化 {best_network['roc_auc']-selection['baseline_auc']:+.6f}。表中 F1 使用各方案在开发集 OOF 上选择的阈值；比赛风险排序仍以 AUC 比较。网络的单独成绩与加入融合后的收益分别评价。")
    blocks.append({"type":"figure","path":"figures/neural_learning_curves.png","caption":"图 8 神经网络内部验证曲线"})
    blocks.append({"type":"figure","path":"figures/neural_comparison.png","caption":"图 9 相同开发集划分下的树模型与神经网络比较"})
    weights=" + ".join(f"{LABELS[name]} × {weight:.1f}" for name,weight in selection["weights"].items())
    para(f"融合将首轮树模型组合作为一路预测，与三种网络组成四路非负权重网格，步长为 0.1，共比较 {selection['candidate_count']} 组。至少提升 0.00005 AUC 且至少两折同向改善时才采用新融合，否则保留原方案。选择结果为 {weights}，开发集 OOF AUC={selection['oof_metrics']['roc_auc']:.6f}，相对首轮变化 {selection['auc_improvement']:+.6f}。三折 AUC 变化分别为 "+"、".join(f"{change:+.6f}" for change in selection["fold_auc_changes"])+"。")
    para("首轮独立测试集已经用于一次最终评估，第二轮沿用开发集固定折开展比较与选型，不把旧留出集重新当作未见数据反复调参。第二轮表格和融合成绩均为开发集 OOF，包含模型和权重选择的影响；首轮独立测试成绩另列，二者不混作同一评估条件。完成本轮选型后，用全部官方有标签样本训练比赛模型并生成新提交文件。")
    if (REPORTS/"finalization.json").exists():
        final=load("finalization.json")
        cv_minutes=sum(result["total_seconds"] for result in results.values())/60
        full_minutes=sum(row["training_seconds"] for row in final["training"].values())/60
        para(f"三种网络的三折流程实测合计 {cv_minutes:.1f} 分钟，全量模型拟合另用 {full_minutes:.1f} 分钟。MLP、FT-Transformer、TabM 的全量训练轮数分别为 {final['epochs']['mlp']}、{final['epochs']['ft_transformer']}、{final['epochs']['tabm']}；均保留可重载权重和单模型比赛预测。融合搜索、文件核验和报告生成耗时另计。")


def run():
    figures()
    from .reporting import run as original_report
    original_report()
    cfg=neural_config()
    selection=load("selection.json")
    benchmark=load("hardware_benchmark.json")
    lines=["# 第二轮神经网络实验记录","",f"硬件决策：{benchmark['decision']}；本机全流程保守估计：{benchmark.get('estimated_suite_minutes')} 分钟。","","## 已运行实验",""]
    for kind in cfg["models"]:
        result=load(f"{kind}.json")
        lines += [f"- {LABELS[kind]}：OOF AUC {result['oof_default']['roc_auc']:.6f}，AP {result['oof_default']['average_precision']:.6f}，CV 实测耗时 {result['total_seconds']/60:.1f} 分钟。"]
    lines += ["",f"融合权重：{selection['weights']}。",f"选择后 OOF AUC：{selection['oof_metrics']['roc_auc']:.6f}，相对首轮 {selection['auc_improvement']:+.6f}。","","本轮只使用开发集选型；旧独立测试成绩保留，未进行本轮重复选参或新的独立测试宣称。","","完整参数和逐轮记录见本目录 JSON；模型、OOF 和最终权重保存在 artifacts/round2；提交文件为 submissions/round2_best.csv。"]
    if (REPORTS/"finalization.json").exists():
        final=load("finalization.json")
        lines += ["","## 全量模型",""]
        for kind,epochs in final["epochs"].items():
            lines += [f"- {LABELS[kind]}：全部有标签样本训练 {epochs} 轮，耗时 {final['training'][kind]['training_seconds']/60:.1f} 分钟，GPU {final['training'][kind]['training_device']}。"]
    (REPORTS/"实验记录.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("第二轮记录、图表和主报告已更新。",flush=True)
