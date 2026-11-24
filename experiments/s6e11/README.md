# S6E11 信用卡欺诈实验与公榜提交

[最高分融合预测](public_blend_45_45_10.csv) 包含 299,984 笔比赛测试交易，每行给出交易 ID 和风险排序分数。Kaggle 提交编号为 `58777864`，实测公榜 AUC 为 **0.95567**。2026-11-24 已通过只读 API 重新核对提交状态，并检查 CSV 的 ID 顺序、数值范围和 SHA256，见 [核验记录](../../reports/competition_progress.json)。

| 公开预测输入 | 版本 | 权重 | 单独提交的公榜 AUC |
|---|---|---:|---:|
| [Malyshev Danil · Fraud Signal Lab](https://www.kaggle.com/code/malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion) | V44 | 45% | 0.95567 |
| [Areeb Ali · Which Signals Rank Fraud?](https://www.kaggle.com/code/areebbb/s6e11-which-signals-rank-fraud-lb-0-95567) | V16 | 45% | 0.95567 |
| [goodpjw2008 · Board Model + RealMLP](https://www.kaggle.com/code/goodpjw2008/s6e11-board-model-realmlp-lb-0-95562) | V40 | 10% | 0.95562 |

复现时取上述固定版本各自的 `submission.csv`，按比赛模板核对 ID 顺序，对每份文件的 `is_fraud` 独立计算 pandas `rank(method="average", pct=True)`，再按表中权重相加。等权方案实测公榜为 0.95566；45∶45∶10 与前两份原版在公榜五位小数精度上持平。本仓库没有归档用于该三路权重选型的 OOF。融合值表达相对风险排序，不能直接解释为校准后的欺诈概率。

## 自训加性网络

`train_context_network.py` 将数值和训练折内的风险编码展开为分段线性基函数，比较可加结构与八维低秩交互，固定训练 12 轮。加性网络单独提交的实测公榜 AUC 为 **0.95016**，提交编号 `58775843`。本仓库保留代码和 API 记录，该轮 `runs/`、逐折预测、本地分数和权重尚未归档。

后续脚本采用种子 42 的独立五折方案：先按 ID 排序，再在每个标签内分出约 20% 标为 `fold=-1`，其余分成五折。课程主实验采用种子 20261121 的三折，两套结果分别记录。

以本目录为工作目录，放置比赛的三个原始 CSV：

```text
.kaggle-cache/competitions/playground-series-s6e11/train.csv
.kaggle-cache/competitions/playground-series-s6e11/test.csv
.kaggle-cache/competitions/playground-series-s6e11/sample_submission.csv
```

在具备 NumPy、pandas、scikit-learn 和 CUDA PyTorch 的环境执行：

```powershell
# 从项目根目录进入本目录，使用已安装神经网络依赖的项目环境
Set-Location experiments/s6e11
New-Item -ItemType Directory -Path runs -Force | Out-Null
..\..\.venv\Scripts\python.exe prepare_split.py
..\..\.venv\Scripts\python.exe train_context_network.py
```

运行会依次生成 `runs/context_network/fold0.npz` 至 `fold4.npz`，汇总 OOF 后按本地 AUC 选择结构，再用全部有标签数据训练并输出 `single_<结构>.csv`。训练行的风险编码采用内部五折，外层验证使用训练折统计；离散键的枚举使用拼接后的输入特征。首次复现先使用空的 `runs/context_network/`，避免复用其他配置的折文件。

## 其他脚本与前序依赖

| 脚本 | 用途 | 额外依赖或输入 |
|---|---|---|
| `run_realmlp.py` | RealMLP 训练与全量重训 | `pytabkit`、PyTorch；生成后续脚本读取的逐折预测 |
| `train_rate_glm.py`、`train_digit_glm.py`、`train_joint_rate_glm.py` | 风险率、数值档位与联合 GLM | 部分比较段读取 RealMLP 预测和公开参考文件 |
| `train_bpe_tree.py` | 原始数值、十进制键和 BPE 数字键的树模型对照 | `lightgbm`、`references/gpt2_124m_vocab.bpe`，源码校验其 SHA256 |
| `compare_tree_member.py`、`compare_second_lm.py` | 新成员互补性及配对排序比较 | `runs/` 中的前序预测、`references/` 和指定公开预测缓存 |
| `build_step_h_glm.py` | 公开 Board Model V35 与自训 GLM 的 95∶5 融合 | 固定版本公开 CSV、自训 GLM 全量预测；校验输入 SHA256 |
| 其余 `train_*`、`audit_*`、`investigate_hard_cases.py` | 潜在状态、掩码字段、配对 AUC 与错排诊断 | 各脚本源码列明的前序文件 |

本目录不是统一的一键流水线，主环境锁文件也不包含全部后续实验依赖。先补齐各脚本指定输入，再运行对应实验。`references/`、`.kaggle-cache/` 和该轮 `runs/` 未随本仓库归档，现成公开融合 CSV 可直接检查和使用。

课程主实验、选做实验与这些探索的关系见 [实验进展与复现索引](../../docs/实验进展与复现索引.md)。
