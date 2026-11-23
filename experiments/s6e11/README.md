# S6E11 信用卡欺诈实验与公榜提交

[最佳融合预测](public_blend_45_45_10.csv) 包含 299,984 笔比赛测试交易。每行给出交易 ID 和用于 ROC AUC 评估的排序分数。Kaggle 提交编号为 `58777864`，公榜 AUC 为 **0.95567**。

| 公开预测输入 | 版本 | 权重 | 单独提交的公榜 AUC |
|---|---|---:|---:|
| [Malyshev Danil · Fraud Signal Lab](https://www.kaggle.com/code/malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion) | V44 | 45% | 0.95567 |
| [Areeb Ali · Which Signals Rank Fraud?](https://www.kaggle.com/code/areebbb/s6e11-which-signals-rank-fraud-lb-0-95567) | V16 | 45% | 0.95567 |
| [goodpjw2008 · Board Model + RealMLP](https://www.kaggle.com/code/goodpjw2008/s6e11-board-model-realmlp-lb-0-95562) | V40 | 10% | 0.95562 |

复现时取上述版本各自的 `submission.csv`，按比赛模板核对 ID 顺序，对每份文件的 `is_fraud` 独立计算 pandas `rank(method="average", pct=True)`，再按表中权重相加。等权方案公榜为 0.95566。最佳融合与前两份原版在公榜显示精度上持平。

本目录另收录团队的训练与诊断脚本。自训加性网络由 `train_context_network.py` 生成，单独提交的公榜 AUC 为 **0.95016**。它的分数与上述三方公开预测融合分别记录。

运行自训基线时，以本目录为工作目录，将比赛的 `train.csv`、`test.csv` 和 `sample_submission.csv` 放在 `.kaggle-cache/competitions/playground-series-s6e11/`，创建 `runs/`，依次运行 `python prepare_split.py` 和 `python train_context_network.py`。该实验使用 NumPy、pandas、scikit-learn 和支持 CUDA 的 PyTorch。其他诊断脚本按各自源码读取 `runs/` 中的前序预测和 `references/` 中的指定公开模型文件。
