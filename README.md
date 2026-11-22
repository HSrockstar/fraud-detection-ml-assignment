# 信用卡欺诈预测与机器学习分类作业

本项目参加 [Kaggle Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11)，同时完成机器学习课程项目 2。比赛指标为 ROC-AUC，预测目标为 `is_fraud`。训练优先使用本机 NVIDIA GPU；逻辑回归与决策树作为 CPU 对照。

## 环境

使用 Python 3.12，在项目内创建独立 `.venv`，不修改已有 conda 环境。当前机器的环境与真实 GPU 自检见 `reports/environment.json`；完整依赖版本见 `requirements-lock.txt`。

```powershell
# 首次创建：将 Python 路径替换为本机 Python 3.12
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Python 'E:\miniconda\envs\agent\python.exe'

# 以下命令均从项目根目录执行，无需激活环境
.\.venv\Scripts\python.exe scripts/check_gpu.py
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe scripts/run.py all
```

XGBoost 和 CatBoost 必须通过实际 GPU 训练自检。正式实验遇到 GPU 不可用会报错，不会悄悄切换 CPU。XGBoost 对 CPU 上的数据进行预测时使用 CPU；训练使用 CUDA。单 GPU 的训练顺序执行，避免争用显存。CatBoost 的 GPU 浮点计算存在非确定性，固定种子不保证逐位一致。

数据下载使用用户目录下已有 Kaggle 凭据；不要将密钥写入本项目。须先在比赛页面报名并接受规则。`download_data.py` 只下载并解压官方三个 CSV，记录 SHA256；原始数据和环境均被 Git 忽略。

## 分阶段运行

```powershell
.\.venv\Scripts\python.exe scripts/run.py eda
.\.venv\Scripts\python.exe scripts/run.py baseline
.\.venv\Scripts\python.exe scripts/run.py optimize --trials 10
.\.venv\Scripts\python.exe scripts/run.py finalize
.\.venv\Scripts\python.exe scripts/run.py extensions
.\.venv\Scripts\python.exe scripts/run.py report
.\.venv\Scripts\python.exe scripts/run.py verify
.\.venv\Scripts\python.exe -m pytest -q
```

- `eda`：检查数据、生成开发集统计和分析图表。
- `baseline`：比较逻辑回归、决策树、GPU XGBoost、GPU CatBoost。
- `optimize`：10 次 XGBoost 搜索，再通过三折比较调参、组合特征与内部交叉拟合目标编码效果。
- `finalize`：先冻结 OOF 选出的模型、权重及阈值，再进行独立测试；随后用全部有标签数据重训，生成比赛 CSV。
- `extensions`：重建手写逻辑回归与 sklearn 的对照，以及训练样本量、输入噪声、异常值三个鲁棒性实验。
- `report`：根据实际结果生成报告正文和图表。
- `verify`：以首轮冻结清单为范围，复算指标、核对划分、重载模型并检查提交格式。已有选做结果时，同时复算其全部场景的折外指标。

`all` 可从头执行以上步骤。已有冻结方案时，继续未完成的最终处理，并运行或复用选做实验后刷新报告与核验。根目录运行 pytest 默认只收集项目的 `tests/`。

## 手写逻辑回归与鲁棒性

`scripts/run.py extensions` 只使用固定开发集三折。手写模型用 NumPy 和带回溯线搜索的牛顿法，采用与 scikit-learn L-BFGS 相同的 L2 目标函数及训练折预处理。记录三折 AUC、AP、对数损失、拟合耗时、训练目标及折外预测差值。

鲁棒性比较逻辑回归和 XGBoost，按标签分层取每折训练集的 10%、25%、50%、100%，验证样本保持相同。预测阶段噪声按训练折 IQR 缩放，异常值按训练折 Q3 加十倍 IQR 生成。两模型共享同一批扰动样本，结果及具体参数见 `reports/extensions/results.json`。用于复算的逐行预测在 `artifacts/extensions/oof.parquet`，由运行入口生成。

## 评估约定

官方 699,960 条有标签数据按标签分层保留 20% 作为独立测试集，其余 80% 为开发集，固定三折。id 不参与建模。数据划分持久化，所有对照共用同一划分。

填补、标准化和类别编码只在训练数据上拟合。树提升的早停只使用外层训练折内部的 10% 样本，选择轮数后重训整个外层训练折；外层验证折不参与早停。搜索和融合使用开发集 OOF，F1 阈值也只在 OOF 上选择。选型用过的 OOF 分数不是独立泛化估计。

冻结方案后，独立测试集仅承担本轮最终评估。比赛测试集没有公开标签，不能用它计算本地分类指标。最终提交使用全部有标签样本重训后的模型。

目标编码在每次实际训练内部使用五折交叉拟合与平滑强度 20；对训练数据调用 `fit_transform(X, y)`，对外部数据只调用 `transform(X)`。测试包含唯一值不暴露自身标签的反例检查。

开发集上的排序上限记在 `reports/ranking_ceiling.md`：现有列上不再加特征或更换模型。本轮测试集已被查看，后续不得把它称为新的未见测试集，也不能通过反复查看该测试集挑选模型。新增实验应另存名称、配置和轮次，保留本轮材料。

## 文件位置

| 目录或文件 | 内容 |
|---|---|
| `configs/` | 固定实验配置、四人名单，分工留白 |
| `scripts/` | 环境、下载、GPU 核验与实验入口 |
| `src/fraud/` | 数据、特征、模型、评估、优化、报告与核验代码 |
| `tests/` | 防止预处理泄漏、划分混淆及无效概率的检查 |
| `data/raw/` | 官方原始 CSV 与下载压缩包，不加入 Git |
| `artifacts/` | 持久化划分、模型、OOF 和最终测试预测，不加入 Git |
| `reports/` | 实际成绩、分析图、环境与数据校验记录、实验报告 |
| `submissions/round1_best.csv` | 首轮所选方案的比赛提交文件 |
| `submissions/round2_best.csv` | 加入三种神经网络比较后的第二轮提交文件 |
| `作业要求/` | 教师原始要求和模板，保留原件 |

首轮结果见 `reports/model_comparison.csv`、`reports/holdout_metrics.json`、`reports/selection.json` 和 `reports/verification.json`；第二轮结果另存于 `reports/round2/`。报告保留模板章节并改为分类任务；组员分工按用户要求暂时留白。

Word 报告根据 `reports/report_content.json` 和教师 DOCX 模板生成。需要重建时，在具备 `requirements-docs.txt` 中可选依赖的 Python 中运行 `scripts/build_docx.py`；本次交付使用 Codex 文档运行时构建并渲染核验。报告与代码的课程要求对应关系见 `docs/作业要求核对.md`。提交前须补齐当前按要求留白的分工栏。

`scripts/package_assignment.py` 在实验和文档核验通过后创建作业 ZIP。ZIP 内的 `实验报告/` 是本轮报告与结果快照，源码重新运行时写入新的 `reports/`，因此不会把快照误认为已经完成的模型缓存。提交包不包含原始数据与模型权重；按下载步骤恢复数据，再运行 `all` 可重新训练。

生成 CSV 不会上传 Kaggle。上传会改变外部账号状态，须由用户明确指示后执行；线上分数和排名只能在真实提交后记录。

已训练模型可直接重载预测，无需重新训练：

```powershell
.\.venv\Scripts\python.exe scripts/predict.py --input data/raw/test.csv --output submissions/predictions_reloaded.csv
```

## 第二轮神经网络与融合

第二轮保留首轮冻结结果，依次训练 MLP、FT-Transformer 和 TabM，使用相同开发集三折。只在训练折内部早停，再重训整个训练折；连续特征的分位数变换、类别词表和 TabM 分箱都仅从对应训练数据拟合。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_neural.ps1
.\.venv\Scripts\python.exe scripts/run_neural.py benchmark
.\.venv\Scripts\python.exe scripts/run_neural.py train --model mlp
.\.venv\Scripts\python.exe scripts/run_neural.py train --model ft_transformer
.\.venv\Scripts\python.exe scripts/run_neural.py train --model tabm
.\.venv\Scripts\python.exe scripts/run_neural.py blend
.\.venv\Scripts\python.exe scripts/run_neural.py finalize
.\.venv\Scripts\python.exe scripts/run_neural.py verify
.\.venv\Scripts\python.exe scripts/run_neural.py report
```

`benchmark` 在完整内部训练折上实测三个 epoch，舍弃第一轮计时热身；按最大轮数、三折早停与重训、全量最终模型，再加 25% 余量，估算整套时间。按用户最新要求，全部训练固定使用本机 RTX 4060，不使用远程服务器。配置中的 60 分钟整套预算和 45 分钟单模型预算仅用于对照估计，超过预算仍在本机继续；训练支持按已完成折恢复。

配置见 `configs/neural_round2.json`；训练使用 BF16 混合精度，分类预测使用 FP32。TabM 采用官方库，成员先分别转换为概率再平均；FT-Transformer 采用官方推荐实现。第二轮结果和逐轮记录写入 `reports/round2/`，模型与 OOF 写入 `artifacts/round2/`，提交文件为 `submissions/round2_best.csv`。

`reports/round2/all_models_comparison.csv` 汇总首轮传统分类器、树提升、三种网络和融合的相同开发集 OOF 指标；`model_comparison.csv` 只展示本轮四路输入及所选融合。网络三折记录同时保存折间 AUC 标准差与实际耗时。

四路融合比较首轮树融合与三种网络，非负权重步长 0.1。只有 OOF AUC 至少提升 0.00005 且至少两折提升才替换原方案。第二轮分数用于开发集选型，旧独立测试集不反复用于调参或宣称新的独立测试成绩。

第二轮最终模型可重载推理：

```powershell
.\.venv\Scripts\python.exe scripts/predict.py --round round2 --input data/raw/test.csv --output submissions/round2_reloaded.csv
```

## 资料

- [比赛与数据](https://www.kaggle.com/competitions/playground-series-s6e11)
- [scikit-learn 模型评估](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [CatBoost GPU](https://catboost.ai/docs/en/features/training-on-gpu)
- [Optuna](https://optuna.readthedocs.io/)
- [FT-Transformer 官方实现](https://github.com/yandex-research/rtdl-revisiting-models)
- [TabM 官方实现](https://github.com/yandex-research/tabm)
