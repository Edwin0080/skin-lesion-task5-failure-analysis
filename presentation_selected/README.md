# 任务 5：评估指标与错误分析展示素材说明

本文件夹只保留最终 PPT 三页实际使用或可能被追溯的数据与图像。任务 5 的目标是把传统方法与深度学习方法的最终结果放在同一套评估框架下比较，并进一步解释错误来自哪里、哪些错误最值得关注。

## 目录结构

```text
presentation_selected/
├── 01_slide_metrics/          # 第 1 页：总体指标与类别表现
├── 02_slide_multidim/         # 第 2 页：部位错误率与模型错误重叠
├── 03_slide_error_examples/   # 第 3 页：具体错误样例
├── tables/                    # PPT 图表背后的关键表格
└── scripts/                   # 生成分析结果和可编辑错误样例页的脚本
```

## 第 1 页：评估指标与错误分析

这一页展示“整体指标 + 混淆矩阵 + 类别召回率”，用于说明传统模型和深度学习模型的总体差距。

使用文件：

- `01_slide_metrics/traditional_confusion_matrix_A_single_stage_control.png`
  - 来源：`传统方法/outputs/artifacts_best_practice/A_single_stage_control/confusion_matrix.png`
  - 含义：传统最佳模型 `A_single_stage_control` 的混淆矩阵。
  - 主要结论：传统模型明显偏向 `nv`，很多 `mel/bkl/bcc/df/vasc` 等少数类会被推向 `nv`。

- `01_slide_metrics/deep_confusion_matrix_tta_ensemble.png`
  - 来源：`failure_analysis_outputs/figures/12_deep_confusion_matrix.png`
  - 含义：深度学习最优方案 `TTA+Ensemble` 的归一化混淆矩阵。
  - 主要结论：深度模型整体更接近对角线，但 `mel -> nv` 仍是最值得关注的临床风险方向。

- `01_slide_metrics/01_overall_metrics.png`
  - 展示 Accuracy、Balanced Accuracy、Macro F1。
  - 关键数值：
    - 传统模型：Accuracy 0.7194，Balanced Accuracy 0.2977，Macro F1 0.3167。
    - 深度模型：Accuracy 0.9261，Balanced Accuracy 0.9143，Macro F1 0.9016。
  - 解释重点：传统模型的 Accuracy 受 `nv` 大类占比影响较大；Balanced Accuracy 和 Macro F1 更能反映少数类表现。

- `01_slide_metrics/02_per_class_recall_compare.png`
  - 展示每个类别的召回率。
  - 解释重点：传统模型 `nv` 召回率较高，但 `df/vasc/mel` 等少数类召回率低；深度模型各类召回更均衡。

## 第 2 页：多维度定量评估指标

这一页展示“不同身体部位错误率 + 两模型错误重叠”，用于说明错误来源。

使用文件：

- `02_slide_multidim/21_error_rate_by_localization_compare.png`
  - 含义：传统模型与深度模型按 `localization` 分组的错误率对比。
  - 主要结论：传统模型在大多数部位的错误率显著高于深度模型，尤其在 `face/ear/scalp/chest` 等部位差距较大。
  - 解释重点：这些部位更容易受到光照、背景纹理、毛发、局部颜色变化影响，传统手工特征对这些变化的鲁棒性不足。

- `02_slide_multidim/15_traditional_deep_error_overlap.png`
  - 含义：在两个模型共同出现的 1002 张图像上，比较二者预测对错的重叠情况。
  - 关键数值：
    - `both_correct`: 677
    - `traditional_wrong_deep_correct`: 251
    - `traditional_correct_deep_wrong`: 32
    - `both_wrong`: 42
  - 解释重点：深度模型修正了大量传统模型错误，但仍存在少量深度回退样本和两个模型都失败的困难样本。

PPT 中重点解释的两类 overlap：

- `traditional_correct_deep_wrong`
  - 32 例。
  - 含义：传统模型正确、深度模型错误。
  - 解释：深度模型可能对局部颜色或结构线索过度敏感，出现 `nv -> mel/vasc/bkl/bcc` 等回退。

- `both_wrong`
  - 42 例。
  - 含义：传统模型和深度模型都预测错误。
  - 解释：多为真正困难样本，尤其是 `mel -> nv`、`mel -> bkl`，说明恶性线索弱且与良性类别视觉重叠。

## 第 3 页：错误分析

这一页用 4 个具体图像样例说明错误类型。

使用文件：

- `03_slide_error_examples/任务5_错误样例_可编辑.pptx`
  - 四宫格可编辑错误样例页。
  - 每个案例包含原图、预处理图、真实标签、传统预测、深度预测、置信度、部位和错误原因。

- `03_slide_error_examples/case_images/`
  - 保存 4 个案例的原图和预处理图，方便替换或重新排版。

4 个展示案例：

1. `ISIC_0032046`
   - 类型：传统错、深度对。
   - 真实标签：`mel`
   - 传统预测：`nv`
   - 深度预测：`mel`
   - 说明：传统手工特征把黑色素瘤推向多数类 `nv`，体现 `mel/nv` 视觉相似时传统方法的表达能力不足。

2. `ISIC_0028611`
   - 类型：高置信错误。
   - 真实标签：`bkl`
   - 传统预测：`nv`
   - 深度预测：`akiec`
   - 深度置信度：0.9996
   - 说明：softmax 高置信不等于医学可靠，需要错误样本复核和模型校准。

3. `ISIC_0034205`
   - 类型：传统对、深度错。
   - 真实标签：`mel`
   - 传统预测：`mel`
   - 深度预测：`nv`
   - 深度置信度：0.9959
   - 说明：这是深度模型相对传统模型的回退，提示深度模型仍可能漏掉少数 `mel` 样本。

4. `ISIC_0030107`
   - 类型：两个模型都错。
   - 真实标签：`mel`
   - 传统预测：`nv`
   - 深度预测：`nv`
   - 深度置信度：0.9997
   - 说明：两个模型都把 `mel` 判为 `nv`，属于共同困难且临床风险较高的样本，应优先人工复核。

## 表格说明

`tables/` 中保留了 PPT 图表背后的关键数据：

- `model_summary_best_traditional_vs_deep.csv`
  - 两个最佳模型的 Accuracy、Balanced Accuracy、Macro F1 等总体指标。

- `traditional_best_per_class_metrics.csv`
  - 传统最佳模型逐类别 precision、recall、F1。

- `deep_best_per_class_metrics.csv`
  - 深度最佳模型逐类别 precision、recall、F1。

- `localization_error_rate_compare.csv`
  - 第 2 页 localization 错误率对比图的数据来源。

- `traditional_deep_error_overlap.csv`
  - 第 2 页 overlap 柱状图的数据来源。

- `traditional_deep_error_overlap_detail.csv`
  - 每张共同图像的真实标签、传统预测、深度预测、置信度和 overlap 分组。

- `overlap_focus_summary.csv`
  - 对 `traditional_correct_deep_wrong` 和 `both_wrong` 两类重点错误的汇总。

- `deep_high_confidence_errors.csv`
  - 深度模型高置信错误样本。这里高置信定义为预测错误且 `confidence >= 0.90`。

## 脚本说明

`scripts/` 中保留了本部分使用的主要脚本副本：

- `failure_analysis.py`
  - 负责读取传统模型与深度模型结果，生成指标表、错误重叠分析、localization 错误率对比、错误样例等。
  - 传统模型使用 `传统方法/outputs/artifacts_best_practice/A_single_stage_control`。
  - 深度模型使用 `深度学习方法/experiments/ensemble_line1_line2_stepA/tta_ensemble_alpha_0.50_predictions.csv`。

- `create_error_examples_slide_pptx.py`
  - 负责生成第 3 页使用的可编辑错误样例 PPTX。

如果需要复现完整分析，在项目根目录运行：

```powershell
python failure_analysis.py
```

注意：运行时可能出现 sklearn 版本不一致 warning，这是因为传统模型的 `model.joblib` 保存时 sklearn 版本与当前环境不同。当前重建指标与原传统模型结果误差在 0.01 内，因此用于错误分析是可接受的。

## 本部分最终结论

任务 5 的核心结论可以概括为三点：

1. 传统模型的 Accuracy 受 `nv` 大类影响较大；Balanced Accuracy 和 Macro F1 更能说明深度模型在少数类上的优势。
2. 深度学习模型整体显著优于传统手工特征模型，但 `mel -> nv` 仍是最需要关注的临床风险方向。
3. 图像级错误分析显示，失败主要来自 `mel/nv/bkl` 的视觉重叠、部位相关图像质量差异，以及少量高置信错误；这些样本需要人工复核、模型校准和针对性数据增强。
