# 模型评估与错误分析报告

## 前言：可直接阅读presentation_selected/目录下的readme，这里是更详细的分析报告，包含了更多细节和数据表格。

## 1. 本模块做了什么

本次重新实现的 `failure_analysis.py` 会从团队已有结果中重新读取数据、重新计算指标、重新生成图表和报告。分析只深入比较两个代表模型：传统方法中表现最好的 `Traditional best`，以及深度学习中表现最好的 `TTA+Ensemble`。这样可以把篇幅集中在错误机制上，而不是罗列所有实验。

报告重点放在图像分析上：深度模型部分使用逐样本预测表定位错误图像；传统模型部分从最佳 SVM artifact 重新构建验证集逐图预测表，并结合分类报告、特征组、SVM 支持向量和特征消融解释为什么手工特征会失败。高置信错误定义为预测错误，并且模型对预测类别的最大 softmax probability >= 0.90。

## 2. 总体指标对比

| model | family | accuracy | balanced_accuracy | precision_macro | recall_macro | f1_macro |
| --- | --- | --- | --- | --- | --- | --- |
| Traditional best | Traditional | 0.7194 | 0.2977 | 0.3523 | 0.2977 | 0.3167 |
| TTA+Ensemble | Deep | 0.9261 | 0.9143 | 0.8938 | 0.9143 | 0.9016 |

`TTA+Ensemble` 相比传统最佳模型 Accuracy 提升 0.2067，Balanced Accuracy 提升 0.6166，Macro F1 提升 0.5849。传统模型 Accuracy 达到 0.7194，但 Balanced Accuracy 只有 0.2977，说明它主要依赖多数类 `nv`；深度模型 Balanced Accuracy 达到 0.9143，说明少数类也被明显改善。

对应图表：`figures/01_overall_metrics.png`、`figures/02_per_class_recall_compare.png`、`figures/03_per_class_f1_compare.png`。

## 3. 传统模型深入分析

传统方法扫描结果中，按加权 F1/Accuracy 排序的最佳实验为 `artifacts_best_practice\A_single_stage_control`。本报告对其进行深入分析。传统模型的主要问题不是完全不能识别图像，而是把图像压缩成 LBP、GLCM、HSV、ABCD 等手工特征后，很多空间细节被丢失。

本次已使用传统方法代码中的 `run_config.json`、`model.joblib`、`features.py` 重新生成验证集逐图预测表，共 2003 张图。当前工作区没有原始 `HAM10000/` 目录，因此验证集由 `preprocessing/*/metadata.csv` 与 origin 图像按同一 `random_state` 和 `stratified_ratio` 重建；重新计算指标与原 artifact 中 `metrics.json` 的关系为：基本对齐，但非逐位一致。校验细节见 `reports/traditional_prediction_validation.json`。

复现差异的主要原因是当前只保留了预处理目录下的 metadata/图像组织，而不是传统模型训练时使用的完整原始 `HAM10000/` 数据目录；此外，当前环境读取 `model.joblib` 时 sklearn 版本与训练版本不同。因此逐图预测表可用于错误分析，但总体指标对比仍以原 artifact 的 `metrics.json` 和 `classification_report.csv` 为准。

| metric | saved | recomputed | abs_diff |
| --- | --- | --- | --- |
| accuracy | 0.7194 | 0.7244 | 0.0050 |
| balanced_accuracy | 0.2977 | 0.3043 | 0.0066 |
| precision_macro | 0.3523 | 0.3533 | 0.0010 |
| recall_macro | 0.2977 | 0.3043 | 0.0066 |
| f1_macro | 0.3167 | 0.3212 | 0.0045 |

### 3.1 逐类别失败

| class | precision | recall | f1-score | support | missed | error_rate |
| --- | --- | --- | --- | --- | --- | --- |
| akiec | 0.4318 | 0.2923 | 0.3486 | 65.0000 | 46.0000 | 0.7077 |
| bcc | 0.3559 | 0.2039 | 0.2593 | 103.0000 | 82.0000 | 0.7961 |
| bkl | 0.4213 | 0.3409 | 0.3769 | 220.0000 | 145.0000 | 0.6591 |
| df | 0.0000 | 0.0000 | 0.0000 | 23.0000 | 23.0000 | 1.0000 |
| mel | 0.4570 | 0.3094 | 0.3690 | 223.0000 | 154.0000 | 0.6906 |
| nv | 0.8001 | 0.9374 | 0.8633 | 1341.0000 | 84.0000 | 0.0626 |
| vasc | 0.0000 | 0.0000 | 0.0000 | 28.0000 | 28.0000 | 1.0000 |

传统模型对 `nv` 的 recall 很高，但 `df` 和 `vasc` 的 recall 为 0，`mel` recall 也只有 0.3094。这说明模型更倾向于学习多数类边界，在小类和高风险类别上召回不足。尤其是 `mel`，漏判数量高、临床风险大，是传统模型最需要解释的失败类别之一。

### 3.2 传统模型逐图错误

| true_label | pred_label | count | pct_of_true | avg_decision_margin | avg_hair_ratio | risky_errors |
| --- | --- | --- | --- | --- | --- | --- |
| mel | nv | 106 | 0.4753 | 1.0587 | 0.0264 | 106 |
| bkl | nv | 92 | 0.4182 | 1.0436 | 0.0242 | 0 |
| bcc | nv | 53 | 0.5146 | 1.0398 | 0.0373 | 0 |
| nv | mel | 42 | 0.0313 | 0.9917 | 0.0248 | 0 |
| mel | bkl | 38 | 0.1704 | 0.9413 | 0.0287 | 38 |
| nv | bkl | 28 | 0.0209 | 1.0201 | 0.0208 | 0 |
| bkl | mel | 24 | 0.1091 | 0.9713 | 0.0237 | 0 |
| vasc | nv | 22 | 0.7857 | 1.0783 | 0.0323 | 0 |
| akiec | nv | 20 | 0.3077 | 1.1464 | 0.0290 | 0 |
| bcc | bkl | 16 | 0.1553 | 1.0872 | 0.0162 | 16 |
| df | nv | 15 | 0.6522 | 1.0388 | 0.0197 | 0 |
| akiec | bcc | 10 | 0.1538 | 1.1209 | 0.0248 | 0 |

传统模型逐图预测表见 `tables/traditional_predictions_reconstructed.csv`，错误样例见 `figures/14_traditional_error_gallery.jpg`。这些样例把传统模型失败落实到具体图像：它经常把 `mel/bkl/bcc/akiec/df/vasc` 等少数类推向 `nv`，这与逐类 recall 结果一致。传统模型的 `decision_margin` 来自 SVM decision function，只能表示分类边界距离，不等同于深度学习 softmax confidence。

| image_id | true_label | pred_label | decision_margin | hair_ratio | age | sex | localization |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ISIC_0029885 | mel | nv | 2.0970 | 0.0130 | 35.0000 | male | back |
| ISIC_0026566 | bkl | nv | 2.0570 | 0.0240 | 70.0000 | female | trunk |
| ISIC_0032384 | bcc | nv | 1.1318 | 0.0090 | 50.0000 | male | back |
| ISIC_0025247 | akiec | nv | 2.0734 | 0.0020 | 70.0000 | male | lower extremity |
| ISIC_0033005 | df | nv | 1.0813 | 0.0490 | 45.0000 | female | lower extremity |
| ISIC_0025707 | vasc | nv | 1.1839 | 0.0460 | 35.0000 | male | trunk |
| ISIC_0028968 | mel | bkl | 2.0557 | 0.0390 | 55.0000 | male | upper extremity |
| ISIC_0024971 | bkl | mel | 2.0376 | 0.0090 | 50.0000 | male | neck |

### 3.3 手工特征覆盖

| group | n_features |
| --- | --- |
| LBP_ROI | 256 |
| LBP_BIMF | 256 |
| HSV | 40 |
| ABCD | 4 |
| GLCM | 4 |

特征组说明传统模型确实利用了颜色、纹理和形态信息：HSV 对应颜色，LBP/GLCM 对应局部纹理，ABCD 对应皮损形态与边界。但是这些特征多是全局或局部摘要，很难保留病灶内部不同区域之间的空间关系。

`figures/05_traditional_feature_groups.png` 只说明传统最佳模型输入特征的组成和数量，不代表特征重要性。它显示 LBP 纹理特征数量最多，因此传统模型输入被纹理描述主导；真正的“哪些特征更重要”需要结合 `figures/09_traditional_evidence_panel.jpg` 中 ANN 权重和特征消融图解释。

### 3.4 可解释性证据

传统模型证据面板见 `figures/09_traditional_evidence_panel.jpg`。SVM 支持向量、ANN 权重和特征消融共同指向同一个结论：`mel/nv/bkl` 在手工特征空间内边界纠缠，而少数类需要大量边界样本才能被区分。

| class | n_support_vectors | sv_distribution_pct | n_train_samples | sv_ratio_vs_train_pct |
| --- | --- | --- | --- | --- |
| akiec | 1137 | 9.3596 | 262 | 433.9695 |
| bcc | 1765 | 14.5291 | 411 | 429.4404 |
| bkl | 2810 | 23.1314 | 879 | 319.6815 |
| df | 544 | 4.4781 | 92 | 591.3043 |
| mel | 2490 | 20.4972 | 890 | 279.7753 |
| nv | 2761 | 22.7280 | 5364 | 51.4728 |
| vasc | 641 | 5.2766 | 114 | 562.2807 |

`bkl`、`mel`、`nv` 的支持向量数量占比最高，说明这些类别经常落在分类边界附近。`df` 和 `vasc` 的支持向量/训练样本比例极高，说明小类边界不稳定。换句话说，传统模型不是没有诊断信息，而是手工特征空间无法把这些视觉相近类别稳定分开。

## 4. 深度模型深入分析

`TTA+Ensemble` 在验证集上共有 74 个错误，其中 confidence>=0.90 的高置信错误有 18 个，高风险错误方向有 29 个。深度模型虽然整体表现好，但错误集中在视觉相似类别，尤其是 `mel/nv/bkl`。

### 4.1 错误流向

| true_label | pred_label | count | pct_of_true | avg_confidence | avg_margin | high_conf_errors |
| --- | --- | --- | --- | --- | --- | --- |
| mel | nv | 24 | 0.2162 | 0.7132 | 0.4788 | 6 |
| nv | mel | 15 | 0.0224 | 0.7426 | 0.4970 | 3 |
| nv | bkl | 6 | 0.0089 | 0.6124 | 0.2532 | 0 |
| bkl | nv | 5 | 0.0455 | 0.9055 | 0.8153 | 3 |
| bkl | mel | 5 | 0.0455 | 0.7831 | 0.6017 | 0 |
| nv | vasc | 4 | 0.0060 | 0.7603 | 0.5649 | 1 |
| mel | bkl | 4 | 0.0360 | 0.6051 | 0.2943 | 0 |
| akiec | mel | 3 | 0.0909 | 0.8914 | 0.8253 | 2 |
| bkl | akiec | 2 | 0.0182 | 0.9952 | 0.9914 | 2 |
| akiec | bcc | 1 | 0.0303 | 0.9576 | 0.9204 | 1 |
| akiec | bkl | 1 | 0.0303 | 0.6638 | 0.3369 | 0 |
| bkl | bcc | 1 | 0.0091 | 0.6497 | 0.4065 | 0 |

最大错误方向是 `mel -> nv`，共 24 例，占真实 `mel` 的 21.62%。这是医学风险最高的方向，因为黑色素瘤被判断为良性色素痣。第二大错误方向是 `nv -> mel`，属于假阳性，反映出模型对深色、不规则或局部色素密集的普通痣较敏感。

### 4.2 错误图像案例

| image_id | true_label | pred_label | confidence | second_label | margin | hair_ratio | age | sex | localization |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISIC_0030107 | mel | nv | 0.9997 | mel | 0.9994 | 0.0730 | 20.0000 | female | scalp |
| ISIC_0033406 | nv | mel | 0.9820 | nv | 0.9693 | 0.0200 | 50.0000 | female | lower extremity |
| ISIC_0027991 | bkl | mel | 0.8815 | bkl | 0.7823 | 0.0140 | 55.0000 | male | back |
| ISIC_0029289 | bkl | nv | 0.9974 | mel | 0.9956 | 0.0070 | 60.0000 | female | trunk |
| ISIC_0025264 | akiec | mel | 0.9947 | nv | 0.9902 | 0.0410 | 75.0000 | male | back |
| ISIC_0028611 | bkl | akiec | 0.9996 | bkl | 0.9995 | 0.0040 | 80.0000 | female | lower extremity |
| ISIC_0027911 | nv | bkl | 0.7601 | nv | 0.5243 | 0.0140 | 50.0000 | male | lower extremity |
| ISIC_0026796 | mel | bkl | 0.8293 | mel | 0.6703 | 0.0030 | 70.0000 | male | back |
| ISIC_0029534 | nv | vasc | 0.9740 | nv | 0.9506 | 0.0100 | 65.0000 | male | face |
| ISIC_0031043 | akiec | bcc | 0.9576 | akiec | 0.9204 | 0.0040 | 85.0000 | male | face |

错误图像画廊见 `figures/07_deep_error_gallery.jpg`。这些案例显示，深度模型的失败通常与以下视觉因素相关：病灶面积小、边界过渡弱、棕黑色调与 `nv/bkl` 接近、局部纹理不典型，以及毛发或背景纹理干扰。为了避免只看图但不知道为什么错，本脚本还生成了逐图解释表 `tables/deep_error_case_notes.csv` 和可视化说明图 `figures/13_deep_error_case_notes.png`。

| image_id | true_label | pred_label | confidence | risk_level | visual_error_note |
| --- | --- | --- | --- | --- | --- |
| ISIC_0030107 | mel | nv | 0.9997 | high | 黑色素瘤被判为痣，通常说明病灶整体色调/形态接近良性色素痣；若边界过渡弱或局部恶性线索不突出，模型会偏向多数类 nv。 该图 hair_ratio 较高，毛发或线状背景纹理可能额外干扰边界和局部纹理判断。 病灶位于 scalp，该部位更容易受到毛发、光照或背景纹理影响。 |
| ISIC_0028611 | bkl | akiec | 0.9996 | medium-high | bkl 被判为 akiec，通常与红褐色、粗糙或角化样区域有关，两类都可能呈现表面角化/鳞屑样视觉线索。 |
| ISIC_0029289 | bkl | nv | 0.9974 | medium-high | bkl 被判为 nv，常见于病灶较小、颜色集中、角化纹理不明显的样本；模型更容易把它当作普通色素痣。 |
| ISIC_0034205 | mel | nv | 0.9959 | high | 黑色素瘤被判为痣，通常说明病灶整体色调/形态接近良性色素痣；若边界过渡弱或局部恶性线索不突出，模型会偏向多数类 nv。 该图 hair_ratio 较高，毛发或线状背景纹理可能额外干扰边界和局部纹理判断。 |
| ISIC_0033444 | mel | nv | 0.9954 | high | 黑色素瘤被判为痣，通常说明病灶整体色调/形态接近良性色素痣；若边界过渡弱或局部恶性线索不突出，模型会偏向多数类 nv。 |
| ISIC_0025264 | akiec | mel | 0.9947 | medium-high | akiec 被判为 mel，可能因为红褐色混合、局部不均匀和边界复杂，被模型解释为恶性黑色素瘤线索。 |
| ISIC_0029991 | bkl | akiec | 0.9908 | medium-high | bkl 被判为 akiec，通常与红褐色、粗糙或角化样区域有关，两类都可能呈现表面角化/鳞屑样视觉线索。 |
| ISIC_0031191 | akiec | mel | 0.9845 | medium-high | akiec 被判为 mel，可能因为红褐色混合、局部不均匀和边界复杂，被模型解释为恶性黑色素瘤线索。 |

### 4.3 高置信错误

| image_id | true_label | pred_label | confidence | second_label | margin | hair_ratio | age | sex | localization |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISIC_0030107 | mel | nv | 0.9997 | mel | 0.9994 | 0.0730 | 20.0000 | female | scalp |
| ISIC_0028611 | bkl | akiec | 0.9996 | bkl | 0.9995 | 0.0040 | 80.0000 | female | lower extremity |
| ISIC_0029289 | bkl | nv | 0.9974 | mel | 0.9956 | 0.0070 | 60.0000 | female | trunk |
| ISIC_0034205 | mel | nv | 0.9959 | mel | 0.9923 | 0.0500 | 55.0000 | female | upper extremity |
| ISIC_0033444 | mel | nv | 0.9954 | mel | 0.9907 | 0.0350 | 50.0000 | male | abdomen |
| ISIC_0025264 | akiec | mel | 0.9947 | nv | 0.9902 | 0.0410 | 75.0000 | male | back |
| ISIC_0029991 | bkl | akiec | 0.9908 | bkl | 0.9833 | 0.0370 | 75.0000 | male | back |
| ISIC_0031191 | akiec | mel | 0.9845 | nv | 0.9750 | 0.0260 | 60.0000 | male | lower extremity |
| ISIC_0028173 | mel | nv | 0.9827 | bcc | 0.9704 | 0.0140 | 45.0000 | male | back |
| ISIC_0033406 | nv | mel | 0.9820 | nv | 0.9693 | 0.0200 | 50.0000 | female | lower extremity |
| ISIC_0025366 | bkl | nv | 0.9761 | bkl | 0.9524 | 0.0560 | 50.0000 | female | trunk |
| ISIC_0029534 | nv | vasc | 0.9740 | nv | 0.9506 | 0.0100 | 65.0000 | male | face |

高置信错误说明 softmax confidence 不能直接等同于医学可靠性。本报告将 confidence >= 0.90 且预测错误的样本定义为高置信错误。例如 `ISIC_0030107` 是 `mel -> nv` 且 confidence 接近 1，这类样本在真实应用中必须进入人工复核或模型校准流程。所有高置信错误的原图和预处理后图像已导出到 `images/high_confidence_errors/`，索引见 `tables/high_confidence_error_image_index.csv`。

### 4.4 Attention/Grad-CAM 证据

深度可解释性画廊见 `figures/08_deep_interpretability_gallery.jpg`。正确样本中，热力图多集中在病灶主体；错误样本中，热力图也常覆盖病灶区域。这说明很多错误不是因为模型完全看错位置，而是因为病灶内部的颜色、纹理、边界特征本身与其他类别高度重叠。

## 5. 图像质量与元数据交叉分析

| true_label | n | error_rate | avg_hair_ratio | invalid_rate | high_conf_error_rate | risky_error_count |
| --- | --- | --- | --- | --- | --- | --- |
| mel | 111 | 0.2613 | 0.0274 | 0.0000 | 0.0541 | 28 |
| akiec | 33 | 0.1818 | 0.0247 | 0.0000 | 0.0909 | 1 |
| bkl | 110 | 0.1182 | 0.0248 | 0.0000 | 0.0455 | 0 |
| nv | 671 | 0.0387 | 0.0467 | 0.0075 | 0.0060 | 0 |
| bcc | 51 | 0.0000 | 0.0385 | 0.0196 | 0.0000 | 0 |
| df | 12 | 0.0000 | 0.0408 | 0.0000 | 0.0000 | 0 |
| vasc | 14 | 0.0000 | 0.0319 | 0.0000 | 0.0000 | 0 |

`hair_ratio` 是预处理阶段估计的毛发/线状遮挡比例，`hair_bin` 是把它离散成 `0-1%`、`1-5%`、`5-10%`、`10-20%`、`>20%` 后得到的分组。这里的错误率并没有随 `hair_bin` 单调升高，甚至高毛发组错误率更低。这个现象不能解释为“毛发越多越容易分类”，更合理的解释是：预处理已经有效去除了相当多毛发；高 hair_ratio 组样本量较小；不同 hair_bin 的类别组成不同，例如大量容易识别的 `nv` 可能落在高 hair_ratio 组。因此 hair_bin 只能作为图像质量辅助线索，不能单独作为错误原因。

`localization` 是 HAM10000 元数据中的病灶身体部位，例如 back、face、lower extremity、scalp、trunk 等。按 localization 看错误率，目的是检查模型是否在某些身体部位更容易失败，或是否存在类别/部位分布偏差。元数据分组图见 `figures/10_error_rate_by_hair_bin.png`、`figures/11_error_rate_by_localization.png`。

## 6. 传统模型与深度模型错误重叠

| overlap_group | n | pct_of_common | avg_deep_confidence |
| --- | --- | --- | --- |
| both_correct | 677 | 0.6756 | 0.9647 |
| traditional_wrong_deep_correct | 251 | 0.2505 | 0.9025 |
| traditional_correct_deep_wrong | 32 | 0.0319 | 0.7420 |
| both_wrong | 42 | 0.0419 | 0.7325 |

重叠分析只在两个模型共同出现的 image_id 上进行，结果见 `tables/traditional_deep_error_overlap.csv` 和 `figures/15_traditional_deep_error_overlap.png`。如果样本落在 `traditional_wrong_deep_correct`，说明深度模型确实修正了传统手工特征的失败；如果落在 `both_wrong`，说明该图可能属于更困难的视觉相似或图像质量问题，需要结合原图、预处理图和可解释性热力图进一步检查。

### 6.1 重点关注的两组 overlap 错误

| overlap_group | true_label | pred_label_traditional | pred_label_deep | count | avg_deep_confidence | avg_hair_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| both_wrong | mel | nv | nv | 13 | 0.7432 | 0.0321 |
| both_wrong | mel | bkl | nv | 5 | 0.6289 | 0.0182 |
| both_wrong | bkl | nv | nv | 4 | 0.9279 | 0.0232 |
| both_wrong | bkl | nv | mel | 3 | 0.7889 | 0.0890 |
| both_wrong | mel | bkl | bkl | 3 | 0.6438 | 0.0167 |
| both_wrong | nv | bkl | bkl | 3 | 0.5352 | 0.0113 |
| both_wrong | akiec | nv | mel | 2 | 0.8397 | 0.0230 |
| both_wrong | mel | akiec | nv | 2 | 0.6525 | 0.0370 |
| both_wrong | bkl | nv | akiec | 1 | 0.9996 | 0.0040 |
| both_wrong | akiec | mel | mel | 1 | 0.9947 | 0.0410 |
| both_wrong | bkl | akiec | nv | 1 | 0.8159 | 0.0010 |
| both_wrong | bkl | akiec | mel | 1 | 0.6676 | 0.0130 |
| both_wrong | akiec | nv | bkl | 1 | 0.6638 | 0.0410 |
| both_wrong | akiec | nv | nv | 1 | 0.6479 | 0.0340 |
| both_wrong | nv | bkl | mel | 1 | 0.5705 | 0.0100 |
| traditional_correct_deep_wrong | nv | nv | mel | 14 | 0.7549 | 0.0239 |

`traditional_correct_deep_wrong` 是深度模型相对传统模型退步的样本，已导出到 `images/traditional_correct_deep_wrong/`，索引见 `tables/overlap_traditional_correct_deep_wrong_image_index.csv`，画廊见 `figures/16_overlap_traditional_correct_deep_wrong_gallery.jpg`。`both_wrong` 是两个模型都失败的困难样本，已导出到 `images/both_wrong/`，索引见 `tables/overlap_both_wrong_image_index.csv`，画廊见 `figures/17_overlap_both_wrong_gallery.jpg`。

### 6.2 两组 overlap 的错误原因

| image_id | overlap_group | true_label | traditional_pred | deep_pred | deep_confidence | reason_category | review_priority | error_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISIC_0030107 | both_wrong | mel | nv | nv | 0.9997 | shared_mel_to_nv_high_risk | high | 两个模型都把 mel 判成 nv，是最高风险方向；这通常说明病灶整体外观接近普通痣，边界/颜色不均等恶性线索不够突出。 深度模型置信度很高，提示这是需要重点复核的过度自信错误。 hair_ratio 较高，毛发或线状纹理可能干扰边界和局部纹理判断。 localization=scalp，该部位更容易出现光照、毛发、角质或背景纹理干扰。 |
| ISIC_0028611 | both_wrong | bkl | nv | akiec | 0.9996 | shared_bkl_boundary_case | medium-high | bkl 同时难倒传统和深度模型，说明其颜色集中、角化线索弱或纹理与 nv/mel/akiec 边界重叠。 深度模型置信度很高，提示这是需要重点复核的过度自信错误。 |
| ISIC_0029289 | both_wrong | bkl | nv | nv | 0.9974 | shared_bkl_boundary_case | medium-high | bkl 同时难倒传统和深度模型，说明其颜色集中、角化线索弱或纹理与 nv/mel/akiec 边界重叠。 深度模型置信度很高，提示这是需要重点复核的过度自信错误。 |
| ISIC_0033444 | both_wrong | mel | bkl | nv | 0.9954 | shared_mel_to_nv_high_risk | high | 两个模型都把 mel 判成 nv，是最高风险方向；这通常说明病灶整体外观接近普通痣，边界/颜色不均等恶性线索不够突出。 深度模型置信度很高，提示这是需要重点复核的过度自信错误。 |
| ISIC_0025264 | both_wrong | akiec | mel | mel | 0.9947 | shared_akiec_boundary_case | medium-high | akiec 少数类样本同时失败，常见原因是红褐色、鳞屑/角化样纹理与 mel、bcc、bkl 或 nv 的局部表现相似。 深度模型置信度很高，提示这是需要重点复核的过度自信错误。 |
| ISIC_0031191 | both_wrong | akiec | nv | mel | 0.9845 | shared_akiec_boundary_case | medium-high | akiec 少数类样本同时失败，常见原因是红褐色、鳞屑/角化样纹理与 mel、bcc、bkl 或 nv 的局部表现相似。 深度模型置信度很高，提示这是需要重点复核的过度自信错误。 |
| ISIC_0028173 | both_wrong | mel | nv | nv | 0.9827 | shared_mel_to_nv_high_risk | high | 两个模型都把 mel 判成 nv，是最高风险方向；这通常说明病灶整体外观接近普通痣，边界/颜色不均等恶性线索不够突出。 深度模型置信度很高，提示这是需要重点复核的过度自信错误。 |
| ISIC_0025366 | both_wrong | bkl | nv | nv | 0.9761 | shared_bkl_boundary_case | medium-high | bkl 同时难倒传统和深度模型，说明其颜色集中、角化线索弱或纹理与 nv/mel/akiec 边界重叠。 深度模型置信度很高，提示这是需要重点复核的过度自信错误。 hair_ratio 较高，毛发或线状纹理可能干扰边界和局部纹理判断。 |
| ISIC_0028760 | both_wrong | mel | nv | nv | 0.9482 | shared_mel_to_nv_high_risk | high | 两个模型都把 mel 判成 nv，是最高风险方向；这通常说明病灶整体外观接近普通痣，边界/颜色不均等恶性线索不够突出。 深度模型置信度很高，提示这是需要重点复核的过度自信错误。 |
| ISIC_0032675 | both_wrong | bkl | nv | nv | 0.9051 | shared_bkl_boundary_case | medium-high | bkl 同时难倒传统和深度模型，说明其颜色集中、角化线索弱或纹理与 nv/mel/akiec 边界重叠。 深度模型置信度很高，提示这是需要重点复核的过度自信错误。 |

`traditional_correct_deep_wrong` 主要说明深度模型存在局部过度敏感：一些 `nv` 被深度模型误报为 `mel/vasc/bkl/bcc`，以及少量 `mel/bkl` 被推向相邻类别。`both_wrong` 更像真正的困难样本：最多的是 `mel -> nv` 和 `mel -> bkl`，说明这些图像的恶性线索弱、颜色/纹理更接近良性痣或 bkl；这类样本应优先用于人工复核、后续模型校准或针对性数据增强。逐样本原因表见 `tables/overlap_case_notes.csv`，说明图见 `figures/18_overlap_reason_panel.jpg`。

## 7. 传统模型与深度模型的对等结论

| class | recall_traditional | recall_deep | recall_gain | f1_traditional | f1_deep | f1_gain |
| --- | --- | --- | --- | --- | --- | --- |
| df | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| vasc | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.8750 | 0.8750 |
| bcc | 0.2039 | 1.0000 | 0.7961 | 0.2593 | 0.9714 | 0.7122 |
| bkl | 0.3409 | 0.8818 | 0.5409 | 0.3769 | 0.8899 | 0.5130 |
| akiec | 0.2923 | 0.8182 | 0.5259 | 0.3486 | 0.8571 | 0.5085 |
| mel | 0.3094 | 0.7387 | 0.4293 | 0.3690 | 0.7593 | 0.3903 |
| nv | 0.9374 | 0.9613 | 0.0239 | 0.8633 | 0.9584 | 0.0951 |

传统模型的失败重点是表达能力不足：手工特征能描述颜色、纹理和形态，但难以保留细粒度空间关系，因此在 `mel/nv/bkl` 和少数类上边界不稳定。深度模型的失败重点是高相似类别的细粒度误判：它能看到病灶，也能显著改善少数类，但仍会在黑色素瘤、普通痣和脂溢性角化病之间产生高风险、高置信错误。

最终建议表述：传统方法提供了可解释的医学特征基线，但在 HAM10000 的不平衡和视觉相似条件下不足以稳定区分关键类别；深度学习通过多层视觉特征、TTA 和 ensemble 显著提升 Macro F1 与 Balanced Accuracy，但仍需要结合错误图像案例、可解释性热力图、图像质量控制、模型校准和人工复核来降低 `mel -> nv` 等高风险错误。
