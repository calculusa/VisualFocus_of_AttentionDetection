# VFOA 实验记录：YOLO26s、ResNet18 与 GazeTR

更新日期：2026-09-14。记录截至目前已完成的实验，供后续工作与论文写作使用。

## 1. 实验目标与当前阶段

本研究比较三条判断“是否看摄像头”的流程：

1. **直接检测：** YOLO26s 从原始图片同时预测人脸位置和注意力类别。
2. **人脸外观分类：** 从图片中取得人脸，再用 ResNet18 判断类别。
3. **基于 gaze 的分类：** 从图片中取得人脸，用 GazeTR 预测 pitch、yaw，再用合成角度阈值判断看／不看。

当前先用人工标注框（GT boxes）为后两条流程提供人脸，观察分类环节的表现。后续再接入实际人脸检测框，评估完整系统中的漏检、误检及裁剪误差。

**GazeTR 后处理继续采用已经运行的合成角度公式法。下一步检查分数的区分能力和阈值稳定性。**

| 实验 | 当前输入 | 输出 | 已完成的评估 |
|---|---|---|---|
| YOLO26s | 原始图片 | 人脸框＋三类标签 | 三分类检测 |
| ResNet18 | GT 人脸裁剪 | 三类标签 | 三分类验证 |
| GazeTR＋角度阈值 | GT 人脸裁剪 | 看／不看 | 两个确定类别的二分类验证 |

这三项结果回答的问题不同，暂时作为阶段性基线记录，不直接据此给三条完整流程排名。

## 2. 共用数据与人脸准备

### 2.1 数据划分

使用重新标注的 WIDER FACE 子集，共 613 张原始图片：train 543 张、val 70 张。此前约 10,000 张图片的二分类实验属于另一套设置，不混入本轮结果。

| ID | 类别 | Train 人脸数 | Val 人脸数 |
|---|---|---:|---:|
| 0 | not_looking_at_camera | 363 | 51 |
| 1 | looking_at_camera | 1,037 | 135 |
| 2 | uncertain | 280 | 32 |
| 合计 | 全部三类 | 1,680 | 218 |
| 二分类子集 | 仅真实标签 0、1 | 1,400 | 186 |

一张原图可能包含多个人脸。数据划分应在原图层面完成，同一原图中的人脸保留在同一 split。

当前所有结果均为 **val** 结果；尚无已确认的独立 test 结果。此前裁剪报错中的 `images/test` 为空，因此本轮仅处理 train、val。

原始数据目录：

```text
/home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/Dataset/widerface_yolo26_subset
```

人脸裁剪目录：

```text
/home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/Dataset/face_crop_3c
```

### 2.2 裁剪方法与命令

`crop_vfoa_faces.py` 读取已有 YOLO 人脸框，换算像素坐标后裁剪，按照原 split 和类别保存。不需要先重新运行人脸检测器。

提供脚本的默认设置为 margin=0，不统一 resize、不额外按人脸大小过滤；各模型在加载时再调整尺寸。实际裁剪设置以已保存的 `summary.json` 为准。

在脚本所在目录运行：

```bash
python crop_vfoa_faces.py \
  --dataset "/home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/Dataset/widerface_yolo26_subset" \
  --output "/home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/Dataset/face_crop_3c" \
  --splits train val
```

输出包括分类目录下的人脸图片，以及 `manifest.csv`、`summary.json`。其中 manifest 记录人脸与原图的对应关系，后续分组评估需要保留。

这一步已经完成，明天无需重跑。此前遇到的两个问题及处理：

- `classes.txt` 被误当作逐图标签：在标签匹配检查中排除该类别元数据文件。
- 空 test 目录导致报错：指定 `--splits train val`。

## 3. 实验一：YOLO26s 三分类检测

### 3.1 实现方法

YOLO26s 直接处理原图，联合输出人脸框与 `not_looking_at_camera`、`looking_at_camera`、`uncertain` 三类标签。检测指标同时受定位、漏检、误检和类别判断影响。

已确认使用 YOLO26s、imgsz=640，训练记录包含 100 epochs。数据 YAML、优化器、batch、设备和其他设置应从该轮 `train.py`、`args.yaml` 与日志归档，不沿用其他运行的配置。

### 3.2 实际运行命令

项目目录：

```bash
cd /home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/yolo26
```

用户确认的实际启动命令：

```bash
setsid nohup python train.py > yolo26s_resume_subset_3c_14092026.log 2>&1 < /dev/null &
```

查看日志：

```bash
tail -f yolo26s_resume_subset_3c_14092026.log
```

模型和训练参数由 `train.py` 配置。日志名中的 `resume` 不能单独证明本次启用了恢复训练；实际行为以脚本和日志为准。

结果目录：

```text
/home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/yolo26/runs/detect/runs/vfoa_yolo26_subset_3c/yolo26s_img640_ep100_subset_3c
```

### 3.3 验证结果

最终终端报告：70 张原图，218 个人脸。

| Class | Precision | Recall | mAP50 | mAP50–95 | Instances |
|---|---:|---:|---:|---:|---:|
| Overall | 0.515 | 0.696 | 0.591 | 0.448 | 218 |
| Not looking at camera | 0.403 | 0.490 | 0.487 | 0.346 | 51 |
| Looking at camera | 0.731 | 0.911 | 0.831 | 0.632 | 135 |
| Uncertain | 0.413 | 0.688 | 0.454 | 0.365 | 32 |

终端报告每张图片平均耗时：preprocess 0.4 ms、inference 0.6 ms、postprocess 0.1 ms。尚未统一各方法的测速条件，因此暂不进行速度排名。

上传的训练 `results(1).csv` 中：

| 训练记录 | Epoch | mAP50 | mAP50–95 |
|---|---:|---:|---:|
| mAP50 最高 | 36 | 0.62144 | 0.44457 |
| mAP50–95 最高 | 40 | 0.59003 | 0.44734 |
| 最后一轮 | 100 | 0.54420 | 0.39709 |

训练 CSV 的逐轮结果与最终终端验证结果分开记录，不拼接不同 epoch 的最高指标。最终加载权重的精确 epoch 仍需结合权重元数据核对。

### 3.4 结果解释

“看摄像头”的 AP 明显高于另外两类。训练后期，训练损失继续下降而验证 mAP 回落，提示继续训练没有带来同步的泛化改善。

该结果反映完整的三分类检测表现。仅凭这些类别指标，无法判断某一类的困难主要来自漏检、框定位还是分类错误，后续需要结合检测匹配结果分析。

## 4. 实验二：ResNet18 人脸外观三分类

### 4.1 实现方法

使用相同 split 的 GT 人脸裁剪，训练 ResNet18 输出三类标签。代码分为 `model.py`、`train.py` 和 `config.yaml`，网络经过目标数据训练。

此前提供的实现方案采用 ImageNet 预训练 ResNet18、224×224 输入、RGB 与 ImageNet 均值／标准差归一化，使用三分类交叉熵和 AdamW。方案设置还包括 letterbox、水平翻转、lr=1e-4、weight_decay=1e-4、最多 40 epochs、patience=10，以 val macro-F1 选模，seed=42。

这些详细参数来自此前提供的方案，尚未与服务器最终配置逐项复核。已确认终端选中 checkpoint epoch=6；论文定稿时以实际保存配置为准。

### 4.2 运行命令

项目目录和此前使用的脚本接口：

```bash
cd /home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/vfoa_crop
python -u train.py --config config.yaml
```

此前讨论的后台运行方式：

```bash
nohup setsid python -u train.py --config config.yaml \
  > resnet18_3c_seed42.log 2>&1 < /dev/null &
```

查看日志：

```bash
tail -f resnet18_3c_seed42.log
```

结果目录与选中模型：

```text
/home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/vfoa_crop/runs/resnet18_3c_seed42
/home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/vfoa_crop/runs/resnet18_3c_seed42/best.pt
```

### 4.3 验证结果

三分类验证，checkpoint epoch=6。

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Overall (macro) | 0.677 | 0.725 | 0.693 |
| Not looking at camera | 0.673 | 0.725 | 0.698 |
| Looking at camera | 0.880 | 0.763 | 0.817 |
| Uncertain | 0.478 | 0.688 | 0.564 |

**Accuracy：0.7431（74.31%）。** 按共同 val 划分对应 218 张人脸；进行配对比较前仍需核对预测文件中的人脸 ID。

保留：`val_predictions.csv`、`val_metrics.csv`、`val_confusion_matrix.csv`、训练 `results.csv`、配置与 `best.pt`。本节指标依据用户提供的终端结果，尚未再次核对完整逐脸预测。

### 4.4 结果解释

在本轮三分类设置下，“看”的 F1 最高，uncertain 最低。该结果表明外观分类器能够学习标签区分，但当前评估没有包含人脸检测误差。

因此，这是分类环节的基线，不能直接把它的 F1 与 YOLO 检测 mAP 比较，也不能认定完整两阶段系统已经优于 YOLO。

## 5. 实验三：GazeTR＋合成角度阈值二分类

### 5.1 实现方法

使用 ETH 预训练权重 `GazeTR-H-ETH.pt`，固定网络参数。流程为：

1. 从 GT 人脸裁剪预测 pitch、yaw。
2. 将两个角度合成相对于零角度参考方向的分数。
3. 使用 train 的确定类别标签选择一个阈值。
4. 固定阈值，在 val 的确定类别上计算二分类指标。

这一步没有微调 GazeTR，但使用了目标训练集标签校准分类规则，因此不是完全零样本分类。

仓库与权重：

```text
/home/lunet/cowz2/Documents/GazeEstimation/GazeTR
/home/lunet/cowz2/Documents/GazeEstimation/GazeTR/GazeTR-H-ETH.pt
```

输入处理依据已核查的 `reader.py`：OpenCV 读取、保留通道顺序、CHW 浮点数除以 255，不使用 ImageNet Normalize。当前推理额外用 INTER_LINEAR 直接 resize 到 224×224。

此流程为普通人脸裁剪上的直接迁移，没有复现 ETH 上游的几何归一化。`data_processing_eth.py` 只导出已有的 face_patch，不展示它最初怎样生成。当前流程不要求运行 dlib。

### 5.2 判定公式与阈值选择

pitch、yaw 以弧度代入，输出分数以度表示：

```math
s = arccos(clip(cos(pitch) * cos(yaw), -1, 1)) * 180 / pi
```

分类规则：

```python
looking = score_deg <= threshold_deg
```

在 train 的 1,400 张确定人脸上，搜索 0°～180°、步长 0.5°，另包含 −1°作为全部判不看的选项。选择两类 macro-F1 最高的阈值；并列取最小值。

**本轮选中 37.5°，随后固定用于 val。** 280 张 train uncertain 不参与阈值选择，但仍保留角度和分类输出。

公式在相应角度约定下具有明确的参考轴夹角含义；但当前输入处理与原图坐标的对应尚未验证，不能将该分数直接称为视线与真实摄像头的物理夹角。头部正前方、角度零方向、眼睛指向摄像头的方向不能直接等同。

### 5.3 代码职责与运行命令

代码包：`gazetr_vfoa_baseline.zip`。

| 文件／函数 | 用途 |
|---|---|
| gazetr_adapter.py | 加载原 GazeTR Model、预处理图片、批量推理 |
| evaluate_gazetr.py / infer() | 全量预测并保存角度 |
| score_deg() | 计算合成角度分数 |
| calibrate() | 用 train 选择阈值 |
| report() | 分类、计算指标、保存结果 |
| config_gazetr_eval.yaml | 路径、GPU、batch、角度约定与搜索步长 |

先前已成功完成模型加载检查：

```bash
python check_gazetr.py \
  --weights "/home/lunet/cowz2/Documents/GazeEstimation/GazeTR/GazeTR-H-ETH.pt" \
  --device cuda:0
```

输入形状 (2,3,224,224)，输出 (2,2)。该检查仅证明权重和接口可用。原模型中的位置编码已按此前建议改为跟随 `feature.device`，避免硬编码 CUDA 设备。

之后运行过 `python predict_gazetr.py --config config_gazetr.yaml`，train 每类抽取 10 张，共 30 张、0 failures；这属于诊断，不用于报告正式性能。

全量实验按交付文件夹结构运行：

```bash
cd /home/lunet/cowz2/Documents/GazeEstimation/GazeTR/gazetr_vfoa_baseline
python -u evaluate_gazetr.py --config config_gazetr_eval.yaml
```

后台运行：

```bash
nohup setsid python -u evaluate_gazetr.py \
  --config config_gazetr_eval.yaml \
  > gazetr_eval.log 2>&1 < /dev/null &
```

查看日志：

```bash
tail -f gazetr_eval.log
```

交付配置默认 batch_size=32、device=cuda:0、seed=42，并沿用 ETH 诊断约定将输出解释为 pitch、yaw（radians）。原始 `output_0/1` 也保存在 CSV，便于后续核查。

**本次日志确认的实际结果目录：**

```text
/home/lunet/cowz2/Documents/GazeEstimation/GazeTR/runs/gazetr_rawcrop_binary_seed42
```

脚本的相对 output 按 config 所在目录解析。因此，交付文件夹结构与服务器实际部署不同时，输出位置可能不同；以 `config_resolved.json` 和上述运行日志为准。现有实验已经完成，明天无需重新推理。

### 5.4 验证结果

val 共预测 218 张脸，以下二分类指标针对 186 张 GT 确定人脸。

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Overall (macro) | 0.578 | 0.549 | 0.546 | 186 |
| Not looking at camera | 0.407 | 0.216 | 0.282 | 51 |
| Looking at camera | 0.748 | 0.881 | 0.810 | 135 |

**Accuracy：0.6989（69.89%）。**

混淆矩阵，行是真实类别、列是预测：

| GT | 预测不看 | 预测看 |
|---|---:|---:|
| 不看 | 11 | 40 |
| 看 | 16 | 119 |

另外 32 张 uncertain 中，6 张判不看、26 张判看。这是输出分布，不是 uncertain 识别准确率，不能将其中某一类的数量直接当作错误数。

Train 校准结果作为诊断保留，不作为独立评估：

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Overall (macro) | 0.597 | 0.576 | 0.580 | 1,400 |
| Not looking at camera | 0.417 | 0.298 | 0.347 | 363 |
| Looking at camera | 0.777 | 0.854 | 0.814 | 1,037 |

Train Accuracy=0.7100。Train uncertain 280 张中，54 张判不看、226 张判看。

### 5.5 结果解释

当前主要问题是“不看”被判成“看”：51 张中误判 40 张。由于看摄像头的样本更多，即使全部预测看，Accuracy 也能达到 72.58%；因此需要同时看 macro-F1 和每类召回率。

当前 macro-F1=0.546，高于全部预测看的约 0.421，但两类的区分仍有限。这个结果不能单独说明公式错误或 GazeTR 本身无效，后续需区分分数重叠、阈值敏感性和输入迁移等因素。

已有 val 分数中位数为：不看约 26.42°、看约 24.61°、uncertain 约 26.36°。中位数接近提示值得进一步查看完整分布，不能仅凭这一统计量判断排序能力。

### 5.6 继续分析需要的文件

| 文件 | 内容 |
|---|---|
| train_angles.csv、val_angles.csv | 全部角度与分数，无需再次运行 GazeTR |
| train_threshold_search.csv、threshold.json | 搜索曲线和选中阈值 |
| val_predictions.csv、val_metrics.csv | 逐脸类别与指标 |
| val_confusion_matrix.csv、val_summary.json | 混淆矩阵、Accuracy、uncertain 数量 |
| config_resolved.json、provenance.json | 实际配置、软件版本、模型文件哈希 |
| run.log、COMPLETE.json | 运行日志与完整结束标记 |

上传时出现的文件名括号编号，例如 `val_metrics(1).csv`，只是副本命名，不是新的实验设置。

## 6. 当前能得出的结论与比较边界

三项实验已分别跑通，阶段结果如下：

| 实验 | 主要指标 | 评估对象 |
|---|---|---|
| YOLO26s | mAP50=0.591，mAP50–95=0.448 | 三分类检测，218 个真实人脸 |
| ResNet18 | Macro-F1=0.693，Accuracy=0.7431 | GT 裁剪三分类 |
| GazeTR＋阈值 | Macro-F1=0.546，Accuracy=0.6989 | GT 裁剪二分类，186 张确定人脸 |

接下来的比较需要解决两点：

1. **统一类别和样本。** 在相同的 186 张确定人脸上重新统计 ResNet 结果；预测为 uncertain 的样本仍保留，计入真实类别的漏判和 Accuracy 错误，并单独报告拒判数。不能通过删除这些样本提高指标。
2. **统一系统输入。** 当前后两种方法得到的是 GT 框，YOLO 要自己检测。完整流程比较还需要给 ResNet、GazeTR 接入实际检测框，并明确匹配、漏检和误检的统计规则。

ResNet 使用类别标签更新网络权重；GazeTR 固定网络，仅学习阈值。论文应明确这一训练设置差异。结果能够支持本研究配置下的流程比较，不能直接推广为某类方法在所有条件下都更好。

当前 val 已用于开发和方案讨论。最终论文的泛化结论还需要未参与决策的数据，或一致、严格的分组评估流程。

## 7. 下一步：先验证分数，再统一比较

### 第一项：检查公式分数的区分能力

使用已保存的 CSV，分别分析 train 与 val：

- 画 pitch–yaw 散点图，按真实类别着色，观察零附近聚集、偏移和重叠。
- 画看／不看的合成分数分布；uncertain 单独展示。
- 以 `−score` 作为“看”的排序分数，计算 ROC-AUC。

这一步回答：分数较小是否通常对应看摄像头？如果两类分数本身高度重叠，单纯调整阈值能改善的程度就有限。

### 第二项：检查阈值稳定性

先看 train 的“阈值—macro-F1”曲线，再利用 manifest 中的原图信息，在 train 内按原始图片分组做交叉验证。每折在拟合部分选阈值，在留出部分计算指标，记录各折阈值和 macro-F1。

这一步回答：37.5° 是否依赖某一批样本？相邻阈值是否也能得到接近的表现？整个过程不依据 val 的逐张错误调整阈值。

这些检查验证的是经验分类规则的有效性。若要声称分数等于真实摄像头夹角，还需匹配的坐标处理与几何真值；当前不作这一物理精度声明。

### 第三项：重新统计可比较的分类结果

读取 ResNet `val_predictions.csv`，按原图／人脸标识匹配相同确定样本，保留预测 uncertain 的拒判情况。先完成 GT 裁剪上的配对比较，再推进检测框上的完整流程评估。

### 明天准备的材料

- [ ] GazeTR：`train_angles.csv`、`train_threshold_search.csv`、`threshold.json`。
- [ ] 裁剪记录：`face_crop_3c/manifest.csv`。
- [ ] ResNet：`val_predictions.csv` 与最终配置。
- [ ] YOLO：本轮 `train.py`、`args.yaml`、data YAML 与实际日志。

服务器检查命令：

```bash
cd /home/lunet/cowz2/Documents/GazeEstimation/GazeTR/runs/gazetr_rawcrop_binary_seed42
ls train_angles.csv val_angles.csv train_threshold_search.csv threshold.json config_resolved.json COMPLETE.json
```

```bash
ls /home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/Dataset/face_crop_3c/manifest.csv
```

明天优先复用现有结果进行分析，不重新训练、不重新裁剪、不再次预测全部角度。

## 8. 写 paper 时可复用的记录

### 方法描述草稿

> We evaluated three approaches to camera-directed visual attention recognition on a relabelled WIDER FACE subset. YOLO26s jointly predicted face bounding boxes and three attention labels. A ResNet18 classifier was trained on ground-truth face crops to predict the same labels. For the gaze-based baseline, a frozen ETH-pretrained GazeTR model produced angular outputs from the face crops. These outputs were converted into an angular score relative to the zero-angle reference direction. A decision threshold was fitted using the two certain classes in the training split by maximizing macro-F1 and was then fixed for validation. Ground-truth uncertain samples were excluded from binary threshold fitting and binary metrics, while their predictions were retained for separate analysis.

### GazeTR 结果描述草稿

> With a training-calibrated threshold of 37.5 degrees, the gaze-based baseline achieved a macro-F1 of 0.546 and an accuracy of 69.89% on 186 validation faces with certain labels. Recall was 0.881 for looking at the camera and 0.216 for not looking at the camera. The main error was the misclassification of non-camera-directed faces as camera-directed. These findings describe the current raw-crop transfer and thresholding configuration; they do not establish the accuracy of physical gaze angles in the original images.

论文中分别报告检测、三分类和二分类结果，注明样本数、训练监督与输入框来源。当前没有 WIDER 子集上的连续 gaze 真值，因此不报告其平均 gaze angular error，也不将这些验证分数表述为最终独立测试结果。

### 结果依据与归档

- YOLO：用户确认的启动命令、最终终端 class table、上传的 `results(1).csv`。
- ResNet：用户提供的 `val results, checkpoint epoch=6` 终端结果；详细配置待与服务器文件核对。
- GazeTR：`gazetr_eval.log`、`val_angles.csv`、`val_metrics(1).csv`、`val_summary.json`、`val_confusion_matrix(1).csv`，以及已核查的 reader/export 和推理代码。

归档时保留每轮脚本、配置、依赖版本、权重、预测文件与日志。所有日志都直接写到启动时的当前目录，无需专门创建 logs 文件夹。后台运行可避免普通终端断开影响，但不提供服务器重启后的自动恢复。

### 方法参考

- [GazeTR 仓库](https://github.com/yihuacheng/GazeTR)与[角度转换函数](https://github.com/yihuacheng/GazeTR/blob/main/gtools.py)。
- [ETH-XGaze 几何归一化示例](https://github.com/xucong-zhang/ETH-XGaze/blob/master/normalization_example.py)。
- [决策阈值调整与交叉验证](https://scikit-learn.org/stable/modules/classification_threshold.html)。
- [Cawley & Talbot：模型选择偏差与性能评估](https://www.jmlr.org/papers/v11/cawley10a.html)。
