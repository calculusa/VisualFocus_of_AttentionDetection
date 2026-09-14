# GazeTR：VFOA 人脸裁剪图直接迁移基线

固定 ETH 预训练权重，不训练或微调 GazeTR。对 train/val 全量人脸推理，在 train 的两个确定类别上选择阈值，固定阈值后评估 val。复用你已经验证过能加载权重的 GazeTR 仓库。

## 文件

- `gazetr_adapter.py`：导入现有仓库的 `model.py`，严格加载权重、处理图片、批量推理。
- `evaluate_gazetr.py`：全量预测、阈值搜索、指标统计及日志。
- `config_gazetr_eval.yaml`：路径、设备和运行参数。

不会覆盖原来的 `predict_gazetr.py`、`model.py`、配置或结果。输出目录必须是新目录；重跑时更改 config 的 output。

## 运行

将本文件夹放到服务器 GazeTR 目录下，进入本文件夹。使用之前运行 check_gazetr.py 成功的 conda 环境，不需要升级 PyTorch。依赖：原 GazeTR 仓库依赖，以及现有的 torch、numpy、opencv-python、PyYAML。

```bash
cd /home/lunet/cowz2/Documents/GazeEstimation/GazeTR/gazetr_vfoa_baseline
python -u evaluate_gazetr.py --config config_gazetr_eval.yaml
```

后台运行并保留启动日志：

```bash
nohup setsid python -u evaluate_gazetr.py --config config_gazetr_eval.yaml > gazetr_eval.log 2>&1 < /dev/null &
tail -f gazetr_eval.log
```

不用 mkdir logs。脚本还会自动写入输出目录的 `run.log`。断开 SSH 后进程可继续运行，但服务器重启或管理员结束进程仍会中断。此版本不自动恢复中断的推理；数据量约 1900 张人脸，重跑请指定新 output。

配置预填了你提供的 Linux 路径。默认 cuda:0、batch_size:32。使用其他卡修改 device。CPU 需要原 model.py 的位置编码已经按之前讨论改为跟随 feature.device。

输入目录支持以下三个类别文件夹名（每类只能出现其中一种）：

| ID | 支持的文件夹名 |
|---|---|
| 0 | `0_not_looking_at_camera`、`not_looking_at_camera`、`0` |
| 1 | `1_looking_at_camera`、`looking_at_camera`、`1` |
| 2 | `2_uncertain`、`uncertain`、`2` |

类别目录位于 data/train 和 data/val 下；不依赖 manifest.csv。应得到 train 363/1037/280，val 51/135/32。如不一致，请先核对你是否改过数据划分。

## 输出

默认 `runs/gazetr_rawcrop_binary_seed42/`：

| 文件 | 内容 |
|---|---|
| train_angles.csv / val_angles.csv | 全部人脸的两个原始输出、角度、判定分数 |
| train_threshold_search.csv | train 上固定阈值网格的指标 |
| threshold.json | train 选出的阈值、优化目标和判定规则 |
| val_metrics.csv | 两个确定类别的 Precision、Recall、F1、Support 及宏平均 |
| val_summary.json | Accuracy、样本数、uncertain 被分到两类的数量 |
| val_predictions.csv | 全部预测和是否纳入二分类指标的标记 |
| val_confusion_matrix.csv | 行是真实类别，列是预测类别 |
| train_metrics.csv 等 | 校准集结果，仅作诊断 |
| run.log / config_resolved.json / provenance.json | 日志、配置、版本及权重哈希 |
| COMPLETE.json | 全部完成才生成；没有此文件时不能把输出当作完整结果 |

读图失败、非有限输出、权重不匹配会报错并停止，避免无声剔除难样本。

## 实验定义

1. 图片处理依据你提供的 reader.py：OpenCV 读取，保留通道顺序，CHW 浮点数 /255，无 ImageNet Normalize。为任意尺寸裁剪图额外使用 INTER_LINEAR 直接缩放至 224×224，不做几何归一化或关键点处理。
2. 默认沿用前一个诊断脚本的 ETH 输出约定：pitch、yaw，弧度。提供的 reader/export 文件本身不定义角度顺序和单位。CSV 保留原始 output_0/1 便于核查。判定分数 `acos(cos(pitch)*cos(yaw))` 对两个角度交换不敏感，但单位必须正确。
3. 分数仅作为此直接迁移实验的角度判定分数，不能当作经过验证的真实摄像头夹角，也不能用这些分类标注计算 gaze angular error。数据集上游 face_patch 的几何处理没有由这两个脚本复现。
4. 在 train 的 GT=0/1 人脸上，按 0.5° 搜索 0–180°，另含 -1°（全部判为不看）。最大化 binary macro-F1；并列取最小阈值。score <= threshold 判为看镜头。阈值写入文件后才处理 val。
5. uncertain 不参与阈值选择或二分类指标，但全部推理并单独保存分配数量。模型不输出第三类，不能称为三分类识别结果。无预测正例时 precision 按 0 处理；macro-F1 是两类 F1 的平均。
6. 预计 train 二分类 1400 张脸，val 二分类 186 张脸。需用相同 186 张确定人脸重新统计 ResNet 二分类比较表，不能直接与之前 218 张脸的三分类 macro-F1 比较。ResNet 若预测 uncertain，不能删除该样本，应明确记为拒判/错误。
7. 这是 GT 人脸框裁剪评估；YOLO 检测指标包含定位与漏检，不能直接等同。完整两阶段实验还需要接入检测框并统一匹配和漏检统计。
8. 当前 val 仍是开发用验证集，不能称独立测试集。后续测试时复用已锁定的方法与阈值规则，不根据测试结果调整配置。原始图片、视频或人物分组划分应在裁剪之前完成；脚本仅检查跨 split 字节完全相同的裁剪图，不能排除所有近重复或同源泄漏。

该流程不需要 gaze 连续角度真值，因为不训练角度回归器；只使用 train 的现有类别标签校准分类阈值。
