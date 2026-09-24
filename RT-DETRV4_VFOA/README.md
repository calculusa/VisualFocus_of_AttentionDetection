# RT-DETRv4 VFOA evaluation

将 eval_rtv4_vfoa.py 放到 RT-DETRv4 根目录，在该目录执行。无需重新训练，不实例化 DINOv3 教师模型。
使用自己训练的可信 checkpoint（torch.load 使用 weights_only=False）。默认优先读取 EMA，与训练期间验证一致。

## 1. 环境

```bash
conda activate rtv4
python -m pip install matplotlib
```

沿用已跑通训练的环境及 requirements。COCO 评估优先使用已安装的 faster-coco-eval，缺少时才使用 pycocotools。

## 2. 快速验证（固定 confidence=0.25）

```bash
CUDA_VISIBLE_DEVICES=0 python eval_rtv4_vfoa.py \
  -c configs/rtv4/rtv4_x_vfoa.yml \
  -w outputs/rtv4_x_vfoa_50ep_seed0_20260923/best_stg1.pth \
  --split val --conf 0.25 --out evaluations/rtv4_x_val_conf025
```

0.25 只是明确的初始工作阈值，不声称最佳。先核对 coco_metrics.json 中 mAP50–95≈0.4753、mAP50≈0.6480；如明显不同，先检查权重、EMA、配置和数据路径。浮点误差可能产生小幅差异。

## 3. 正式比较：在 train 上选阈值，再固定到 val/test

```bash
CUDA_VISIBLE_DEVICES=0 python eval_rtv4_vfoa.py \
  -c configs/rtv4/rtv4_x_vfoa.yml \
  -w outputs/rtv4_x_vfoa_50ep_seed0_20260923/best_stg1.pth \
  --split train \
  --images /home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/Dataset/widerface_vfoa_rtDetr/images/train \
  --ann /home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/Dataset/widerface_vfoa_rtDetr/annotations/instances_train.json \
  --select-threshold --out evaluations/rtv4_x_train_threshold

CUDA_VISIBLE_DEVICES=0 python eval_rtv4_vfoa.py \
  -c configs/rtv4/rtv4_x_vfoa.yml \
  -w outputs/rtv4_x_vfoa_50ep_seed0_20260923/best_stg1.pth \
  --split val \
  --threshold-file evaluations/rtv4_x_train_threshold/threshold.json \
  --out evaluations/rtv4_x_val_frozen
```

train 推理也使用无随机增强的 val transforms。搜索 0–1、步长0.005，以三类 macro-F1 最大选阈值；并列取最低阈值。训练集选择会受拟合程度影响，应在论文说明；不可再根据 val/test 曲线调整冻结阈值。checkpoint 已在 val 上选择，最终泛化结果要在独立 test 上报告。

测试集：以上第二条命令改为 --split test，并增加实际 --images 和 --ann 路径，输出至新目录，保持 threshold-file 不变。

## 4. 输出

- coco_metrics.json：COCO 总体 AP/AR、每类 AP50 和 AP50–95。
- per_class_metrics.csv：每类 support、TP、FP、FN、P、R、F1、AP。
- summary.json：macro-P/R/F1、阈值、匹配规则、checkpoint/标注 SHA256、权重来源。
- predictions_coco.json：官方后处理的全部预测（原图坐标 xywh），用于后续统一评估。
- confusion_matrix.csv/png：行=真实，列=预测；background 行=未匹配预测，background 列=漏检。
- confusion_matrix_normalized.png：按真实类别行归一化。
- BoxPR_curve.png：COCO IoU0.50 的101点插值 PR，maxDets100。
- BoxP_curve.png、BoxR_curve.png、BoxF1_curve.png：本脚本空间匹配协议下的阈值曲线。
- threshold_curve.csv：阈值与 macro 指标；仅 --select-threshold 输出 threshold.json。

## 5. 评估定义与公平性

AP 沿用官方 top-300 后处理，不额外 NMS、不先按 --conf 过滤；交给 COCO 评估器按 maxDets=100 计算。AP 汇总中的 -1 表示该分组无有效 GT，不能当作性能值。

P/R/F1 使用固定 confidence 与 IoU>=0.50：每图预测按置信度降序排列，每个预测匹配 IoU 最大且尚未匹配的 GT，空间匹配不要求类别相同。错分类计入真实类别 FN 和预测类别 FP；重复框/未匹配预测计 FP；未匹配 GT 计 FN。Macro-F1 是三个类别 F1 的算术平均，不包含 background。没有预测的类别 P 记0。当前只支持0/1/2三类、无 crowd/ignore 的 VFOA 数据。

COCO AP 自己使用类别相关匹配；因此 COCO PR 与本脚本的空间混淆矩阵是两种明确分开的统计，不要相互推导。

本脚本不是 Ultralytics 指标的逐行复刻；YOLO 日志的 P/R 常对应另一套阈值和匹配方式。正式横向比较时，对各方法的预测使用同一评估器、IoU、阈值选择规则及相同图片/GT。裁剪分类报告仍不能直接与全图检测 F1 对比：需把 crop 预测映射回候选框，计入漏检和误检。RT-DETR 独立检测结果属于端到端比较，不属于共同 YOLO 候选框分类实验。

运行使用 FP32、默认 batch8；显存不足加 --batch-size 2。单GPU无需 torchrun。保留现有训练配置，不添加 ImageNet normalization，不使用 letterbox；脚本直接读取配置中的 val dataset/transforms。

## 验证范围

已通过 Python 语法检查和合成案例（完美预测、错分类、重复框、漏检、空预测）验证匹配统计。未在用户 GPU/真实 checkpoint 上运行；请先用第2步核对已知 AP。脚本会自动移动旧编码器缓存的位置编码，避免先前 CPU/GPU 不一致问题。
