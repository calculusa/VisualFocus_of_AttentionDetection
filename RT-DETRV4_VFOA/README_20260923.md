# RT-DETRv4-S VFOA 三分类

代码依据 https://github.com/RT-DETRs/RT-DETRv4 ，检查版本：
55fefaaed7efe2a5f72d0a18fd4e05965e35c292。

1. 新建环境，避免改动现有 YOLO/GazeTR 环境：

```bash
git clone https://github.com/RT-DETRs/RT-DETRv4.git
cd RT-DETRv4
conda create -n rtv4 python=3.11.9 -y
conda activate rtv4
python -m pip install -r requirements.txt
git clone https://github.com/facebookresearch/dinov3.git dinov3
mkdir -p pretrain
```

2. 从官方 README 的模型表下载 **RT-DETRv4-S COCO checkpoint**，保存为
`pretrain/rtv4_s_coco.pth`。从 README 指向的 DINOv3 官方下载入口获取
**ViT-B/16 LVD-1689M** 权重，遵循官方访问流程。不要换成 ViT-S/L 或其他预训练版本。

3. 将 `rtv4_s_vfoa.yml` 放到 `configs/rtv4/`；将 `check_vfoa_coco.py`
放到仓库根目录。修改 YAML 内四个数据路径和 DINOv3 权重路径。
数据必须是原图加 COCO 检测框，使用与 YOLO 相同的 train/val 划分，不是 face crops。
category_id 必须为 0=not_looking_at_camera，1=looking_at_camera，2=uncertain。
若当前是 1/2/3，先按类别名称同时映射 categories 和 annotations；检查器不会自动改文件。

4. 检查数据和环境：

```bash
python check_vfoa_coco.py configs/rtv4/rtv4_s_vfoa.yml
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

5. 正式训练（四卡，总 batch=16，每卡4）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  -c configs/rtv4/rtv4_s_vfoa.yml \
  -t pretrain/rtv4_s_coco.pth --use-amp --seed=0
```

后台运行（二选一，不要同时启动）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --standalone --nproc_per_node=4 train.py \
  -c configs/rtv4/rtv4_s_vfoa.yml \
  -t pretrain/rtv4_s_coco.pth --use-amp --seed=0 \
  > rtv4_s_vfoa.log 2>&1 < /dev/null &
```

`-t` 是迁移微调，`-r` 是恢复同一实验；不要用 `-r` 加载80类COCO权重开始3类实验。
初始化日志应显示大量权重正常加载；类别相关 head 不匹配是预期行为，但 backbone/encoder
大量不匹配需要停下来检查 checkpoint 型号。

配置保留官方 DINOv3 蒸馏及增强，输入基准640，50轮；warmup=200 iterations，
flat_epoch=25，增强切换=[4,25,40]，最后10轮关闭相应强增强。
这些是本任务适配参数，不是作者的原始132轮设置，也不能称与YOLO的全部训练设置相同。
学习率和损失沿用官方S配置。蒸馏教师是额外预训练资源，论文应明确记录。
相同50轮是预算控制，不保证不同模型都达到最优。

输出在 `outputs/rtv4_s_vfoa_50ep_seed0/`，日志、best_stg1.pth、best_stg2.pth
均保留；依据验证集AP选择阶段checkpoint，不要默认stg2更好。
官方评估AP不等于统一目标级Macro-F1，后续仍需统一导出与评估。
测试集保持独立，不用于训练/选择参数。

成功运行后记录环境和两个仓库版本：

```bash
python -m pip freeze > outputs/rtv4_s_vfoa_50ep_seed0/environment.txt
git rev-parse HEAD > outputs/rtv4_s_vfoa_50ep_seed0/rtdetr_commit.txt
git -C dinov3 rev-parse HEAD > outputs/rtv4_s_vfoa_50ep_seed0/dinov3_commit.txt
```

本包已验证YAML继承合并及检查器语法/小型样例；未在GPU或你的实际数据上训练。
若出现依赖或CUDA错误，保留完整报错，先不要改动现有yolo环境。
