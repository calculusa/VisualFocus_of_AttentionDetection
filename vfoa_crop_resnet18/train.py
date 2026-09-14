import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import (
    CLASS_NAMES,
    CLASS_FOLDERS,
    NUM_CLASSES,
    IMAGE_SIZE,
    build_model,
    make_transform,
)


def parse_args():
    # 第一次解析：获取配置文件路径。
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
    )
    config_args, _ = config_parser.parse_known_args()
    config_path = config_args.config.expanduser().resolve()

    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Train or evaluate a VFOA ResNet18 classifier",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help="仅评估指定模型，不训练",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="仅评估模式使用的数据划分",
    )

    if not config_path.is_file():
        parser.error("找不到配置文件：{}".format(config_path))

    try:
        with config_path.open(encoding="utf-8") as handle:
            settings = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        parser.error("无法读取配置文件：{}".format(exc))

    if not isinstance(settings, dict):
        parser.error("配置文件必须是参数名称与数值的映射")

    converters = {
        "data": Path,
        "output": Path,
        "epochs": int,
        "batch": int,
        "lr": float,
        "weight_decay": float,
        "patience": int,
        "workers": int,
        "seed": int,
        "device": str,
        "eval_checkpoint": Path,
        "split": str,
    }

    unknown = set(settings) - set(converters)
    if unknown:
        parser.error(
            "未知配置参数：{}".format(
                ", ".join(sorted(map(str, unknown)))
            )
        )

    defaults = {}
    for key, value in settings.items():
        if value is None:
            if key != "eval_checkpoint":
                parser.error("{} 不能为空".format(key))
            defaults[key] = None
            continue

        converter = converters[key]
        try:
            if converter in (int, float) and isinstance(value, bool):
                raise ValueError("数值参数不能是布尔值")

            if converter is int and isinstance(value, float):
                if not value.is_integer():
                    raise ValueError("需要整数")

            defaults[key] = converter(value)

        except (TypeError, ValueError):
            parser.error(
                "{} 的值或类型无效：{}".format(key, value)
            )

    # 优先级：命令行 > YAML > 内置默认值。
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    args.config = config_path

    if args.data is None or args.output is None:
        parser.error("必须设置 data 和 output")

    if min(args.epochs, args.batch, args.patience) < 1:
        parser.error("epochs、batch、patience 必须大于 0")

    if (
        args.workers < 0
        or args.seed < 0
        or args.seed >= 2 ** 32
        or not np.isfinite(args.lr)
        or args.lr <= 0
        or not np.isfinite(args.weight_decay)
        or args.weight_decay < 0
    ):
        parser.error("workers、seed、lr 或 weight_decay 无效")

    if args.split not in ("val", "test"):
        parser.error("split 只能是 val 或 test")

    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if args.eval_checkpoint is not None:
        args.eval_checkpoint = (
            args.eval_checkpoint.expanduser().resolve()
        )
        if not args.eval_checkpoint.is_file():
            parser.error("找不到指定的 checkpoint")

    if not args.data.is_dir():
        parser.error("数据根目录不存在：{}".format(args.data))

    if args.output.exists():
        parser.error("输出目录已存在，请使用新目录")

    if args.output == args.data or args.data in args.output.parents:
        parser.error("输出目录不能位于数据目录内部")

    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(root, split, args, training=False):
    folder = root / split

    if not folder.is_dir():
        raise ValueError("找不到数据目录：{}".format(folder))

    dataset = ImageFolder(
        str(folder),
        transform=make_transform(training),
    )

    expected = {
        folder_name: index
        for index, folder_name in enumerate(CLASS_FOLDERS)
    }

    if dataset.class_to_idx != expected:
        raise ValueError(
            "类别目录不一致。\n实际：{}\n预期：{}".format(
                dataset.class_to_idx, expected
            )
        )

    counts = np.bincount(
        dataset.targets,
        minlength=NUM_CLASSES,
    )

    if np.any(counts == 0):
        raise ValueError(
            "{} 必须包含三个类别，实际数量：{}".format(
                split, counts.tolist()
            )
        )

    print("{}: {} faces, counts={}".format(
        split, len(dataset), counts.tolist()
    ))

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=training,
        num_workers=args.workers,
        pin_memory=torch.device(args.device).type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )

    return dataset, loader


def calculate_metrics(labels, predictions):
    # 行：真实类别；列：预测类别。
    matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )
    np.add.at(matrix, (labels, predictions), 1)

    tp = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)

    precision = np.divide(
        tp,
        predicted_counts,
        out=np.zeros(NUM_CLASSES),
        where=predicted_counts > 0,
    )
    recall = np.divide(
        tp,
        support,
        out=np.zeros(NUM_CLASSES),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(NUM_CLASSES),
        where=(precision + recall) > 0,
    )

    metrics = {
        "accuracy": float(tp.sum() / matrix.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": [
            {
                "class": CLASS_NAMES[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(NUM_CLASSES)
        ],
    }

    return metrics, matrix


def train_one_epoch(
    model, loader, optimizer, criterion, device
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError("训练 loss 出现 NaN 或 Inf")

        loss.backward()
        optimizer.step()

        count = labels.size(0)
        total_loss += loss.item() * count
        correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total += count

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probabilities = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError("评估 loss 出现 NaN 或 Inf")

        total_loss += loss.item() * labels.size(0)

        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(
            logits.softmax(dim=1).cpu().numpy()
        )

    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    predictions = probabilities.argmax(axis=1)

    metrics, matrix = calculate_metrics(
        labels, predictions
    )
    metrics["loss"] = float(total_loss / len(labels))

    return (
        metrics,
        matrix,
        labels,
        predictions,
        probabilities,
    )


def write_csv(path, header, rows):
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save_report(
    output, split, result, dataset, root, epoch
):
    metrics, matrix, labels, predictions, probs = result

    metrics.update({
        "split": split,
        "checkpoint_epoch": epoch,
        "evaluation": "GT face-crop classification",
    })

    (output / "{}_metrics.json".format(split)).write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    rows = [[
        "Overall (macro)",
        metrics["macro_precision"],
        metrics["macro_recall"],
        metrics["macro_f1"],
        len(labels),
    ]]

    for row in metrics["per_class"]:
        rows.append([
            row["class"],
            row["precision"],
            row["recall"],
            row["f1"],
            row["support"],
        ])

    write_csv(
        output / "{}_metrics.csv".format(split),
        ["Class", "Precision", "Recall", "F1", "Support"],
        rows,
    )

    write_csv(
        output / "{}_confusion_matrix.csv".format(split),
        ["true / predicted"] + CLASS_NAMES,
        [
            [name] + row.tolist()
            for name, row in zip(CLASS_NAMES, matrix)
        ],
    )

    # 读取裁剪时保存的原图对应信息。
    provenance = {}
    manifest = root / "manifest.csv"

    if manifest.is_file():
        with manifest.open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                provenance[row["crop_path"]] = row

    prediction_rows = []

    # 评估 loader 不打乱顺序，与 dataset.samples 对齐。
    for sample, true, pred, probability in zip(
        dataset.samples, labels, predictions, probs
    ):
        relative = (
            Path(sample[0]).relative_to(root).as_posix()
        )
        source = provenance.get(relative, {})

        prediction_rows.append([
            relative,
            source.get("source_image", ""),
            source.get("label_line", ""),
            int(true),
            int(pred),
        ] + probability.tolist())

    write_csv(
        output / "{}_predictions.csv".format(split),
        [
            "crop_path",
            "source_image",
            "label_line",
            "true_class_id",
            "predicted_class_id",
        ] + ["prob_" + name for name in CLASS_NAMES],
        prediction_rows,
    )

    print("\n{} results, checkpoint epoch={}".format(
        split, epoch
    ))
    print("{:<28} {:>9} {:>9} {:>9}".format(
        "Class", "Precision", "Recall", "F1"
    ))

    for row in rows:
        print("{:<28} {:>9.3f} {:>9.3f} {:>9.3f}".format(
            row[0], row[1], row[2], row[3]
        ))

    print("Accuracy: {:.4f}".format(metrics["accuracy"]))
    print("Results saved to: {}".format(output))


def load_checkpoint(path, device):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("Checkpoint 类别映射不一致")

    if checkpoint["image_size"] != IMAGE_SIZE:
        raise ValueError("Checkpoint 输入尺寸不一致")

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])

    return model.to(device), checkpoint


def save_config(args):
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    config.update({
        "architecture": "resnet18",
        "class_names": CLASS_NAMES,
        "class_to_idx": {
            name: i for i, name in enumerate(CLASS_FOLDERS)
        },
        "image_size": IMAGE_SIZE,
        "resize": "aspect-preserving resize and pad",
        "train_augmentation": "horizontal flip, p=0.5",
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "loss": "unweighted cross entropy",
        "optimizer": "AdamW",
        "selection_metric": "validation macro-F1",
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
    })

    (args.output / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    return config


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用，请检查环境或设置 device: cpu"
        )

    print("Device: {}".format(device))
    criterion = nn.CrossEntropyLoss()

    # 仅评估模式：不会训练或修改模型参数。
    if args.eval_checkpoint is not None:
        dataset, loader = make_loader(
            args.data, args.split, args
        )
        model, checkpoint = load_checkpoint(
            args.eval_checkpoint, device
        )

        args.output.mkdir(parents=True, exist_ok=False)
        save_config(args)

        result = evaluate(
            model, loader, criterion, device
        )
        save_report(
            args.output,
            args.split,
            result,
            dataset,
            args.data,
            checkpoint["epoch"],
        )
        return

    # 训练模式：只使用 train 和 val。
    _, train_loader = make_loader(
        args.data, "train", args, training=True
    )
    val_dataset, val_loader = make_loader(
        args.data, "val", args, training=False
    )

    model = build_model(pretrained=True).to(device)

    # 微调整个网络，不冻结 backbone。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    config = save_config(args)

    best_f1 = -1.0
    stale_epochs = 0
    best_path = args.output / "best.pt"

    with (args.output / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "val_macro_precision",
            "val_macro_recall",
            "val_macro_f1",
        ])

        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )

            metrics = evaluate(
                model, val_loader, criterion, device
            )[0]

            writer.writerow([
                epoch,
                train_loss,
                train_accuracy,
                metrics["loss"],
                metrics["accuracy"],
                metrics["macro_precision"],
                metrics["macro_recall"],
                metrics["macro_f1"],
            ])
            handle.flush()

            improved = (
                metrics["macro_f1"] > best_f1 + 1e-8
            )

            if improved:
                best_f1 = metrics["macro_f1"]
                stale_epochs = 0

                torch.save({
                    "model_state": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "best_val_macro_f1": best_f1,
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "config": config,
                }, best_path)
            else:
                stale_epochs += 1

            print(
                "Epoch {:02d}/{:02d} | "
                "train loss {:.4f} | train Acc {:.4f} | "
                "val loss {:.4f} | val Acc {:.4f} | "
                "val Macro-F1 {:.4f}{}".format(
                    epoch,
                    args.epochs,
                    train_loss,
                    train_accuracy,
                    metrics["loss"],
                    metrics["accuracy"],
                    metrics["macro_f1"],
                    " [best]" if improved else "",
                ),
                flush=True,
            )

            if stale_epochs >= args.patience:
                print("Early stopping.")
                break

    # 最终结果来自最佳模型，不是最后一轮。
    model, checkpoint = load_checkpoint(
        best_path, device
    )
    result = evaluate(
        model, val_loader, criterion, device
    )

    save_report(
        args.output,
        "val",
        result,
        val_dataset,
        args.data,
        checkpoint["epoch"],
    )

    print("Best model: {}".format(best_path))


if __name__ == "__main__":
    main()