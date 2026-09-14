#!/usr/bin/env python3
"""Crop faces from a YOLO-labelled WIDER FACE dataset.

Expected dataset layout:

    DATASET_ROOT/
      images/train/...
      images/val/...
      images/test/...
      labels/train/...
      labels/val/...
      labels/test/...

Each YOLO label line must be:

    class_id x_center y_center width height

where the four box coordinates are normalized to [0, 1]. Crops are saved in
class folders and a CSV manifest is produced for reproducible downstream use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop faces from YOLO labels for GazeTR and two-stage classification."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Dataset splits to process (default: train val test).",
    )
    parser.add_argument(
        "--class-names", nargs="+", default=["not_look", "look", "uncertain"],
        help="Class names in class-id order.",
    )
    parser.add_argument(
        "--include-classes", nargs="*", type=int, default=None,
        help="Optional class IDs to keep. By default, all classes are kept.",
    )
    parser.add_argument(
        "--padding", type=float, default=0.15,
        help="Fraction added to every side of the bounding box (default: 0.15).",
    )
    parser.add_argument(
        "--size", type=int, default=224,
        help="Square output size in pixels; use 0 to preserve crop size (default: 224).",
    )
    parser.add_argument(
        "--min-face", type=int, default=0,
        help="Skip boxes whose original width or height is below this many pixels.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing crop files. Otherwise they are skipped.",
    )
    args = parser.parse_args()

    if args.padding < 0:
        parser.error("--padding must be non-negative")
    if args.size < 0:
        parser.error("--size must be zero or a positive integer")
    if args.min_face < 0:
        parser.error("--min-face must be non-negative")
    return args


def read_yolo_labels(path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, got {len(fields)}")
        class_text, *coords_text = fields
        class_id = int(class_text)
        coords = [float(value) for value in coords_text]
        if not all(0.0 <= value <= 1.0 for value in coords):
            raise ValueError(f"{path}:{line_number}: normalized coordinates must be in [0, 1]")
        if coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"{path}:{line_number}: box width and height must be positive")
        boxes.append(YoloBox(class_id, *coords))
    return boxes


def box_to_pixels(box: YoloBox, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x1 = (box.x_center - box.width / 2.0) * image_width
    y1 = (box.y_center - box.height / 2.0) * image_height
    x2 = (box.x_center + box.width / 2.0) * image_width
    y2 = (box.y_center + box.height / 2.0) * image_height
    return x1, y1, x2, y2


def padded_square_box(
    box: tuple[float, float, float, float],
    padding: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    x1 -= width * padding
    x2 += width * padding
    y1 -= height * padding
    y2 += height * padding

    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = int(round(cx + side / 2.0))
    bottom = int(round(cy + side / 2.0))

    # PIL permits coordinates outside the image and fills them with black. We
    # keep them here and use edge padding below, avoiding asymmetric clipping.
    if right <= left or bottom <= top:
        raise ValueError("invalid crop box after padding")
    return left, top, right, bottom


def crop_with_edge_padding(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - image.width)
    pad_bottom = max(0, bottom - image.height)

    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=(0, 0, 0),
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top
    return image.crop((left, top, right, bottom))


def safe_stem(relative_image_path: Path) -> str:
    path_text = relative_image_path.with_suffix("").as_posix()
    readable = path_text.replace("/", "__").replace(" ", "_")
    digest = hashlib.sha1(path_text.encode("utf-8")).hexdigest()[:8]
    return f"{readable}__{digest}"


def iter_images(images_dir: Path) -> Iterable[Path]:
    return sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def process_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    class_names: Sequence[str],
    include_classes: set[int] | None,
    padding: float,
    output_size: int,
    min_face: int,
    overwrite: bool,
) -> dict[str, int]:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    if not images_dir.is_dir():
        print(f"[WARNING] Missing image directory, skipping: {images_dir}", file=sys.stderr)
        return {"images": 0, "crops": 0, "skipped": 0, "errors": 0}
    if not labels_dir.is_dir():
        print(f"[WARNING] Missing label directory, skipping: {labels_dir}", file=sys.stderr)
        return {"images": 0, "crops": 0, "skipped": 0, "errors": 0}

    split_output = output_root / split
    split_output.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"manifest_{split}.csv"
    rows: list[dict[str, object]] = []
    counts = {"images": 0, "crops": 0, "skipped": 0, "errors": 0}

    for image_path in iter_images(images_dir):
        relative_image = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_image.with_suffix(".txt")
        if not label_path.exists():
            print(f"[WARNING] No label for {relative_image}", file=sys.stderr)
            counts["skipped"] += 1
            continue

        try:
            boxes = read_yolo_labels(label_path)
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            counts["images"] += 1
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            print(f"[ERROR] {image_path}: {exc}", file=sys.stderr)
            counts["errors"] += 1
            continue

        prefix = safe_stem(relative_image)
        for box_index, yolo_box in enumerate(boxes):
            if include_classes is not None and yolo_box.class_id not in include_classes:
                counts["skipped"] += 1
                continue
            if not 0 <= yolo_box.class_id < len(class_names):
                print(
                    f"[ERROR] {label_path}: class ID {yolo_box.class_id} has no class name",
                    file=sys.stderr,
                )
                counts["errors"] += 1
                continue

            original_box = box_to_pixels(yolo_box, image.width, image.height)
            original_width = original_box[2] - original_box[0]
            original_height = original_box[3] - original_box[1]
            if original_width < min_face or original_height < min_face:
                counts["skipped"] += 1
                continue

            try:
                crop_box = padded_square_box(
                    original_box, padding, image.width, image.height
                )
                crop = crop_with_edge_padding(image, crop_box)
                if output_size:
                    crop = crop.resize((output_size, output_size), Image.Resampling.BILINEAR)
            except ValueError as exc:
                print(f"[ERROR] {label_path}, box {box_index}: {exc}", file=sys.stderr)
                counts["errors"] += 1
                continue

            class_name = class_names[yolo_box.class_id]
            class_dir = split_output / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            crop_name = f"{prefix}__face_{box_index:03d}.jpg"
            crop_path = class_dir / crop_name
            already_exists = crop_path.exists() and not overwrite
            if already_exists:
                counts["skipped"] += 1
            else:
                crop.save(crop_path, format="JPEG", quality=95, subsampling=0)
                counts["crops"] += 1

            x1, y1, x2, y2 = original_box
            cx1, cy1, cx2, cy2 = crop_box
            rows.append(
                {
                    "split": split,
                    "source_image": relative_image.as_posix(),
                    "source_label": label_path.relative_to(dataset_root).as_posix(),
                    "box_index": box_index,
                    "class_id": yolo_box.class_id,
                    "class_name": class_name,
                    "crop_path": crop_path.relative_to(output_root).as_posix(),
                    "image_width": image.width,
                    "image_height": image.height,
                    "bbox_x1": round(x1, 3),
                    "bbox_y1": round(y1, 3),
                    "bbox_x2": round(x2, 3),
                    "bbox_y2": round(y2, 3),
                    "crop_x1": cx1,
                    "crop_y1": cy1,
                    "crop_x2": cx2,
                    "crop_y2": cy2,
                    "original_face_width": round(original_width, 3),
                    "original_face_height": round(original_height, 3),
                }
            )

    fieldnames = [
        "split", "source_image", "source_label", "box_index", "class_id",
        "class_name", "crop_path", "image_width", "image_height", "bbox_x1",
        "bbox_y1", "bbox_x2", "bbox_y2", "crop_x1", "crop_y1", "crop_x2",
        "crop_y2", "original_face_width", "original_face_height",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return counts


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    include_classes = set(args.include_classes) if args.include_classes is not None else None
    output_root.mkdir(parents=True, exist_ok=True)

    total = {"images": 0, "crops": 0, "skipped": 0, "errors": 0}
    for split in args.splits:
        counts = process_split(
            dataset_root=dataset_root,
            output_root=output_root,
            split=split,
            class_names=args.class_names,
            include_classes=include_classes,
            padding=args.padding,
            output_size=args.size,
            min_face=args.min_face,
            overwrite=args.overwrite,
        )
        print(
            f"[{split}] images={counts['images']} crops={counts['crops']} "
            f"skipped={counts['skipped']} errors={counts['errors']}"
        )
        for key in total:
            total[key] += counts[key]

    print(
        f"[TOTAL] images={total['images']} crops={total['crops']} "
        f"skipped={total['skipped']} errors={total['errors']}"
    )
    return 1 if total["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
