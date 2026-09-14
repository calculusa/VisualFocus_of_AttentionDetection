# coco_to_yolo_labels.py

import json
from pathlib import Path
from collections import defaultdict, Counter


# =========================
# 你只需要改这里
# =========================

COCO_JSON = Path("/home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/Dataset/widerface_mmdet/widerface_val_selected61.json")
IMAGE_DIR = Path("/home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/Dataset/widerface_mmdet/widerface_val_selected61_image")
YOLO_LABEL_DIR = Path("/home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/Dataset/widerface_yolo26_subset/labels/val")

# COCO category_id -> YOLO class_id
# category_id 1 -> class 0 -> not_looking_at_camera
# category_id 2 -> class 1 -> looking_at_camera
CATEGORY_ID_TO_YOLO_ID = {
    1: 0,
    2: 1,
}

# =========================


def coco_bbox_to_yolo(bbox, image_width, image_height):
    """
    Convert COCO bbox to YOLO bbox.

    COCO bbox:
        [x_min, y_min, width, height]

    YOLO bbox:
        x_center, y_center, width, height
        all normalized to 0-1
    """
    x, y, w, h = bbox

    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    w_norm = w / image_width
    h_norm = h / image_height

    return x_center, y_center, w_norm, h_norm


def main():
    YOLO_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(COCO_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    image_info = {img["id"]: img for img in images}

    annotations_by_image = defaultdict(list)
    for ann in annotations:
        annotations_by_image[ann["image_id"]].append(ann)

    class_counter = Counter()
    missing_images = []
    invalid_boxes = []

    for image_id, img in image_info.items():
        file_name = img["file_name"]
        image_width = img["width"]
        image_height = img["height"]

        # Keep the original image filename.
        # Example:
        # data_20221119_1.jpg -> data_20221119_1.txt
        label_name = Path(file_name).with_suffix(".txt").name
        label_path = YOLO_LABEL_DIR / label_name

        # Optional image existence check.
        # This does not copy or modify images.
        image_path = IMAGE_DIR / file_name
        if not image_path.exists():
            missing_images.append(str(image_path))

        lines = []

        for ann in annotations_by_image.get(image_id, []):
            category_id = ann["category_id"]

            if category_id not in CATEGORY_ID_TO_YOLO_ID:
                continue

            class_id = CATEGORY_ID_TO_YOLO_ID[category_id]
            bbox = ann["bbox"]

            x, y, w, h = bbox

            if w <= 0 or h <= 0:
                invalid_boxes.append((file_name, bbox))
                continue

            x_center, y_center, w_norm, h_norm = coco_bbox_to_yolo(
                bbox,
                image_width,
                image_height,
            )

            # Basic range check.
            # If the original COCO bbox is valid, these should be between 0 and 1.
            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < w_norm <= 1
                and 0 < h_norm <= 1
            ):
                invalid_boxes.append((file_name, bbox))
                continue

            line = (
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{w_norm:.6f} "
                f"{h_norm:.6f}"
            )

            lines.append(line)
            class_counter[class_id] += 1

        label_path.write_text("\n".join(lines), encoding="utf-8")

    print("Conversion finished.")
    print(f"COCO annotation: {COCO_JSON}")
    print(f"YOLO labels saved to: {YOLO_LABEL_DIR}")
    print()
    print("Class counts:")
    print(f"  class 0 not_looking_at_camera: {class_counter[0]}")
    print(f"  class 1 looking_at_camera: {class_counter[1]}")
    print()
    print(f"Total images in COCO json: {len(images)}")
    print(f"Total annotations in COCO json: {len(annotations)}")
    print(f"Total label files created: {len(list(YOLO_LABEL_DIR.glob('*.txt')))}")

    if missing_images:
        print()
        print(f"Warning: {len(missing_images)} images were not found in IMAGE_DIR.")
        print("Example missing images:")
        for p in missing_images[:10]:
            print(" ", p)

    if invalid_boxes:
        print()
        print(f"Warning: {len(invalid_boxes)} invalid boxes were skipped.")
        print("Example invalid boxes:")
        for item in invalid_boxes[:10]:
            print(" ", item)


if __name__ == "__main__":
    main()