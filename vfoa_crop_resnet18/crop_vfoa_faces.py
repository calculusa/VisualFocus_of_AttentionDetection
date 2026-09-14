#!/usr/bin/env python3
"""Crop VFOA faces from existing YOLO detection labels (Python >= 3.8).

Install: python -m pip install Pillow
Run: python crop_vfoa_faces.py --dataset /path/to/dataset --output /path/to/face_crops

Input layouts: images/train + labels/train OR train/images + train/labels,
and corresponding val/test folders. Splits are preserved, never regenerated.
Use --splits train val when there is no separate test folder.
Labels: 0 not_looking_at_camera, 1 looking_at_camera, 2 uncertain.
Each label line must be: class_id center_x center_y width height (normalized).

Output: split/0_not_looking_at_camera/*.png, split/1_looking_at_camera/*.png,
split/2_uncertain/*.png, manifest.csv and summary.json.
All three class directories are created in every split, even if empty.
Numeric prefixes preserve class order for alphabetical folder-based loaders.
Some loaders reject empty classes; review the summary before training.

Default margin is zero; --margin 0.15 adds 15% of box width/height on EACH side.
Faces retain their native resolution and aspect ratio; there is no size filter.
PIL exif_transpose is applied before interpreting labels (upright image coords).
No source files are modified. Output directory must not exist.
The manifest retains geometry for audit and later gaze preprocessing. These
crops alone do not implement GazeTR's geometric normalization.
Exact decoded-image duplicates across splits are rejected. This is not a
near-duplicate, identity, scene or earlier-training-overlap check.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageOps

CLASSES = {0: 'not_looking_at_camera', 1: 'looking_at_camera', 2: 'uncertain'}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
FIELDS = ['split', 'source_image', 'source_label', 'label_line', 'class_id',
          'class_name', 'crop_path', 'image_width', 'image_height',
          'center_x', 'center_y', 'box_width_norm', 'box_height_norm',
          'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
          'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2',
          'crop_width', 'crop_height', 'margin_per_side', 'clipped_to_image',
          'upright_rgb_sha256']


def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def snap_integer(value):
    """Avoid an extra border pixel caused by e.g. 7.000000000000001."""
    nearest = round(value)
    return nearest if math.isclose(value, nearest, rel_tol=0, abs_tol=1e-9) else value


def directories(root, split):
    layouts = [(root / 'images' / split, root / 'labels' / split),
               (root / split / 'images', root / split / 'labels')]
    valid = [pair for pair in layouts if all(p.is_dir() for p in pair)]
    if len(valid) != 1:
        raise ValueError('Expected exactly one image/label directory pair for '
                         '%s; found %d. Use --splits to specify existing splits.'
                         % (split, len(valid)))
    return valid[0]


def parse_boxes(label, width, height, margin):
    result = []
    for line_number, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(v) for v in line.split()]
            if len(values) != 5 or not all(math.isfinite(v) for v in values):
                raise ValueError('expected five finite numbers')
            cls, cx, cy, bw, bh = values
            if cls != int(cls) or int(cls) not in CLASSES:
                raise ValueError('class must be 0, 1 or 2')
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError('invalid normalized coordinates or non-positive box size')
            box = [(cx - bw / 2) * width, (cy - bh / 2) * height,
                   (cx + bw / 2) * width, (cy + bh / 2) * height]
            if min(width, box[2]) <= max(0, box[0]) or min(height, box[3]) <= max(0, box[1]):
                raise ValueError('box has no intersection with image')
            expanded = [box[0] - margin * bw * width, box[1] - margin * bh * height,
                        box[2] + margin * bw * width, box[3] + margin * bh * height]
            bounds = [max(0, math.floor(snap_integer(expanded[0]))), max(0, math.floor(snap_integer(expanded[1]))),
                      min(width, math.ceil(snap_integer(expanded[2]))), min(height, math.ceil(snap_integer(expanded[3])))]
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError('box is too small to yield a non-empty pixel crop')
            clipped = expanded[0] < 0 or expanded[1] < 0 or expanded[2] > width or expanded[3] > height
            result.append((line_number, int(cls), values[1:], box, bounds, clipped))
        except ValueError as exc:
            raise ValueError('%s:%d: %s' % (label, line_number, exc)) from exc
    return result


def prepare(args):
    records, seen, counts = [], {}, {}
    for split in args.splits:
        images_dir, labels_dir = directories(args.dataset, split)
        images = sorted(p for p in images_dir.rglob('*') if p.suffix.lower() in EXTENSIONS and p.is_file())
        if not images:
            raise ValueError('No images in %s' % images_dir)
        counts[split] = {'images': len(images), 'images_without_faces': 0,
                         'faces': {str(k): 0 for k in CLASSES}}
        used_labels = set()
        for image_path in images:
            label = labels_dir / image_path.relative_to(images_dir).with_suffix('.txt')
            if not label.is_file():
                raise ValueError('Missing label: %s. A background image needs an explicit empty .txt.' % label)
            if label in used_labels:
                raise ValueError('Multiple images map to the same label: %s' % label)
            used_labels.add(label)
            image = load_rgb(image_path)
            width, height = image.size
            digest = hashlib.sha256(('%d,%d:' % (width, height)).encode() + image.tobytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                raise ValueError('Identical image crosses splits: %s and %s' % (seen[digest][1], image_path))
            seen[digest] = (split, str(image_path))
            boxes = parse_boxes(label, width, height, args.margin)
            counts[split]['images_without_faces'] += int(not boxes)
            for _, cls, _, _, _, _ in boxes:
                counts[split]['faces'][str(cls)] += 1
            records.append((split, image_path, label, width, height, digest, boxes))
        # orphan_labels = set(labels_dir.rglob('*.txt')) - used_labels
        orphan_labels = {
            p for p in labels_dir.rglob('*.txt')
            if p.name != 'classes.txt'
        } - used_labels
        if orphan_labels:
            raise ValueError('Labels without matching supported images in %s, e.g. %s'
                             % (split, sorted(orphan_labels)[0]))
        print('Checked %s: %d images, faces=%s' % (split, len(images), counts[split]['faces']))
    return records, counts


def run(args):
    records, counts = prepare(args)
    if args.check_only:
        print('Preflight passed; no output written.')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    for split in args.splits:
        for cls, name in CLASSES.items():
            (args.output / split / ('%d_%s' % (cls, name))).mkdir(parents=True)
    with (args.output / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, (split, source, label, width, height, digest, boxes) in enumerate(records, 1):
            image = load_rgb(source)
            for line_number, cls, norm, box, bounds, clipped in boxes:
                filename = 'image_%06d_face_%04d.png' % (index, line_number)
                relative = Path(split) / ('%d_%s' % (cls, CLASSES[cls])) / filename
                image.crop(tuple(bounds)).save(args.output / relative)
                values = [split, str(source), str(label), line_number, cls, CLASSES[cls],
                          relative.as_posix(), width, height] + norm + box + bounds + [
                              bounds[2] - bounds[0], bounds[3] - bounds[1], args.margin, int(clipped), digest]
                writer.writerow(dict(zip(FIELDS, values)))
    summary = {'dataset': str(args.dataset), 'output': str(args.output),
               'classes': CLASSES, 'margin_per_side': args.margin,
               'resize': None, 'minimum_face_size_filter': None,
               'exif_orientation': 'upright via PIL ImageOps.exif_transpose',
               'split_counts': counts, 'total_crops': sum(sum(c['faces'].values()) for c in counts.values()),
               'note': 'GT crops only. Gaze geometric normalization is not performed.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Saved %d crops to %s' % (summary['total_crops'], args.output))
    print('Class mapping: 0=not looking, 1=looking, 2=uncertain. See manifest.csv and summary.json.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, required=True, help='Root containing images/ and labels/ or split folders')
    parser.add_argument('--output', type=Path, required=True, help='NEW directory for face crops')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val', 'test'])
    parser.add_argument('--margin', type=float, default=0.0, help='Fraction of original box dimension added on each side')
    parser.add_argument('--check-only', action='store_true', help='Validate labels/images and split duplicates without writing')
    args = parser.parse_args()
    args.dataset = args.dataset.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not math.isfinite(args.margin) or args.margin < 0:
        parser.error('--margin must be finite and non-negative')
    if len(set(args.splits)) != len(args.splits):
        parser.error('--splits contains duplicates')
    if not args.dataset.is_dir():
        parser.error('dataset directory does not exist')
    if args.output == args.dataset or args.dataset in args.output.parents or args.output in args.dataset.parents:
        parser.error('output must be separate from the input dataset tree')
    if args.output.exists() and not args.check_only:
        parser.error('output already exists; choose a new output directory to prevent stale crops')
    try:
        run(args)
    except (OSError, ValueError) as exc:
        print('ERROR: %s' % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
