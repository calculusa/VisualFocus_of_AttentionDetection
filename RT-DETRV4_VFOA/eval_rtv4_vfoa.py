"""Run from the RT-DETRv4 root. Evaluate trusted local checkpoints only."""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def iou_matrix(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    a = np.column_stack((a[:, :2], a[:, :2] + a[:, 2:]))
    b = np.column_stack((b[:, :2], b[:, :2] + b[:, 2:]))
    wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]))
    inter = wh.prod(2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def confusion(gt, pred, image_ids, nclasses, threshold, iou):
    # Rows=true, columns=predicted; last index=background/unmatched.
    cm = np.zeros((nclasses + 1, nclasses + 1), dtype=np.int64)
    for image_id in image_ids:
        g = gt[image_id]
        p = sorted((x for x in pred[image_id] if x['score'] >= threshold), key=lambda x: -x['score'])
        overlaps = iou_matrix([x['bbox'] for x in p], [x['bbox'] for x in g])
        used = set()
        for j, det in enumerate(p):
            candidates = [k for k in range(len(g)) if k not in used and overlaps[j, k] >= iou]
            if candidates:
                k = max(candidates, key=lambda k: overlaps[j, k])
                used.add(k)
                cm[g[k]['category_id'], det['category_id']] += 1
            else:
                cm[nclasses, det['category_id']] += 1
        for k, target in enumerate(g):
            if k not in used:
                cm[target['category_id'], nclasses] += 1
    return cm


def scores(cm):
    tp = np.diag(cm)[:-1].astype(float)
    predicted = cm[:, :-1].sum(0)
    support = cm[:-1, :].sum(1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=precision + recall > 0)
    return precision, recall, f1


def infer(args):
    import torch
    sys.path.insert(0, str(Path.cwd()))
    from engine.core import YAMLConfig
    from torch.utils.data import DataLoader
    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    if cfg.yaml_cfg.get('num_classes') != 3 or cfg.yaml_cfg.get('remap_mscoco_category', False):
        raise ValueError('This evaluator requires num_classes=3 and remap_mscoco_category=False')
    ds_cfg = cfg.yaml_cfg['val_dataloader']['dataset']
    if args.images:
        ds_cfg['img_folder'] = args.images
    if args.ann:
        ds_cfg['ann_file'] = args.ann
    ann_path = ds_cfg['ann_file']
    # Uses the configured validation transforms even when selecting a train threshold.
    dataset = cfg.val_dataloader.dataset
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=lambda batch: tuple(zip(*batch)))
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
    ema = checkpoint.get('ema')
    state = ema['module'] if ema is not None else checkpoint['model']
    model = cfg.model
    model.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()
    # Older upstream encoder caches ordinary CPU tensors, not registered buffers.
    for module in model.modules():
        for name, value in list(vars(module).items()):
            if name.startswith('pos_embed') and torch.is_tensor(value):
                setattr(module, name, value.to(args.device))
    post = cfg.postprocessor.to(args.device).eval()
    results = []
    with torch.inference_mode():
        for step, (images, targets) in enumerate(loader):
            images = torch.stack(list(images)).to(args.device)
            sizes = torch.stack([t['orig_size'] for t in targets]).to(args.device)
            outputs = post(model(images), sizes)
            for target, out in zip(targets, outputs):
                image_id = int(target['image_id'].item())
                for label, box, score in zip(out['labels'].cpu().tolist(), out['boxes'].cpu().tolist(), out['scores'].cpu().tolist()):
                    x1, y1, x2, y2 = box
                    results.append(dict(image_id=image_id, category_id=int(label),
                                        bbox=[x1, y1, x2-x1, y2-y1], score=float(score)))
            if step % 20 == 0:
                print(f'Inference {step+1}/{len(loader)}', flush=True)
    return ann_path, results, int(checkpoint.get('last_epoch', -1)), 'ema' if ema is not None else 'model'


def coco_eval(ann_path, predictions, outdir, names):
    try:
        from faster_coco_eval import COCO, COCOeval_faster as COCOeval
    except ImportError:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    coco = COCO(str(ann_path))
    coco.dataset.setdefault('info', {})
    if predictions:
        detections = coco.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = dict(images=coco.dataset['images'], categories=coco.dataset['categories'], annotations=[])
        detections.createIndex()
    ev = COCOeval(coco, detections, 'bbox')
    ev.params.catIds = list(range(len(names)))
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    keys = ['mAP50_95', 'mAP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
            'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
    overall = dict(zip(keys, map(float, ev.stats[:12])))
    pr = ev.eval['precision']  # IoU, recall, category, area, maxDets
    rows = []
    def avg(v):
        v = v[v >= 0]
        return float(v.mean()) if len(v) else None
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, name in enumerate(names):
        rows.append(dict(class_name=name, AP50_95=avg(pr[:, :, k, 0, -1]), AP50=avg(pr[0, :, k, 0, -1])))
        curve = pr[0, :, k, 0, -1]
        ax.plot(ev.params.recThrs, np.maximum(curve, 0), label=name)
    ax.set(xlabel='Recall', ylabel='Interpolated precision', title='COCO PR at IoU=0.50, maxDets=100', xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / 'BoxPR_curve.png', dpi=180); plt.close(fig)
    save_json(outdir / 'coco_metrics.json', dict(overall=overall, per_class=rows))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-c', '--config', required=True)
    p.add_argument('-w', '--weights', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    p.add_argument('--images'); p.add_argument('--ann')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--iou', type=float, default=0.5)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--conf', type=float)
    group.add_argument('--threshold-file')
    group.add_argument('--select-threshold', action='store_true')
    args = p.parse_args()
    if args.split != 'val' and not (args.ann and args.images):
        p.error('train/test require both --ann and --images')
    if args.select_threshold and args.split != 'train':
        p.error('Threshold selection is restricted to train; freeze it for val/test')
    if not 0 < args.iou <= 1 or (args.conf is not None and not 0 <= args.conf <= 1):
        p.error('Invalid IoU or confidence')
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / 'summary.json').exists():
        p.error('Output already contains a completed evaluation; use a new --out')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    weight_hash = sha256(args.weights)
    ann_path, predictions, epoch, weight_source = infer(args)
    data = json.loads(Path(ann_path).read_text())
    cats = sorted(data['categories'], key=lambda x: x['id'])
    if [x['id'] for x in cats] != [0, 1, 2]:
        raise ValueError('Expected COCO category IDs 0,1,2')
    names = [x['name'] for x in cats]
    ids = [x['id'] for x in data['images']]
    gt, pred = defaultdict(list), defaultdict(list)
    for item in data['annotations']:
        if item.get('iscrowd', 0) or item.get('ignore', 0):
            raise ValueError('Custom F1 does not support crowd/ignored GT; use a non-crowd VFOA dataset')
        gt[item['image_id']].append(item)
    for item in predictions:
        if item['category_id'] not in (0, 1, 2):
            raise ValueError('Unexpected predicted class')
        pred[item['image_id']].append(item)
    save_json(outdir / 'predictions_coco.json', predictions)
    # AP uses ALL official top-300 outputs, before confidence filtering; no added NMS.
    ap_rows = coco_eval(ann_path, predictions, outdir, names)
    thresholds = np.linspace(0, 1, 201)
    curves = np.array([scores(confusion(gt, pred, ids, 3, float(t), args.iou)) for t in thresholds])
    if args.select_threshold:
        conf = float(thresholds[np.argmax(curves[:, 2, :].mean(1))])
        save_json(outdir / 'threshold.json', dict(conf=conf, iou=args.iou, source_split='train',
                  criterion='macro_f1', weight_sha256=weight_hash, categories=names,
                  annotation_sha256=sha256(ann_path), grid_step=0.005, tie_break='lowest threshold'))
    elif args.threshold_file:
        t = json.loads(Path(args.threshold_file).read_text())
        if t['weight_sha256'] != weight_hash or t['categories'] != names or t['iou'] != args.iou:
            raise ValueError('Threshold file does not match weights/categories/IoU')
        conf = float(t['conf'])
    else:
        conf = args.conf
    cm = confusion(gt, pred, ids, 3, conf, args.iou)
    precision, recall, f1 = scores(cm)
    rows = []
    for k, name in enumerate(names):
        rows.append(dict(class_name=name, support=int(cm[k].sum()), TP=int(cm[k,k]),
                         FP=int(cm[:,k].sum()-cm[k,k]), FN=int(cm[k].sum()-cm[k,k]),
                         precision=float(precision[k]), recall=float(recall[k]), f1=float(f1[k]),
                         **{key: ap_rows[k][key] for key in ('AP50', 'AP50_95')}))
    with open(outdir / 'per_class_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    labels = names + ['background']
    with open(outdir / 'confusion_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['true / predicted'] + labels)
        for name, row in zip(labels, cm): writer.writerow([name] + row.tolist())
    for normalized in (False, True):
        matrix = cm.astype(float)
        if normalized:
            matrix = np.divide(matrix, matrix.sum(1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(1, keepdims=True)>0)
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(matrix, cmap='Blues'); fig.colorbar(im, ax=ax)
        ax.set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels,
               xlabel='Predicted', ylabel='True', title=f'IoU={args.iou}, confidence={conf:.3f}')
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{matrix[i,j]:.2f}' if normalized else str(cm[i,j]), ha='center', va='center', color='white' if matrix[i,j] > matrix.max()/2 else 'black')
        fig.tight_layout(); fig.savefig(outdir / ('confusion_matrix_normalized.png' if normalized else 'confusion_matrix.png'), dpi=180); plt.close(fig)
    with open(outdir / 'threshold_curve.csv', 'w', newline='') as f:
        writer=csv.writer(f); writer.writerow(['confidence','macro_precision','macro_recall','macro_f1'])
        for t, values in zip(thresholds, curves): writer.writerow([t]+values.mean(1).tolist())
    for j, metric in enumerate(['P','R','F1']):
        fig, ax = plt.subplots(figsize=(7,5))
        for k, name in enumerate(names): ax.plot(thresholds, curves[:,j,k], label=name)
        ax.plot(thresholds, curves[:,j,:].mean(1), 'k', linewidth=2, label='macro average')
        ax.axvline(conf, color='gray', linestyle='--', label=f'fixed confidence={conf:.3f}')
        ax.set(xlabel='Confidence threshold', ylabel=metric, ylim=(0,1.02), xlim=(0,1))
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(outdir / f'Box{metric}_curve.png', dpi=180); plt.close(fig)
    summary = dict(split=args.split, images=len(ids), instances=len(data['annotations']), epoch_index=epoch,
                   weight_source=weight_source, weight_sha256=weight_hash, annotation_sha256=sha256(ann_path),
                   confidence=conf, iou=args.iou, macro_precision=float(precision.mean()),
                   macro_recall=float(recall.mean()), macro_f1=float(f1.mean()), per_class=rows,
                   matching='confidence-descending predictions; best unused GT IoU; class-agnostic spatial matching',
                   ap_protocol='COCO maxDets=100; all official top-300 outputs; no extra NMS',
                   confusion_rows='true', confusion_columns='predicted', arguments=vars(args))
    save_json(outdir / 'summary.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Saved to {outdir.resolve()}')


if __name__ == '__main__':
    main()
