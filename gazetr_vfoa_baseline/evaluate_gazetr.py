"""Frozen GazeTR raw-crop baseline: train calibration, fixed-threshold val evaluation."""
import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import random
import sys
from pathlib import Path

NAMES = {0: 'not_looking_at_camera', 1: 'looking_at_camera', 2: 'uncertain'}
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def save_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0])
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def discover(root, split):
    folder = root / split
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    rows = []
    for class_id, name in NAMES.items():
        candidates = [folder / label for label in (str(class_id) + '_' + name, name, str(class_id))]
        found = [p for p in candidates if p.is_dir()]
        if len(found) > 1:
            raise ValueError('Ambiguous class directories: ' + str(found))
        if not found:
            if class_id == 2:
                logging.warning('%s: uncertain directory absent.', split)
                continue
            raise FileNotFoundError('Missing class directory: ' + str(candidates[0]))
        paths = sorted(p for p in found[0].rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not paths and class_id != 2:
            raise ValueError('No images: ' + str(found[0]))
        logging.info('%s | %s: %d faces', split, name, len(paths))
        for path in paths:
            rows.append({'split': split, 'crop_path': str(path.resolve()),
                         'true_class_id': class_id, 'true_class': name})
    return rows


def score_deg(pitch_rad, yaw_rad):
    value = math.cos(pitch_rad) * math.cos(yaw_rad)
    return math.degrees(math.acos(max(-1.0, min(1.0, value))))


def binary_metrics(rows, threshold):
    certain = [row for row in rows if int(row['true_class_id']) in (0, 1)]
    if not certain:
        raise ValueError('No certain samples for binary evaluation.')
    cm = [[0, 0], [0, 0]]  # rows=GT, columns=prediction
    for row in certain:
        cm[int(row['true_class_id'])][int(float(row['score_deg']) <= threshold)] += 1
    per_class = []
    for i in (0, 1):
        tp = cm[i][i]
        fp = cm[1-i][i]
        fn = cm[i][1-i]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class.append({'Class': NAMES[i], 'Precision': precision, 'Recall': recall,
                          'F1': f1, 'Support': tp + fn})
    overall = {'Class': 'Overall (macro)', **{
        key: sum(r[key] for r in per_class) / 2 for key in ('Precision', 'Recall', 'F1')},
        'Support': len(certain)}
    accuracy = (cm[0][0] + cm[1][1]) / len(certain)
    return [overall] + per_class, cm, accuracy


def calibrate(rows, step):
    # Fixed search range covers all scores. Tie: first/smallest threshold wins.
    if not math.isfinite(step) or step < 0.05 or step > 180:
        raise ValueError('threshold_step_deg must be between 0.05 and 180.')
    thresholds = [-1.0] + [i * step for i in range(int(180 / step) + 1)]
    if thresholds[-1] < 180:
        thresholds.append(180.0)
    curve, best = [], None
    for threshold in thresholds:
        metrics, _, accuracy = binary_metrics(rows, threshold)
        row = {'threshold_deg': threshold, 'macro_f1': metrics[0]['F1'],
               'macro_precision': metrics[0]['Precision'], 'macro_recall': metrics[0]['Recall'],
               'accuracy': accuracy}
        curve.append(row)
        if best is None or row['macro_f1'] > best['macro_f1']:
            best = row
    return best, curve


def infer(rows, model, device, cfg, output):
    from gazetr_adapter import predict_batch
    import numpy as np
    result = []
    fields = list(rows[0]) + ['output_0', 'output_1', 'pitch_deg', 'yaw_deg', 'score_deg']
    # Stream results to disk: partial files are diagnostic only, COMPLETE.json marks success.
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for start in range(0, len(rows), cfg['batch_size']):
            group = rows[start:start + cfg['batch_size']]
            predictions = predict_batch(model, device, [r['crop_path'] for r in group])
            for row, raw in zip(group, predictions):
                angles = raw.astype(float)
                if cfg['angle_unit'] == 'degrees':
                    angles = np.deg2rad(angles)
                pitch, yaw = angles if cfg['angle_order'] == 'pitch_yaw' else angles[::-1]
                enriched = dict(row, output_0=float(raw[0]), output_1=float(raw[1]),
                                pitch_deg=math.degrees(pitch), yaw_deg=math.degrees(yaw),
                                score_deg=score_deg(pitch, yaw))
                writer.writerow(enriched)
                result.append(enriched)
            stream.flush()
            logging.info('%s: %d/%d', rows[0]['split'], len(result), len(rows))
    return result


def report(rows, threshold, out, split):
    tagged = [dict(row, predicted_class_id=int(row['score_deg'] <= threshold),
                   predicted_class=NAMES[int(row['score_deg'] <= threshold)],
                   included_in_binary_metrics=int(row['true_class_id'] in (0, 1))) for row in rows]
    save_csv(out / (split + '_predictions.csv'), tagged)
    metrics, cm, accuracy = binary_metrics(rows, threshold)
    save_csv(out / (split + '_metrics.csv'), metrics)
    save_csv(out / (split + '_confusion_matrix.csv'), [
        {'True_class': NAMES[i], NAMES[0]: cm[i][0], NAMES[1]: cm[i][1]} for i in (0, 1)])
    uncertain = [r for r in tagged if r['true_class_id'] == 2]
    counts = {NAMES[i]: sum(r['predicted_class_id'] == i for r in uncertain) for i in (0, 1)}
    summary = {'split': split, 'threshold_deg': threshold, 'binary_accuracy': accuracy,
               'binary_faces': metrics[0]['Support'], 'all_faces': len(rows),
               'uncertain_faces': len(uncertain), 'uncertain_prediction_counts': counts,
               'note': 'Uncertain predictions are a distribution, not uncertainty recognition accuracy.'}
    save_json(out / (split + '_summary.json'), summary)
    logging.info('%s BINARY results (GT classes 0/1 only), threshold=%.3f deg', split, threshold)
    logging.info('%-28s %10s %10s %10s %8s', 'Class', 'Precision', 'Recall', 'F1', 'Support')
    for row in metrics:
        logging.info('%-28s %10.3f %10.3f %10.3f %8d', row['Class'], row['Precision'],
                     row['Recall'], row['F1'], row['Support'])
    logging.info('Accuracy: %.4f; uncertain faces: %d; uncertain assigned counts: %s',
                 accuracy, len(uncertain), counts)
    return summary


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config_gazetr_eval.yaml')
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    for key in ('repo', 'weights', 'data', 'output'):
        path = Path(cfg[key]).expanduser()
        cfg[key] = str((config_path.parent / path).resolve() if not path.is_absolute() else path.resolve())
    if cfg['calibration_split'] != 'train' or cfg['evaluation_split'] not in ('val', 'test'):
        raise ValueError('Calibrate on train; evaluate on val or test. Never tune on test.')
    if cfg['angle_order'] not in ('pitch_yaw', 'yaw_pitch') or cfg['angle_unit'] not in ('radians', 'degrees'):
        raise ValueError('Invalid angle convention.')
    if not isinstance(cfg['batch_size'], int) or cfg['batch_size'] < 1:
        raise ValueError('batch_size must be positive integer.')
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(out / 'run.log', encoding='utf-8')])
    save_json(out / 'config_resolved.json', cfg)
    try:
        run(cfg, out)
    except Exception:
        logging.exception('FAILED: partial outputs are not complete evaluation results.')
        raise


def run(cfg, out):
    import numpy as np
    import torch
    import cv2
    from gazetr_adapter import load_model
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logging.info('Frozen GazeTR, raw GT crops, binary evaluation; no geometric normalization.')
    root = Path(cfg['data'])
    train = discover(root, 'train')
    split = cfg['evaluation_split']
    evaluation = discover(root, split)
    # This catches byte-identical crops, not every source/person/near-duplicate leak.
    train_hashes = {sha256(row['crop_path']) for row in train}
    for row in evaluation:
        if sha256(row['crop_path']) in train_hashes:
            raise ValueError('Identical crop appears in both splits: ' + row['crop_path'])
    model, device = load_model(cfg['repo'], cfg['weights'], cfg['device'])
    save_json(out / 'provenance.json', {
        'weights_sha256': sha256(cfg['weights']), 'model_py_sha256': sha256(Path(cfg['repo']) / 'model.py'),
        'python': platform.python_version(), 'torch': torch.__version__, 'opencv': cv2.__version__,
        'preprocessing': 'cv2 BGR; direct resize224 INTER_LINEAR; CHW float/255; no mean/std normalization',
        'score': 'degrees(acos(clip(cos(pitch)*cos(yaw), -1, 1)))',
        'interpretation': 'Raw-crop transfer score; not validated physical camera-angle or angular error.',
        'split_check': 'Byte-identical cross-split crop check only; split by original source image upstream.'})
    train_predictions = infer(train, model, device, cfg, out / 'train_angles.csv')
    best, curve = calibrate(train_predictions, float(cfg['threshold_step_deg']))
    threshold = best['threshold_deg']
    save_csv(out / 'train_threshold_search.csv', curve)
    save_json(out / 'threshold.json', dict(best, calibration_split='train',
        objective='binary macro F1 on GT classes 0 and 1', tie_break='smallest threshold',
        prediction_rule='looking iff score_deg <= threshold_deg'))
    logging.info('Threshold fixed using train only: %.3f deg', threshold)
    report(train_predictions, threshold, out, 'train')
    predictions = infer(evaluation, model, device, cfg, out / (split + '_angles.csv'))
    summary = report(predictions, threshold, out, split)
    save_json(out / 'COMPLETE.json', {'status': 'complete', 'evaluation': summary})
    logging.info('Finished. Results: %s', out)


if __name__ == '__main__':
    main()
