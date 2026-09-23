"""Run from RT-DETRv4 root: python check_vfoa_coco.py configs/rtv4/rtv4_s_vfoa.yml"""
import sys
import json
import math
from pathlib import Path
from collections import Counter
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
expected = {0: 'not_looking_at_camera', 1: 'looking_at_camera', 2: 'uncertain'}
paths = {}
for split in ['train', 'val']:
    spec = cfg[split + '_dataloader']['dataset']
    root = Path(spec['img_folder'])
    data = json.loads(Path(spec['ann_file']).read_text())
    cats = {x['id']: x['name'] for x in data['categories']}
    if cats != expected or len(data['categories']) != 3:
        raise ValueError('Expected category mapping %r; found %r. Do not relabel silently.' % (expected, cats))
    images = {x['id']: x for x in data['images']}
    assert len(images) == len(data['images']), 'Duplicate image IDs'
    assert len({x['id'] for x in data['annotations']}) == len(data['annotations']), 'Duplicate annotation IDs'
    paths[split] = set()
    for item in images.values():
        path = root / item['file_name']
        assert path.is_file(), 'Missing image: ' + str(path)
        assert item['width'] > 0 and item['height'] > 0
        paths[split].add(path.resolve())
    counts = Counter()
    for a in data['annotations']:
        assert a['image_id'] in images and a['category_id'] in expected
        im = images[a['image_id']]
        x, y, w, h = a['bbox']
        assert all(math.isfinite(v) for v in [x,y,w,h])
        assert x >= 0 and y >= 0 and w > 0 and h > 0, a
        assert x+w <= im['width']+0.01 and y+h <= im['height']+0.01, a
        assert 'area' in a and math.isfinite(a['area']) and a['area'] > 0, 'Missing/invalid area'
        assert a.get('iscrowd', 0) == 0, 'Crowd annotation requires separate review'
        counts[a['category_id']] += 1
    print(split, 'images:', len(images), 'annotations:', len(data['annotations']), 'classes:', dict(sorted(counts.items())))
assert not paths['train'].intersection(paths['val']), 'Same image path occurs in train and val'
assert (Path(cfg['teacher_model']['dinov3_repo_path'])/'hubconf.py').is_file(), 'Missing DINOv3 repository'
assert Path(cfg['teacher_model']['dinov3_weights_path']).is_file(), 'Missing DINOv3 weights'
print('PASS. This checks paths/annotations, not near-duplicates, split provenance or checkpoint validity.')
