"""Load the user's existing official GazeTR Model; no architecture changes."""
import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import torch


def load_model(repo, weights, device):
    repo = Path(repo).expanduser().resolve()
    source = repo / 'model.py'
    if not source.is_file():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location('vfoa_official_gazetr', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device(device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable.')
        torch.cuda.set_device(device)
    state = torch.load(str(weights), map_location='cpu', weights_only=True)
    if isinstance(state, Mapping):
        for key in ('state_dict', 'model_state', 'model_state_dict', 'model'):
            if key in state and isinstance(state[key], Mapping):
                state = state[key]
                break
    if not isinstance(state, Mapping) or not state or not all(
            isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a tensor state_dict checkpoint.')
    clean = {}
    for key, value in state.items():
        name = key[7:] if key.startswith('module.') else key
        if name in clean:
            raise ValueError('Duplicate checkpoint key: ' + name)
        clean[name] = value
    model = module.Model()
    model.load_state_dict(clean, strict=True)
    return model.to(device).eval(), device


def preprocess(path):
    # Matches provided reader's OpenCV channel order + ToTensor scaling.
    # Resize is additional adaptation for arbitrary WIDER FACE crop dimensions.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot read image: ' + str(path))
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    array = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(array).float().div_(255.0)


def predict_batch(model, device, paths):
    batch = torch.stack([preprocess(path) for path in paths]).to(device)
    with torch.inference_mode():
        result = model({'face': batch})
    if not isinstance(result, torch.Tensor) or tuple(result.shape) != (len(paths), 2):
        raise ValueError('Expected GazeTR output shape (batch, 2).')
    if not torch.isfinite(result).all():
        raise ValueError('Non-finite GazeTR output; evaluation stopped.')
    return result.cpu().numpy()
