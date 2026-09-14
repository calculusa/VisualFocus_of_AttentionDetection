from PIL import Image, ImageOps
import torch.nn as nn
from torchvision import models, transforms


CLASS_NAMES = [
    "not_looking_at_camera",
    "looking_at_camera",
    "uncertain",
]

CLASS_FOLDERS = [
    "0_not_looking_at_camera",
    "1_looking_at_camera",
    "2_uncertain",
]

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224


class ResizePad:
    """等比例缩放并补边到 224×224，保留完整人脸。"""

    def __call__(self, image):
        resampling = getattr(Image, "Resampling", Image)

        return ImageOps.pad(
            image.convert("RGB"),
            (IMAGE_SIZE, IMAGE_SIZE),
            method=resampling.BILINEAR,
            color=(124, 116, 104),
            centering=(0.5, 0.5),
        )


def make_transform(training=False):
    operations = [ResizePad()]

    if training:
        # 水平翻转不改变 looking / not looking / uncertain 标签。
        operations.append(
            transforms.RandomHorizontalFlip(p=0.5)
        )

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(operations)


def build_model(pretrained=True):
    """加载 ResNet18，并将输出层改为三分类。"""

    weights = (
        models.ResNet18_Weights.IMAGENET1K_V1
        if pretrained else None
    )

    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES,
    )

    return model