import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple
import numpy as np


def get_transforms(
        img_size: int,
        cutout_ratio: float,
        ) -> Tuple[A.Compose, A.Compose]:
    
    train_trainsforms = A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([
            A.GaussianBlur(p=0.75), # TODO: figure out GaussNoise
            A.GaussNoise(p=0.75, var_limit=(5., 30.)),
        ]),
        A.OneOf([
            A.GridDistortion(),
            A.ElasticTransform(),
        ], p=0.75),
        A.CLAHE(clip_limit=4.0, p=0.75),
        A.Resize(img_size, img_size),
        # A.CoarseDropout(max_holes=1, max_height=int(img_size * cutout_ratio), max_width=int(img_size * cutout_ratio), p=0.75),
        A.Normalize(
            # mean=np.load("../2024/mean_2024.npy").tolist(),
            # std=np.load("../2024/std_2024.npy").tolist(),
        ),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            # mean=np.load("../2024/mean_2024.npy").tolist(),
            # std=np.load("../2024/std_2024.npy").tolist(),
        ),
        ToTensorV2(),
    ])

    return train_trainsforms, val_transforms