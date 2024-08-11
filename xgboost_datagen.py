from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch.nn as nn
from melanoma_classifier import (HDF5_TEST, HDF5_TRAIN, TARGET_COL,
                                 TEST_METADATA_PATH, TRAIN_METADATA_PATH,
                                 ISICDataModule, ISICModel, get_transforms)
from torch.utils.data import DataLoader


class Config:
    MODEL_NAME = "EfficientNetB1"
    BATCH_SIZE = 64
    CUTOUT_RATIO = 0.25
    NUM_WORKERS = 6
    FOLD = [1,3,5]
    RUN_NAME = "2024_fold_1_3_5"
    BASE_RUN_DIR = Path("runs")
    USE_META_FEATURES = [
        'age_approx',
        'clin_size_long_diam_mm',
        'tbp_lv_A',
        'tbp_lv_Aext',
        'tbp_lv_B',
        'tbp_lv_Bext',
        'tbp_lv_C',
        'tbp_lv_Cext',
        'tbp_lv_H',
        'tbp_lv_Hext',
        'tbp_lv_L',
        'tbp_lv_Lext',
        'tbp_lv_areaMM2',
        'tbp_lv_area_perim_ratio',
        'tbp_lv_color_std_mean',
        'tbp_lv_deltaA',
        'tbp_lv_deltaB',
        'tbp_lv_deltaL',
        'tbp_lv_deltaLBnorm',
        'tbp_lv_eccentricity',
        'tbp_lv_minorAxisMM',
        'tbp_lv_nevi_confidence',
        'tbp_lv_norm_border',
        'tbp_lv_norm_color',
        'tbp_lv_perimeterMM',
        'tbp_lv_radial_color_std_max',
        'tbp_lv_stdL',
        'tbp_lv_stdLExt',
        'tbp_lv_symm_2axis',
        'tbp_lv_symm_2axis_angle',
        'tbp_lv_x',
        'tbp_lv_y',
        'tbp_lv_z',
        'sex',
        'age_approx',
        'site_anterior torso',
        'site_head/neck',
        'site_lower extremity',
        'site_posterior torso',
        'site_upper extremity',
        'site_nan',
        'location_Head & Neck',
        'location_Left Arm',
        'location_Left Arm - Lower',
        'location_Left Arm - Upper',
        'location_Left Leg',
        'location_Left Leg - Lower',
        'location_Left Leg - Upper',
        'location_Right Arm',
        'location_Right Arm - Lower',
        'location_Right Arm - Upper',
        'location_Right Leg',
        'location_Right Leg - Lower',
        'location_Right Leg - Upper',
        'location_Torso Back',
        'location_Torso Back Bottom Third',
        'location_Torso Back Middle Third',
        'location_Torso Back Top Third',
        'location_Torso Front',
        'location_Torso Front Bottom Half',
        'location_Torso Front Top Half',
        'location_Unknown',
        'location_nan',
        'location_simple_Head & Neck',
        'location_simple_Left Arm',
        'location_simple_Left Leg',
        'location_simple_Right Arm',
        'location_simple_Right Leg',
        'location_simple_Torso Back',
        'location_simple_Torso Front',
        'location_simple_Unknown',
        'location_simple_nan'
    ]
    META_DIMS = [
        256,
        128,
        16,
    ]

trainer = L.Trainer(
    devices=[0],
    logger=False
)


class XGBoostModel(ISICModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, image, metadata=None):
        image_features = self.model(image)
        return image_features

    def predict_step(self, batch, batch_idx):
        (image, metadata), labels = batch
        feats = self.forward(image, metadata)
        return feats
    
    def _post_init(self):
        self.model._fc = nn.Identity()
        self.model._swish = nn.Identity()
        


model = XGBoostModel.load_from_checkpoint(
    checkpoint_path="runs/2024_fold_0_2_4/best-val-auc_20-epoch=7-val_auc_20=0.1633.ckpt",
    calculate_metrics=False,
    num_meta_features=len(Config.USE_META_FEATURES),
    meta_network_dim=Config.META_DIMS,
    weight_decay=5e-2,
    model_name=Config.MODEL_NAME,
    weight=4.0,
)
model._post_init()

train_trainsforms, val_transforms = get_transforms(
    img_size=model.image_size,
    cutout_ratio=Config.CUTOUT_RATIO,
)

data_module = ISICDataModule(
    train_hdf5_file=HDF5_TRAIN,
    test_hdf5_file=HDF5_TEST,
    label_col=TARGET_COL,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    image_size=model.image_size,
    # cutout_ratio=Config.CUTOUT_RATIO,
    train_transform=val_transforms,
    # val_transform=val_transforms,
    train_metadata=TRAIN_METADATA_PATH,
    test_metadata=TEST_METADATA_PATH,
    meta_features=Config.USE_META_FEATURES,
    # fold=Config.FOLD,
)

data_module.prepare_data()
data_module.setup(stage="fit")

train_dataloader = DataLoader(
    data_module.train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
    pin_memory=True,
)

preds = trainer.predict(model, train_dataloader) # N, 1280
preds = np.concatenate(preds, axis=0)  # N, 1280

# create df of preds
df_preds = pd.DataFrame(preds, columns=[f"pred_{i}" for i in range(preds.shape[1])])

# add preds features to the metadata
df_train = data_module.train_metadata

df_train = pd.concat([df_train, df_preds], axis=1)

# save the new metadata
df_train.to_csv("train_metadata_with_image_feats.csv", index=False)