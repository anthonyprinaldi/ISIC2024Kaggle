from pathlib import Path

import lightning as L
from data.constants import (HDF5_TEST, HDF5_TRAIN, TARGET_COL,
                            TEST_METADATA_PATH, TRAIN_METADATA_PATH)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from modeling import ISICDataModule, ISICModel, get_transforms


class Config:
    MODEL_NAME = "EfficientNetB1"
    BATCH_SIZE = 64
    CUTOUT_RATIO = 0.25
    NUM_WORKERS = 4
    FOLD = [0,2,4]
    RUN_NAME = "2024_fold_0_2_4"
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

callbacks = [
    ModelCheckpoint(
        dirpath=Config.BASE_RUN_DIR / Config.RUN_NAME,
        monitor="train_loss",
        verbose=True,
        filename="best-train-loss-{epoch}-{train_loss:.4f}",
    ),
    ModelCheckpoint(
        dirpath=Config.BASE_RUN_DIR / Config.RUN_NAME,
        monitor="val_loss",
        verbose=True,
        filename="best-val-loss-{epoch}-{val_loss:.4f}",
    ),
    ModelCheckpoint(
        dirpath=Config.BASE_RUN_DIR / Config.RUN_NAME,
        monitor="val_auc_20",
        verbose=True,
        filename="best-val-auc_20-{epoch}-{val_auc_20:.4f}",
        mode="max",
    ),
]

lightning_logger = TensorBoardLogger(
    Config.BASE_RUN_DIR / Config.RUN_NAME / "tensorboard",
)

trainer = L.Trainer(
    default_root_dir=Config.BASE_RUN_DIR / Config.RUN_NAME,
    max_epochs=10,
    accelerator="auto",
    devices=[0],
    accumulate_grad_batches=1,
    check_val_every_n_epoch=1,
    callbacks=callbacks,
    precision="16-mixed",
    gradient_clip_algorithm="value",
    gradient_clip_val=0.5,
    logger=lightning_logger,
    benchmark=False,
    deterministic=True,
    enable_progress_bar=True,
    log_every_n_steps=1,
    val_check_interval=0.1,
    num_sanity_val_steps=0,
)

model = ISICModel(
    calculate_metrics=True,
    num_meta_features=len(Config.USE_META_FEATURES),
    meta_network_dim=Config.META_DIMS,
    weight_decay=5e-2,
    model_name=Config.MODEL_NAME,
    weight=4.0,
)

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
    cutout_ratio=Config.CUTOUT_RATIO,
    train_transform=train_trainsforms,
    val_transform=val_transforms,
    train_metadata=TRAIN_METADATA_PATH,
    test_metadata=TEST_METADATA_PATH,
    meta_features=Config.USE_META_FEATURES,
    fold=Config.FOLD,
)

trainer.fit(model, data_module)