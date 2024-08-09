from pathlib import Path

HDF5_TRAIN = Path("train-image.hdf5")
HDF5_TEST = Path("test-image.hdf5")
TRAIN_METADATA_PATH = Path("train-metadata.csv")
TEST_METADATA_PATH = Path("test-metadata.csv")
TARGET_COL = "target"
MEAN_IMG = Path("mean_2024.npy")
STD_IMG = Path("std_2024.npy")
FOLD_PATH = Path("folds.json")
ISIC_ID_COLUMN = "isic_id"
