import io
from pathlib import Path
from typing import List, Optional

import albumentations as A
import cv2
import h5py
import lightning as L
import numpy as np
import pandas as pd
import torch
from ..data import FOLD_PATH, get_df
from ..data.constants import ISIC_ID_COLUMN
from torch.utils.data import DataLoader, Dataset


class ISICDataset(Dataset):
    def __init__(
        self,
        hdf5_filepath: str,
        metadata: pd.DataFrame,
        labels: Optional[np.ndarray]=None,
        transform: Optional[A.Compose] = None,
        mode: str = "train",
        meta_features: Optional[List[str]] = None,
    ):
        self.hdf5 = h5py.File(hdf5_filepath)
        self.labels = labels
        self.transform = transform
        self.metadata = metadata
        self.mode = mode
        self.meta_features = meta_features
        if isinstance(self.meta_features, list) and len(self.meta_features) == 0:
            self.meta_features = None

        self.use_metadata = False if self.meta_features is None else True

        self.subjects = metadata[ISIC_ID_COLUMN].values

        if self.mode == "train":
            assert (
                self.labels.shape[0] == self.metadata.shape[0]
            ), f"Labels and dataset size do not match: {self.labels.shape[0]} != {self.metadata.shape[0]}"

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        metadata = None
        buffer = self.hdf5[self.subjects[idx]][()]
        image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        
        if self.use_metadata:
            metadata = torch.tensor(
                self.metadata.iloc[idx][self.meta_features].values.astype(np.float32)
            ).float()

        if self.mode == "train":
            label = self.labels[idx]
            return (image, metadata), label
        else:
            return ((image, metadata), torch.tensor(0).float())


class ISICDataModule(L.LightningDataModule):

    def __init__(
        self,
        train_hdf5_file: Path,
        test_hdf5_file: Path,
        batch_size: int,
        num_workers: int,
        image_size: int,
        train_metadata: Path,
        test_metadata: Path,
        train_transform: Optional[A.Compose] = None,
        val_transform: Optional[A.Compose] = None,
        meta_features: Optional[List[str]] = None,
        labels: Optional[np.ndarray] = None,
        label_col: Optional[str] = None,
        fold: Optional[int | List[int]] = None,
        fold_path: Optional[Path] = FOLD_PATH,
    ):
        super().__init__()
        self.train_hdf5_file = train_hdf5_file
        self.test_hdf5_file = test_hdf5_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        self.meta_features = meta_features if meta_features is not None else []
        self.fold = [fold] if isinstance(fold, int) else fold
        self.fold_path = fold_path

        if labels is None and label_col is None:
            raise ValueError("Either labels or label_col must be provided.")

        self.labels = None
        self.label_col = label_col

        if labels is not None:
            self.labels = labels

        if self.label_col is not None and self.labels is None:
            self.labels = pd.read_csv(self.train_metadata)[self.label_col].values

    def prepare_data(self) -> None:
        (
            df_train,
            df_test,
            _,
            _,
            self.possible_meta_features,
            self.possible_n_meta_features,
        ) = get_df(
            use_meta=len(self.meta_features) > 0,
            train_metadata_path=self.train_metadata,
            test_metadata_path=self.test_metadata,
            train_hdf5=self.train_hdf5_file,
            test_hdf5=self.test_hdf5_file,
            fold_path=self.fold_path,
        )

        if self.fold is not None:
            self.train_metadata = df_train[~df_train["fold"].isin(self.fold)]
            self.train_labels = self.labels[~df_train["fold"].isin(self.fold)]
            self.val_metadata = df_train[df_train["fold"].isin(self.fold)]
            self.val_labels = self.labels[df_train["fold"].isin(self.fold)]
            self.test_metadata = df_test
        else:
            self.train_metadata = df_train
            self.val_metadata = None
            self.test_metadata = df_test

        # make sure desired meta features are in possible meta features
        assert all(
            [feat in self.possible_meta_features for feat in self.meta_features]
        ), (
            f"Meta feature not found in possible meta features: "
            f"{[feat for feat in self.meta_features if feat not in self.possible_meta_features]}"
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_dataset = ISICDataset(
                hdf5_filepath=self.train_hdf5_file,
                labels=(
                    self.train_labels if hasattr(self, "train_labels") else self.labels
                ),
                transform=self.train_transform,
                metadata=self.train_metadata,
                mode="train",
                meta_features=self.meta_features,
            )

            if self.val_metadata is not None:
                self.val_dataset = ISICDataset(
                    hdf5_filepath=self.train_hdf5_file,
                    labels=self.val_labels,
                    transform=self.val_transform,
                    metadata=self.val_metadata,
                    mode="train",
                    meta_features=self.meta_features,
                )
            else:
                self.val_dataset = None

        if stage == "test":
            self.test_dataset = ISICDataset(
                hdf5_filepath=self.test_hdf5_file,
                labels=None,
                transform=self.val_transform,
                metadata=self.test_metadata,
                mode="test",
                meta_features=self.meta_features,
            )
        
        if stage == "validate":
            raise NotImplementedError("Validation dataset not implemented yet.")
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
