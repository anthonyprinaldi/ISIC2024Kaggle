import json
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .constants import (FOLD_PATH, HDF5_TEST, HDF5_TRAIN, TEST_METADATA_PATH,
                        TRAIN_METADATA_PATH)
from .metadata_columns import categorical_columns, numeric_columns


def get_scaler(
    df_train: pd.DataFrame, numeric_columns: List[str] = numeric_columns
) -> MinMaxScaler:
    """Get MinMaxScaler for numeric columns.

    :param df_train: train meta data
    :type df_train: pd.DataFrame
    :return: Scaler to be used by preprocessing function
    :rtype: MinMaxScaler
    """
    scaler = MinMaxScaler()
    scaler.fit(df_train[numeric_columns])
    return scaler


def process_dataset(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    numeric_scaler: MinMaxScaler,
    categorical_columns: List[str] = categorical_columns,
    numeric_columns: List[str] = numeric_columns,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Process training and testing dataframe by encoding categorical
    columns and normalizing numeric columns.

    :param df_train: train metadata
    :type df_train: pd.DataFrame
    :param df_test: test metadata
    :type df_test: pd.DataFrame
    :param categorical_columns: columns to perform one-hot encoding,
        defaults to categorical_columns
    :type categorical_columns: List[str], optional
    :param numeric_columns: columns to perform scaling,
        defaults to numeric_columns
    :type numeric_columns: List[str], optional
    :return: transformed dataframes and list of features to use
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, List[str]]
    """
    # one hot encoding for categorical columns
    # remove sex from categorical columns
    dummy_columns = [col for col in categorical_columns if col != "sex"]
    # combine train and test data to avoid mismatch in columns
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    dummies = pd.get_dummies(
        df[dummy_columns],
        columns=dummy_columns,
        dummy_na=True,
        dtype=np.uint8,
        prefix=[
            "site",
            "location",
            "location_simple",
        ],
    )

    # drop original columns
    df = df.drop(columns=dummy_columns)
    df = pd.concat([df, dummies], axis=1)

    # separate back to train and test
    df_train = df.iloc[: df_train.shape[0]].reset_index(drop=True)
    df_test = df.iloc[df_train.shape[0] :].reset_index(drop=True)

    # sex column
    df_train["sex"] = df_train["sex"].map({"male": 1, "female": 0})
    df_train["sex"] = df_train["sex"].fillna(-1)
    df_test["sex"] = df_test["sex"].map({"male": 1, "female": 0})
    df_test["sex"] = df_test["sex"].fillna(-1)

    # scale numeric columns
    df_train[numeric_columns] = numeric_scaler.transform(df_train[numeric_columns])
    df_test[numeric_columns] = numeric_scaler.transform(df_test[numeric_columns])

    # fill numeric NA columns with -1
    for col in numeric_columns:
        df_train[col] = df_train[col].fillna(-1)
        df_test[col] = df_test[col].fillna(-1)

    # count images per patient
    # df_train["n_images"] = df_train["patient_id"].map(df_train["patient_id"].value_counts())
    # df_test["n_images"] = df_test["patient_id"].map(df_test["patient_id"].value_counts())

    meta_features = numeric_columns + ["sex", "age_approx"] + list(dummies.columns)

    return df_train, df_test, meta_features


def get_df(
    use_meta: bool,
    train_metadata_path: Path = TRAIN_METADATA_PATH,
    test_metadata_path: Path = TEST_METADATA_PATH,
    train_hdf5: Path = HDF5_TRAIN,
    test_hdf5: Path = HDF5_TEST,
    fold_path: Path = FOLD_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, h5py.File, h5py.File, List[str], int, int]:
    """Load in the data, preprocess, and return the dataframes.

    :param use_meta: Whether to use metadata, defaults to True
    :type use_meta: bool
    :param train_metadata_path: Name of train metadata csv,
        defaults to TRAIN_METADATA_PATH
    :type train_metadata_path: Path, optional
    :param test_metadata_path: Name of test metadata csv,
        defaults to TEST_METADATA_PATH
    :type test_metadata_path: Path, optional
    :param train_hdf5: Name of train hdf5,
        defaults to HDF5_TRAIN
    :type train_hdf5: Path, optional
    :param test_hdf5: Name of test hdf5,
        defaults to HDF5_TEST
    :type test_hdf5: Path, optional
    :param fold_path: Name of fold json file,
        defaults to FOLD_PATH
    :type fold_path: Path, optional
    :return: Dataframes, hdf5 files, meta features, number of meta features
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, h5py.File, h5py.File, List[str], int]
    """

    # load metadata
    df_train = pd.read_csv(train_metadata_path)
    df_test = pd.read_csv(test_metadata_path)

    # load hdf5 files
    df_train_hdf5 = h5py.File(train_hdf5)
    df_test_hdf5 = h5py.File(test_hdf5)

    if use_meta:
        df_train, df_test, meta_features = process_dataset(
            df_train=df_train,
            df_test=df_test,
            numeric_scaler=get_scaler(df_train=df_train),
        )
        n_meta_features = len(meta_features)
    else:
        meta_features = None
        n_meta_features = 0

    # add fold column
    with open(fold_path, "r") as f:
        folds = json.load(f)

    flip_folds = {}
    for fold_num, ids in folds.items():
        for id in ids:
            flip_folds[id] = int(fold_num)
    
    df_train["fold"] = df_train["isic_id"].map(flip_folds)

    return (
        df_train,
        df_test,
        df_train_hdf5,
        df_test_hdf5,
        meta_features,
        n_meta_features,
    )
