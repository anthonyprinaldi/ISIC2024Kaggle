import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from constants import TRAIN_METADATA_PATH, TARGET_COL
from sklearn.model_selection import StratifiedGroupKFold


def main():
    NUM_FOLDS = 15
    train_metadata = pd.read_csv(TRAIN_METADATA_PATH)

    patient_id_to_num_pics = train_metadata.groupby("patient_id").size()

    labels = train_metadata["target"]

    # make groups of patient IDS to use in the folds
    patient_id_to_identifier = {
        x: i for i, x in enumerate(patient_id_to_num_pics.index)
    }

    train_metadata["grouping"] = np.zeros(len(train_metadata))
    train_metadata["grouping"] = train_metadata["patient_id"].apply(
        lambda x: patient_id_to_identifier[x]
    )

    # create 3x5 gird for histogram plots
    fig, axs = plt.subplots(3, 5, figsize=(20, 10))
    axs = axs.flatten()

    k = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=25)
    folds = {}
    for i, (_, test_idx) in enumerate(
        k.split(train_metadata, labels, train_metadata["grouping"])
    ):
        folds[i] = train_metadata.iloc[test_idx]["isic_id"].values.tolist()
        # add a histogram of the number of pictures per patient in the test fold
        patient_ids = train_metadata.iloc[test_idx].groupby("patient_id").size().values
        axs[i].hist(patient_ids, bins=20)
        axs[i].set_title(f"Fold {i}")
        # add text for the percentage of positive cases in the fold
        positive_cases = train_metadata.iloc[test_idx][TARGET_COL].mean()
        axs[i].text(
            0.5,
            0.5,
            f"Positive cases:\n{positive_cases*100:.3f}%",
            transform=axs[i].transAxes,
        )

    plt.savefig("folds.png", bbox_inches="tight")

    with open("folds.json", "w") as f:
        json.dump(folds, f)


if __name__ == "__main__":
    main()
