import io

import h5py
import numpy as np
from constants import HDF5_TRAIN
from PIL import Image
from tqdm import tqdm


def main():
    # get mean and std of all train images
    dataset = h5py.File(HDF5_TRAIN)
    train_imgs = list(dataset.keys())

    img_accumulator = np.array([0.0, 0.0, 0.0])
    img_sq_accumulator = np.array([0.0, 0.0, 0.0])
    num_pixels = 0

    for i, data in enumerate(tqdm(train_imgs)):
        image = Image.open(io.BytesIO(dataset[data][()]))
        image = np.array(image)
        image = image / 255.0
        image = image.astype(np.float32)
        img_accumulator += image.sum(axis=(0, 1))
        img_sq_accumulator += (image**2).sum(axis=(0, 1))
        num_pixels += image.shape[0] * image.shape[1]

    mean = img_accumulator / num_pixels
    var = (img_sq_accumulator / num_pixels) - (mean**2)
    std = np.sqrt(var)

    print(f"Mean: {mean}")
    print(f"Std: {std}")

    # save np array of std and mean
    np.save("mean_2024.npy", mean)
    np.save("std_2024.npy", std)


if __name__ == "__main__":
    main()
