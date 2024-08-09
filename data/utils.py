#%%
import os
os.chdir("../")
#%%
import matplotlib.pyplot as plt
from modeling import get_transforms
import h5py
import random
import cv2
import numpy as np

#%%
if __name__ == "__main__":
    train_transforms, val_transforms = get_transforms(512, 0.1)
    dataset = h5py.File("train-image.hdf5", "r")
    keys = list(dataset.keys())

    image = dataset[keys[random.randint(0, len(keys))]][()]
    image = cv2.imdecode(
        np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR
    )
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    print(img.max(), img.min())
    print(img.dtype)

    plt.imshow(img)

    plt.show()

    transformed = val_transforms(image=img)
    img = transformed["image"].numpy().astype(np.float32)
    img = img.transpose(1, 2, 0)

    plt.imshow(img)
    plt.show()

    print(img.shape)
    print(img.dtype)
    print(img.max(), img.min())