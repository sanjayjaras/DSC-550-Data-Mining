import concurrent.futures as cf
import os
import random
import time
import cv2

import h5py as h5  # library to load HDF5 binary file format
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib import __version__ as mpv
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GridSearchCV, train_test_split

import mat_file_reader as mfr
import image_utils as iu
from keras.models import Sequential
from keras.layers import (
    ZeroPadding2D,
    MaxPool2D,
    Conv2D,
    BatchNormalization,
    Activation,
    Flatten,
    Dense,
    Input,
)

print("Using version %s of pandas" % pd.__version__)
print("Using version %s of matplotlib" % mpv)
print("Using version %s of seaborn" % sns.__version__)
print("Using version %s of sklearn" % sklearn.__version__)
print("Using version %s of numpy" % np.__version__)
print("Using version %s of h5py" % h5.__version__)

seed = 13
random.seed(seed)
folder = "original-mat-files"
target_folder = "converted-images/"

# %%
onlyfiles = [
    os.path.join(target_folder, f)
    for f in os.listdir(target_folder)
    if os.path.isfile(os.path.join(target_folder, f)) and f.endswith(".png")
]


# %% [markdown]
# ### Function tp load all images into numpy arrays with labels

# %%
def load_image(image_path):
    img = cv2.imread(image_path)
    # enable this to remove last dimension 3
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    filename = image_path.split("/")[-1].split(".")[0]
    label = filename.split("_")[-1]
    return img, int(label), filename


def get_data_set(for_cnn=False):
    images = []
    labels = []

    filenames = []
    with cf.ProcessPoolExecutor() as executor:
        results = executor.map(load_image, onlyfiles)
    for result in results:
        if result[0] is not None:
            if result[0].shape[0] != 256 or result[0].shape[1] != 256:
                print("Dropping image as dimensions are not correct", result[2])
            else:
                image = result[0]
                if for_cnn:
                    # images.append(image)
                    if result[1] == 1:
                        labels.append([1, 0, 0])
                    elif result[1] == 2:
                        labels.append([0, 1, 0])
                    else:
                        labels.append([0, 0, 1])
                else:
                    labels.append(result[1])
                images.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
                filenames.append(result[2])
    # remove last dimension if removed gray
    output_image = np.empty((len(images), 256, 256))

    for i in range(len(images)):
        output_image[i] = images[i]

    return output_image, np.array(labels)


# %%
start_time = time.time()
dataset_xc, dataset_yc = get_data_set(for_cnn=True)
print("Shape of dataset-X", dataset_xc.shape)
print("Shape of dataset-Y", dataset_yc.shape)
print(
    f"--- {(time.time() - start_time)} seconds for loading {len(dataset_xc)} files---"
)

xc = dataset_xc.reshape(len(dataset_xc), -1)
xc.shape

# %%
print("dataset_yc:", dataset_yc.shape)
yc = dataset_yc.reshape(len(dataset_yc), -1)
print("yc:", dataset_yc.shape)

# %% [markdown]
# ### Normalize image data

# %% [markdown]
# ### Split dataset in testsets and training sets

# %%
train_xc, test_xc, train_yc, test_yc, = train_test_split(
    xc, yc, train_size=0.80, random_state=seed
)
print("Training Set length", len(train_xc))
print("Test Set length", len(test_xc))


# %%
train_xc.shape


# %%


def create_keras_model(input_shape):
    model = Sequential()
    model.add(Input(input_shape))
    model.add(BatchNormalization(axis=1, name="bn0"))
    model.add(Dense(10000))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
    )

    return model


# %%

IMG_SHAPE = 65536
# model = create_keras_model(IMG_SHAPE)
# model.summary()


# %%
for batch_size in [30, 50]:
    model = create_keras_model(IMG_SHAPE)
    model.summary()
    model.fit(
        x=train_xc,
        y=train_yc,
        batch_size=batch_size,
        epochs=25,
        validation_split=0.2,
        verbose=1,
    )


# %%

