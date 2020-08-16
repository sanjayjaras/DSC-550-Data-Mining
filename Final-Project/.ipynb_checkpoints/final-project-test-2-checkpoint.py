# %%
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

import mat_file_reader as mfr
import image_utils as iu

print("Using version %s of pandas" % pd.__version__)
print("Using version %s of matplotlib" % mpv)
print("Using version %s of seaborn" % sns.__version__)
print("Using version %s of sklearn" % sklearn.__version__)
print("Using version %s of numpy" % np.__version__)
print("Using version %s of h5py" % h5.__version__)

# %% [markdown]
# ### Configurations

# %%
seed = 13
random.seed(seed)
folder = "original-mat-files"
target_folder = "converted-images/"

# %% [markdown]
# ### Scan Source image files

# %%
onlyfiles = [
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if os.path.isfile(os.path.join(folder, f)) and f.endswith(".mat")
]
os.makedirs(target_folder, exist_ok=True)


# %% [markdown]
# ### Read mat files
# ### 1. Use median filter to enhance images
# ### 2. Crop the images to only contain actual brain image and remove extra image parts
# ### 3. Resize all images to 256*256 dimension
# %% [markdown]
# def save_filese_as_image(mat_file_path):
#     mat_image = mfr.MatImage(mat_file_path)
#     filename = mat_image.file_name.split("/")[-1].split(".")[0]
#     target_file_path = target_folder + filename + "_" + str(int(mat_image.label[0][0])) + ".png"
#     iu.save_image(mat_image.get_median_filtered_image(), target_file_path)
#     conv_image = iu.crop_image(target_file_path)
#     iu.save_image(conv_image, target_file_path)
#
# start_time = time.time()
# with cf.ProcessPoolExecutor() as executor:
#         results = executor.map(save_filese_as_image, onlyfiles)
# print(f"--- {(time.time() - start_time)} seconds for transforming {len(onlyfiles)} files---"  )
# %% [markdown]
# ### Scan all converyed image files

# %%
onlyfiles = [
    os.path.join(target_folder, f)
    for f in os.listdir(target_folder)
    if os.path.isfile(os.path.join(target_folder, f)) and f.endswith(".png")
]

# %% [markdown]
# ### Images after above transformations


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
    output_image = np.empty((len(images), 256, 256))

    for i in range(len(images)):
        output_image[i] = images[i]

    return output_image, np.array(labels)


# %% [markdown]
# ### Load all images

# %%
start_time = time.time()
dataset_x, dataset_y = get_data_set()
print("Shape of dataset-X", dataset_x.shape)
print("Shape of dataset-Y", dataset_y.shape)
print(f"--- {(time.time() - start_time)} seconds for loading {len(dataset_x)} files---")

# %% [markdown]
# ### Reshape the datasets

# %%
print("dataset_x:", dataset_x.shape)
X = dataset_x.reshape(len(dataset_x), -1)
print("X:", X.shape)
print("dataset_y:", dataset_y.shape)
y = dataset_y.reshape(len(dataset_y), -1)
print("y:", dataset_y.shape)


# %%
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.fit_transform(X)
print(X.shape)


# %%
print(X[0])

# %% [markdown]
# ### Split dataset into training and test set

# %%
train_x, test_x, train_y, test_y, = train_test_split(
    X, y, train_size=0.80, random_state=seed
)
print("Training Set length", len(train_x))
print("Test Set length", len(test_x))

# %% [markdown]
# ### Number of clusters in data

# %%
n_clusters = len(np.unique(dataset_y))
print(n_clusters)


# %%
from sklearn.neural_network import MLPClassifier

labels = train_y
sample_size = 30

estimator = MLPClassifier(
    solver="sgd",
    activation="relu",
    alpha=1e-5,
    hidden_layer_sizes=(5, 3),
    random_state=seed,
    batch_size=sample_size,
)

estimator.fit(train_x, train_y.reshape(-1))
estimator.labels_
print(
    "%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"
    % (
        (time.time() - t0),
        inertia,
        metrics.homogeneity_score(labels, estimator.labels_),
        metrics.completeness_score(labels, estimator.labels_),
        metrics.v_measure_score(labels, estimator.labels_),
        metrics.adjusted_rand_score(labels, estimator.labels_),
        metrics.adjusted_mutual_info_score(labels, estimator.labels_),
        metrics.silhouette_score(
            data, estimator.labels_, metric="euclidean", sample_size=sample_size
        ),
    )
)

