import os
import h5py as h5
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

# parse each mat structure into a python patient object
class MatImage(object):
    patient_id = ""
    image = ""
    label = ""
    tumor_border = ""
    tumor_mask = ""
    file_name = ""

    def __init__(self, file_name, file_path=None):
        path = file_name
        if file_path is not None:
            path = os.path.join(file_path, file_name)
        f = h5.File(path, "a")

        self.file_name = file_name
        self.image = np.mat(f["/cjdata/image"])
        self.patient_id = np.array(f["/cjdata/PID"])
        self.label = np.array(f["/cjdata/label"])
        self.tumor_border = np.mat(f["/cjdata/tumorBorder"])
        self.tumor_mask = np.mat(f["/cjdata/tumorMask"])

    def draw_image_with_mask(self):
        ax = sns.heatmap(self.tumor_mask, alpha=0.2)
        ax = sns.heatmap(self.image)

        plt.show()

    def get_median_filtered_image(self, filter_size=2):
        return ndimage.median_filter(self.image, filter_size)

    def get_scaled_matrix(self, scaling_factor=10):
        return self.image / scaling_factor

    def tostring(self):
        result = "Patient Id:"
        for pid in self.patient_id:
            result += str(id) + ","
        result += " label"
        for label in self.label:
            result += str(label) + ","
        result += "File Name:" + self.file_name
        return result

