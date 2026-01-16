import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import sys
from functools import lru_cache
np.set_printoptions(threshold=sys.maxsize)

caching = True


def set_caching(toggle: bool):
    ...


def accumulate():
    ...

# static

# dynamic


def percentage_value():
    ...


def squeezed_list():
    ...


def datamat_to_datachunks(mat: np.ndarray | list, nb_plots=20, mode="distribute"):
    if mode == "distribute":
        split_matrixes = np.array_split(
            mat, nb_plots)
        combined_matrix = [mat.ravel() for mat in split_matrixes]

        return combined_matrix

    if mode == "round":
        split_matrixes = np.split(
            mat, int(nb_plots//mat.shape[1]))
        combined_matrix = np.asarray([mat.ravel() for mat in split_matrixes])

        return combined_matrix

    if mode == "divide":
        flipped_matrix = mat.T
        flattened_matrix = flipped_matrix.flatten()
        combined_matrix = np.array_split(
            flattened_matrix, nb_plots)
        return combined_matrix

    if mode == "cut":
        ...

    if mode == "squeeze":
        ...

    else:
        raise ("invalid Mode for distribution")


class BoxPlot():

    def __init__(self, x_data: (np.ndarray | list), y_data: (np.ndarray | list), nb_plots: int = 20, dtype: str = "float32", show_floaters: bool = False):
        self.raw_x = x_data
        self.raw_y = y_data
        self.dtype = dtype
        self.nb_plots = nb_plots
        self.boxplot_data = datamat_to_datachunks(self.raw_y, self.nb_plots)

    def create_work_data(self):
        x_vals = np.linspace()
        y_vals = datamat_to_datachunks(self.raw_y, self.nb_plots)
        return

        ...

    def get_partial_plot():
        ...

    def get_full_plot():
        ...

    def update_plot():
        ...

    def set_nb_plot():
        ...

    def get_dt_tensor():
        ...

    def add_plot():
        ...

    def add_():
        ...

    def set_data(self, new_x=None, new_y=None):
        self.raw_x = self.raw_x if new_x == None else new_x
        self.raw_y = self.raw_y if new_y == None else new_y

    def load_data():
        ...

    def __repr__(self):
        return "Boxplot"

    def __add__(self, val: Union['BoxPlot', np.ndarray]):
        ...


class Heatmap():

    def __init__(self, x, y, res, cmap,):
        ...

    def create_heatmap():
        ...

    def cumulate_data():
        ...

    def get_heatmap():
        ...

    def get_dt_tensor():
        ...

    def config():
        ...

    def update_heatmap():
        ...

    def set_resolution():
        ...


class PictureMap():
    ...


class RobotPath():
    ...


class CDFGraph():
    ...


class CompositeGraph():
    ...


class graph():
    ...


rng = np.random.default_rng()
int_matrix = rng.integers(low=-100, high=100, size=(12, 200))
a = datamat_to_datachunks(int_matrix, nb_plots=8)
print(len(a), len(a[1]), len(a[-2]), len(a[-1]))

# create full min size arrays
# combine rest at eq index + stuff
# distribute
#


class UnhoMat():
    

