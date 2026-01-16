import matplotlib.pyplot as plt
import os
import functools
import logging
import numpy as np
import pandas as pd
import json
from yaml import safe_load
from scipy.stats import pearsonr
from analysis_tools import *
from functools import lru_cache


class SingleRun():
    def __init__(self, path_to_run):
        self.read_raw_data(path_to_run)
        self.calculate_needed_data()
        ...

    def read_raw_data(self, path):
        df = pd.read_csv(path)
        self.time_arr = np.asarray(df["Timestamp"].tolist())
        self.robot_x_arr = np.asarray(df["Robot_X"].tolist())
        self.robot_y_arr = np.asarray(df["Robot_Y"].tolist())
        self.robot_theta_arr = np.asarray(df["Robot_Theta"].tolist())
        self.plan_x_arr = np.asarray(df["Plan_X"].tolist())
        self.plan_y_arr = np.asarray(df["Plan_Y"].tolist())
        self.amcl_covar_arr = np.asarray(df["AMCL_Covar"].tolist())
        self.linear_velocity_arr = np.asarray(df["Velocity_Linear"].tolist())
        self.angular_velocity_arr = np.asarray(df["Velocity_Angular"].tolist())
        self.event_arr = np.asarray(df["Event_Flag"].tolist())

    def calculate_needed_data(self):
        self.completion_time = self.time_arr[-1] - self.time_arr[0]
        self.dt_arr = np.diff(self.time_arr)
        error_x = self.plan_x_arr - self.robot_x_arr
        error_y = self.plan_y_arr - self.robot_y_arr
        self.lateral_error_arr = np.hypot(error_x, error_y)

    def calculate_error_values(self):
        mid_err = 0.5 * \
            (self.lateral_error_arr[:-1] + self.lateral_error_arr[1:])
        rmse = np.sqrt(
            np.sum(np.square(mid_err) * self.dt_arr)/self.completion_time)
        mean_error = np.mean(self.lateral_error_arr)
        max_error = np.max(self.lateral_error_arr)

        return rmse, mean_error, max_error

    def calculate_time_ratio(self, theo_time):
        return self.completion_time / theo_time

    def calculate_replan_rate(self):
        nb_replan = (self.event_arr == "replan").sum()
        replan_per_km = nb_replan / self.distance * 1000
        replan_per_min = nb_replan / self.completion_time * 60

        return replan_per_km, replan_per_min

    def calculate_amcl_time(self):
        amcl_time = None
        indexes = np.where(self.amcl_covar_arr < 5)
        if len(indexes) != 0:
            time_index = np.min(indexes)
            amcl_time = self.time_arr[time_index]

        return amcl_time

    def calculate_stability(self):
        delta_amcl = np.diff(self.amcl_covar_arr)
        jumps, = np.where(delta_amcl > 2)

        return len(self.jumps)

    def calculate_pearson(self):  # cdsvsvsvdsvs
        return pearsonr(self.lateral_error_arr,
                        self.linear_velocity_arr)

    def calculate_scalar_distance(self):
        robot_dx_arr = np.diff(self.robot_x_arr)
        robot_dy_arr = np.diff(self.robot_y_arr)
        return np.sum(np.hypot(robot_dx_arr, robot_dy_arr))

    def calculate_scalar_plan_positions(self):
        plan_dx_arr = np.diff(self.plan_x_arr)
        plan_dy_arr = np.diff(self.plan_y_arr)
        plan_dd = np.hypot(plan_dx_arr, plan_dy_arr)
        return np.concatenate(([0.0], np.cumsum(plan_dd)))

    def fail_type(self, time_ratio):
        if time_ratio >= 2:
            return "timeout"
        elif self.amcl_covar_arr[-1] >= 20:
            return "localisation loss"
        else:
            return "none"

    def get_values():
        ...

    def get_line_data(self):
        ...

    def get_heat_map_data(self):
        ...

    def get_cdfe_data():
        ...

    def fetch_expected_values():
        ...


class Scenario():
    def __init__(self):
        ...

    def set_dir_path():
        ...

    def get_run_paths():
        ...

    def set_default_path():
        ...

    def fetch_values():
        ...


class FullAnalysis():
    ...
