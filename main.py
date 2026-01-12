import matplotlib.pyplot as plt
import os
import functools
import logging
import numpy as np
import pandas as pd
import json
from yaml import safe_load
from scipy.stats import pearsonr


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def error_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception caught in {func.__name__}: {e}")
            return None

    return wrapper


def dir_path():
    main_dir = os.path.dirname(os.path.abspath(
        __file__)).removesuffix(r"\main.py")
    return main_dir


def round_mult(num, base):
    return base * np.round(num / base)


class SingleRun:
    def __init__(self, path, expected_time):
        self.read_data(path)
        self.completion_time = self.time_arr[-1] - self.time_arr[0]
        self.calculate_error_values()
        self.pearson, self.p_value = pearsonr(self.lateral_error_arr,
                                              self.linear_velocity_arr)
        self.completion_time_ratio(expected_time)
        self.calculate_distance()
        self.calculate_amcl_time()
        self.stability()
        self.replan_rate()
        self.fail = self.fail_type()

    def read_data(self, path):
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

    def calculate_error_values(self):
        self.dt_arr = np.diff(self.time_arr)
        error_x = self.plan_x_arr - self.robot_x_arr
        error_y = self.plan_y_arr - self.robot_y_arr

        self.lateral_error_arr = np.hypot(error_x, error_y)

        mid_err = 0.5 * \
            (self.lateral_error_arr[:-1] + self.lateral_error_arr[1:])

        self.rmse = np.sqrt(
            np.sum(np.square(mid_err) * self.dt_arr)/self.completion_time)

        self.mean_error = np.mean(self.lateral_error_arr)
        self.max_error = np.max(self.lateral_error_arr)

    def completion_time_ratio(self, exp_dur):
        self.time_ratio = self.completion_time/exp_dur

    def replan_rate(self):
        self.nb_replan = (self.event_arr == "replan").sum()
        self.replan_per_km = self.nb_replan / self.distance * 1000
        self.replan_per_min = self.nb_replan / self.completion_time * 60

    def calculate_amcl_time(self):
        indexes = np.where(self.amcl_covar_arr < 5)
        if len(indexes) != 0:
            time_index = np.min(indexes)
            self.amcl_time = self.time_arr[time_index]
        else:
            self.amcl_time = None


    def stability(self):
        delta_amcl = np.diff(self.amcl_covar_arr)
        self.jumps, = np.where(delta_amcl > 2)
        self.nb_jumps = len(self.jumps)

    def calculate_distance(self):
        robot_dx_arr = np.diff(self.robot_x_arr)
        robot_dy_arr = np.diff(self.robot_y_arr)
        self.distance = np.sum(np.hypot(robot_dx_arr, robot_dy_arr))

    def calculate_plan_distance(self):
        plan_dx_arr = np.diff(self.plan_x_arr)
        plan_dy_arr = np.diff(self.plan_y_arr)
        plan_dd = np.hypot(plan_dx_arr, plan_dy_arr)
        return np.concatenate(([0.0], np.cumsum(plan_dd)))

    def fail_type(self):
        if self.time_ratio >= 2:
            return "timeout"
        elif self.amcl_covar_arr[-1] >= 20:
            return "localisation loss"
        else:
            return "none"

    def plot(self):
        plt.axis('equal')
        plt.plot(self.plan_x_arr, self.plan_y_arr, "-b")
        plt.plot(self.robot_x_arr, self.robot_y_arr, "-r")
        plt.show()

    def return_values(self):
        return self.rmse, self.mean_error, self.max_error, self.pearson, self.p_value, self.time_ratio, self.distance, self.amcl_time, self.nb_jumps, self.nb_replan, self.replan_per_km, self.replan_per_min, self.completion_time


class Scenario:

    def __init__(self, name, expected_duration_s, expected_distance_m,
                 description, base_error_cm, typical_replan_rate_per_km):

        self.name = name
        self.exp_dur = expected_duration_s
        self.exp_dis = expected_distance_m
        self.desc = description
        self.base_err = base_error_cm
        self.replan_rate = typical_replan_rate_per_km
        self.path = dir_path()
        self.paths = self.get_paths()

        self.analyse_runs()

    def analyse_runs(self):
        count = len(self.paths)
        data_matrix = np.empty((count, 13))  # nb data = 14
        errors_mat = []
        runs_x = []
        runs_y = []

        self.failure_arr = np.empty(count, dtype="str")
        for i, path in enumerate(self.paths):
            run = SingleRun(path, self.exp_dur)
            data_matrix[i] = run.return_values()
            self.failure_arr[i] = run.fail
            runs_x.append(run.robot_x_arr)
            runs_y.append(run.robot_y_arr)
            errors_mat.append(run.lateral_error_arr)
            print(f"run calculated: {self.name}_{i}")
            if i == 0:
                self.plan_path_dist = run.calculate_plan_distance()
                self.theo_x = run.plan_x_arr.copy()
                self.theo_y = run.plan_y_arr.copy()
        self.errors_matrix = np.array(errors_mat)
        self.runs_x_arr = np.array(runs_x)
        self.runs_y_arr = np.array(runs_y)

        (self.rmse_arr, self.mean_error_arr, self.max_error_arr,
         self.pearson_arr, self.p_value_arr, self.time_ratio_arr,
         self.distance_arr, self.amcl_time_arr, self.nb_jumps_arr,
         self.nb_replans_arr, self.replan_km_arr, self.replan_min_arr,
         self.lapse_time) = data_matrix.T

        self.failure_rate = 1 - np.mean(self.failure_arr == "none")
        self.mean_completion_time_ratio = np.mean(self.lapse_time)
        self.success_only_mean_ratio = np.mean(
            self.lapse_time[self.failure_arr == "none"])
        self.mean_rmse = np.mean(self.rmse_arr)
        self.mean_error_total = np.mean(self.mean_error_arr)
        self.max_error_total = np.max(self.max_error_arr)
        self.pearson_mean = np.mean(self.pearson_arr)
        self.amcl_mean = np.mean(self.amcl_time_arr)
        self.amcl_std = np.std(self.amcl_time_arr)

        self.boxplot()
        self.cdf_error()

    def get_paths(self):
        paths = []
        counter = 1

        base_dir = os.path.join(self.path, "navigation_data", "runs")
        while True:
            base = os.path.join(base_dir, f"run_{self.name}_{counter:02d}")
            for suffix in ("_success.csv", "_failure.csv"):
                full_path = base + suffix
                if os.path.exists(full_path):
                    paths.append(full_path)
                    counter += 1
                    break
            else:
                break
        return paths

    def boxplot(self, nb_plots=20):
        grouped_error_matrix = np.array_split(self.errors_matrix.T, nb_plots)
        combined_error_matrix = [arr.flatten() for arr in grouped_error_matrix]
        float_indices = np.linspace(0, len(self.plan_path_dist) - 1, nb_plots)
        indices = np.round(float_indices).astype(int)
        plt.boxplot(combined_error_matrix,
                    positions=np.round(self.plan_path_dist[indices], 2), widths=self.plan_path_dist[-1]/nb_plots, showfliers=False)
        plt.xticks(np.linspace(
            self.plan_path_dist[0], self.plan_path_dist[-1], 10))

        plt.show()

    def heatmap_data(self, x, y, res=1):
        base_x = int(x / res)
        base_y = int(y / res)

        heat_map_mat = np.zeros((base_y, base_x))

        rounded_x = np.floor(self.runs_x_arr / res).astype(int)
        rounded_y = np.floor(self.runs_y_arr / res).astype(int)

        rounded_x = np.clip(rounded_x, 0, base_x - 1)
        rounded_y = np.clip(rounded_y, 0, base_y - 1)

        for i, run in enumerate(self.errors_matrix):
            for j, error in enumerate(run):
                heat_map_mat[rounded_y[i][j], rounded_x[i][j]] += error

        return heat_map_mat

    def cdf_error(self, nb_plots=20):
        errors = self.errors_matrix.flatten() * 100
        max_value = np.ceil(np.max(errors))
        frequencies = np.empty(nb_plots)
        x_vals = np.linspace(0, max_value, nb_plots)
        for i, j in enumerate(x_vals):
            frequencies[i] = np.mean(errors <= j)
        plt.plot(x_vals, frequencies, "-b")
        plt.axhline(0.95, color="r", linestyle="--", label="95%")
        plt.axhline(0.5, color="g", linestyle="--", label="50%")
        p_95 = np.percentile(errors, 95)
        p_50 = np.percentile(errors, 50)
        plt.axvline(p_95, color="r", linestyle="--")
        plt.axvline(p_50, color="g", linestyle="--")
        plt.show()

    def plot_paths(self):
        plt.plot(self.theo_x, self.theo_y, ":r")
        summed_error = np.array([np.sum(arr) for arr in self.errors_matrix])
        mean_x = np.mean(self.runs_x_arr, axis=0)
        mean_y = np.mean(self.runs_y_arr, axis=0)
        min_x = self.runs_x_arr[np.argmin(summed_error)]
        min_y = self.runs_y_arr[np.argmin(summed_error)]
        max_x = self.runs_x_arr[np.argmax(summed_error)]
        max_y = self.runs_y_arr[np.argmax(summed_error)]

        plt.plot(mean_x, mean_y, ":b")
        plt.plot(min_x, min_y, "-g")
        plt.plot(max_x, max_y, "-r")
        plt.plot(self.theo_x, self.theo_y, ":k")

        plt.show()


class Analyse():
    def __init__(self):
        self.scenario = []
        self.process_data()
        self.process_runs()
        self.heat_map("map1", self.scenario, 1)
        for scenario in self.scenario:
            self.heat_map("map1", (scenario,), 1)
        for scenario in self.scenario:
            scenario.plot_paths()

    def process_data(self):
        ...

    def load_JSON(self):
        path_to_JSON = dir_path() + r"\navigation_data\scenarios_metadata.json"
        with open(path_to_JSON, 'r') as file:
            data = json.load(file)
            return data

    def process_runs(self):
        metadata = self.load_JSON()
        for name, params in metadata.items():
            self.scenario.append(Scenario(name, **params))

    def heat_map(self, map_name, scenarios, res):
        parent_dir = dir_path() + "/navigation_data/maps/" + map_name
        map_location_png = parent_dir + "/map.png"
        map_location_yaml = parent_dir + "/map.yaml"
        with open(map_location_yaml, 'r') as file:
            config_dict = safe_load(file)
        from PIL import Image
        with Image.open(map_location_png) as img:
            width_px, height_px = img.size
        width_meter = width_px * config_dict["resolution"]
        height_meter = height_px * config_dict["resolution"]
        img = plt.imread(map_location_png)
        plt.imshow(img, zorder=0, extent=[
                   config_dict["origin"][0], config_dict["origin"][0] + width_meter, config_dict["origin"][1], config_dict["origin"][1] + height_meter])
        base_x = int(width_meter // res)
        base_y = int(height_meter // res)
        cum_heat_map = np.zeros((base_y, base_x))

        for scenario in scenarios:
            cum_heat_map += scenario.heatmap_data(
                width_meter, height_meter, res)

        plt.imshow(cum_heat_map, extent=[
                   config_dict["origin"][0], config_dict["origin"][0] + width_meter, config_dict["origin"][1], config_dict["origin"][1] + height_meter],
                   cmap="hot", alpha=0.5, origin="lower")

        plt.show()


Analyse()
