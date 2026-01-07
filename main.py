# import matplotlib.plt as pyplot
import os


def load_map(map_name: str):
    try:
        parent_dir = dir_path() + "/navigation_data/maps/" + map_name
        map_location_png = parent_dir + "/map.png"
        map_location_yaml = parent_dir + "/map.yaml"
        print(map_location_yaml)
        with open(map_location_yaml, "r") as file:
            for line in file:
                print(line)

    except:
        print("error")


def read_data():
    ...


def pearson():
    ...


def main():
    ...


def dir_path():
    main_dir = os.path.dirname(os.path.abspath(
        __file__)).removesuffix("\main.py")
    return main_dir


print(dir_path())

load_map("map1")
