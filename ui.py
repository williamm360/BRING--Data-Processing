import asyncio
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import os
import sys
# from data_handling import *
os.environ["QT_API"] = "PyQt6"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.place_elements()

    def open_menu():
        ...

    def open_():
        ...


class MenuBar(QWidget):
    ...


class GraphView(QWidget):
    def __init__(self):
        super().__init__()



class StatsOverview(QWidget):
    ...


class ToolBar(QToolBar):
    ...


class StatusBar(QStatusBar):
    ...

