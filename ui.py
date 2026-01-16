
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import os
import sys
# from data_handling import *
os.environ["QT_API"] = "PyQt6"

WINDOW_X_SCALE = 0.8
WINDOW_Y_SCALE = 0.8


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_window()
        self.init_elements()

    def init_window(self):
        self.setWindowTitle("Bring Error Analyser")
        self.setBaseSize(1920, 1080)
        screen = QApplication.primaryScreen()
        rect = screen.availableGeometry()
        width = int(rect.width() * WINDOW_X_SCALE)
        height = int(rect.height() * WINDOW_Y_SCALE)
        self.resize(width, height)
        self.setMinimumSize(600, 360)

    def init_elements(self):
        self.infographics = Infographics()
        self.ctlpanel = ControlPanel()
        self.menubar = MenuBar()
        self.statusbar = StatusBar()
        self.toolbar = ToolBar()
        self.setCentralWidget(self.infographics)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.ctlpanel)
        self.setMenuBar(self.menubar)
        self.setStatusBar(self.statusbar)
        self.addToolBar(self.toolbar)


class ControlPanel(QDockWidget):
    ...

    def __init__(self):
        super().__init__()
        self.init_widget()
        self.init_elems()

    def init_widget(self):
        self.setMinimumWidth(100)
        self.setMaximumWidth(400)

    def init_elems(self):
        self.container = QWidget()
        self.setWidget(self.container)
        self.main_layout = QVBoxLayout(self.container)
        self.file_tree = FileTree()
        self.logs = LogMessages()
        self.main_layout.addWidget(TestWidget(300, 200))
        self.main_layout.addWidget(QLabel("hi"))

    def show_start():
        ...

    def show_stats():
        ...

    def create_elem():
        ...

    def hide_bar():
        ...


class Infographics(QWidget):
    def __init__(self):
        super().__init__()


class TrajGraphs():
    def __init__(self):
        super().__init__()
    ...


class StatsGraph():
    def __init__(self):
        super().__init__()
    ...


class NumStatsView():
    def __init__(self):
        super().__init__()
    ...


class StatsOverview(QTabWidget):
    def __init__(self):
        super().__init__()
    ...


class MenuBar(QMenuBar):
    def __init__(self):
        super().__init__()


class StatusBar(QStatusBar):
    def __init__(self):
        super().__init__()
    ...


class ToolBar(QToolBar):
    def __init__(self):
        super().__init__()
    ...
    # Static Error tab with record run feature
    # Raw data tab
    # Robot input output tab


class StatsPage(QWidget):
    def __init__(self):
        super().__init__()
    ...


class TestWidget(QWidget):
    def __init__(self, x, y):
        super().__init__()
        self.setFixedSize(x, y)
        self.resize(x, y)
        self.setStyleSheet("background: magenta ")

        print("TEST WIDGET CREATED", self.size())


class FileTree():
    ...

    # setacceptdrop


class LogMessages():
    def __init__():
        super().__init__()
        self.


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    with open("style.qss") as f:
        window.setStyleSheet(f.read())
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
