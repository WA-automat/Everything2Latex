from lat import *
from PyQt5.QtWidgets import QApplication,QMainWindow
import sys


class uiMainWindow(QMainWindow):
        def __init__(self):
                super().__init__()
                self.ui = Ui_uiMainWindow()
                self.ui.setupUi(self)
                self.ui.stackedWidget.setCurrentIndex(0)
                self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
                self.show()
        def turntopage_pic(self):
                self.ui.stackedWidget.setCurrentIndex(0)
        def turntopage_paint(self):
                self.ui.stackedWidget.setCurrentIndex(1)
        def turntopage_about(self):
                self.ui.stackedWidget.setCurrentIndex(2)

if __name__ == '__main__':
        app = QApplication(sys.argv)
        dlg = uiMainWindow()
        sys.exit(app.exec_())
