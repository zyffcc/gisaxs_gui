# main.py

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui.main_window import Ui_MainWindow  # 导入转换后的 UI 类

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("GISAXS Toolkit")
        # 在这里连接信号和槽，初始化界面内容等
        # 例如：self.pushButton.clicked.connect(self.on_button_clicked)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()