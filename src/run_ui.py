import sys
from PyQt6.QtWidgets import QApplication

from view import MainWindow  # view.py jest w tym samym folderze (src)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
