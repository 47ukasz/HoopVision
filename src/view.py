from pathlib import Path

from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer
import time
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QRadioButton,
    QFileDialog, QComboBox, QHBoxLayout, QVBoxLayout, QMessageBox
)

from worker import AnalyzerWorker
from stats_window import StatsWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HoopVision UI")
        self.resize(1100, 700)
        self.start_ts = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)

        self.worker = None
        self.selected_file: Path | None = None

        # podgląd
        self.video_label = QLabel("Podgląd wideo")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(900, 500)
        self.video_label.setStyleSheet("background:#111; color:#bbb; border:1px solid #333;")
        self.video_label.setScaledContents(True)

        # wybór źródła
        self.rb_file = QRadioButton("Plik")
        self.rb_cam = QRadioButton("Kamera")
        self.rb_file.setChecked(True)

        self.btn_pick = QPushButton("Wybierz plik…")
        self.lbl_file = QLabel("Nie wybrano")
        self.lbl_file.setStyleSheet("color:#ccc;")
        self.lbl_total = QLabel("Rzuty: 0")
        self.lbl_hits = QLabel("Trafione: 0")
        self.lbl_time = QLabel("Czas: 00:00")

        for lbl in (self.lbl_total, self.lbl_hits, self.lbl_time):
            lbl.setStyleSheet("color: white; font-size: 16px;")

        self.cb_cam = QComboBox()
        self.cb_cam.addItems(["0", "1", "2"])
        self.cb_cam.setEnabled(False)

        # start/stop
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.last_total = 0
        self.last_hits = 0
        self.stats_window = None

        # layout
        top = QVBoxLayout()
        top.addWidget(self.video_label, 1)

        row_src = QHBoxLayout()
        row_src.addWidget(self.rb_file)
        row_src.addWidget(self.btn_pick)
        row_src.addWidget(self.lbl_file, 1)
        row_src.addSpacing(20)
        row_src.addWidget(self.rb_cam)
        row_src.addWidget(QLabel("Index:"))
        row_src.addWidget(self.cb_cam)
        row_src.addStretch(1)

        row_btn = QHBoxLayout()
        row_btn.addWidget(self.btn_start)
        row_btn.addWidget(self.btn_stop)
        row_btn.addStretch(1)
        row_btn.addStretch(1)
        row_btn.addWidget(self.lbl_total)
        row_btn.addWidget(self.lbl_hits)
        row_btn.addWidget(self.lbl_time)

        top.addLayout(row_src)
        top.addLayout(row_btn)

        root = QWidget()
        root.setLayout(top)
        self.setCentralWidget(root)

        # sygnały
        self.btn_pick.clicked.connect(self.pick_file)
        self.rb_file.toggled.connect(self.update_source_ui)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

    def update_source_ui(self):
        is_file = self.rb_file.isChecked()
        self.btn_pick.setEnabled(is_file)
        self.cb_cam.setEnabled(not is_file)

    def pick_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Wybierz wideo", "", "Video (*.mp4 *.avi *.mov *.mkv);;All (*.*)"
        )
        if file_path:
            self.selected_file = Path(file_path)
            self.lbl_file.setText(str(self.selected_file))

    def start(self):
        if self.worker is not None:
            return

        if self.rb_file.isChecked():
            if not self.selected_file or not self.selected_file.exists():
                QMessageBox.warning(self, "Brak pliku", "Wybierz poprawny plik wideo.")
                return
            source = self.selected_file
        else:
            source = int(self.cb_cam.currentText())

        self.worker = AnalyzerWorker(source)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.stats_ready.connect(self.on_stats)
        self.worker.finished_with_data.connect(self.on_stats_finished)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.rb_file.setEnabled(False)
        self.rb_cam.setEnabled(False)
        self.btn_pick.setEnabled(False)
        self.cb_cam.setEnabled(False)

        self.worker.start()

        self.start_ts = time.time()
        self.timer.start(1000)

    def stop(self):
        if self.worker:
            self.worker.request_stop()
            self.btn_stop.setEnabled(False)
            self.timer.stop()

    def on_frame(self, img):
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def on_error(self, msg: str):
        QMessageBox.critical(self, "Błąd", msg)

    def on_finished(self):
        self.worker = None
        self.timer.stop()

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.rb_file.setEnabled(True)
        self.rb_cam.setEnabled(True)
        self.update_source_ui()

    def on_stats(self, total: int, hits: int):
        self.lbl_total.setText(f"Rzuty: {total}")
        self.lbl_hits.setText(f"Trafione: {hits}")
        self.last_total = total
        self.last_hits = hits

    def on_stats_finished(self, shots_data: list):
        duration_str = self.lbl_time.text().replace("Czas: ", "")
        QTimer.singleShot(0, lambda: self._show_stats_window(shots_data, duration_str))

    def _show_stats_window(self, shots_data, duration_str):
        self.stats_window = StatsWindow(shots_data, duration_str)
        self.stats_window.show()

    def update_time(self):
        if self.start_ts is None:
            return
        elapsed = int(time.time() - self.start_ts)
        mm = elapsed // 60
        ss = elapsed % 60
        self.lbl_time.setText(f"Czas: {mm:02d}:{ss:02d}")