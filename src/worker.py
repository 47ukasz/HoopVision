
# odpala analize w osobnym watku, zeby nie blokowac UI
from dataclasses import dataclass

import cv2 as cv
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from .main import analyze  


@dataclass
class StopFlag:
    stop: bool = False


class AnalyzerWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    error = pyqtSignal(str)
    stats_ready = pyqtSignal(int, int)  # total_shots, hit_shots

    def __init__(self, source):
        super().__init__()
        self.source = source  # Path/str albo int
        self.stop_flag = StopFlag(False)

    def request_stop(self):
        self.stop_flag.stop = True

    def _to_qimage(self, bgr):
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

    def run(self):
        try:
            def on_frame(frame_bgr):
                self.frame_ready.emit(self._to_qimage(frame_bgr))

            def on_stats(total, hits):
                self.stats_ready.emit(total, hits)

            analyze(
                source=self.source,
                frame_callback=on_frame,
                stats_callback=on_stats,
                stop_flag=self.stop_flag,
            )
        except Exception as e:
            self.error.emit(str(e))
