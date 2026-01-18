# stats_window.py

from PyQt6.QtWidgets import (
    QWidget, QListWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt


class StatsWindow(QWidget):
    def __init__(self, shots_data: list[dict], duration: str):  # Dodano argument duration
        super().__init__()
        self.setWindowTitle("Statystyki rzutów")
        self.resize(750, 500)

        self.shots_data = shots_data

        # --- Obliczenia ogólne ---
        total_shots = len(self.shots_data)
        hits = sum(1 for s in self.shots_data if s['result'] == 'hit')
        accuracy = (hits / total_shots * 100) if total_shots > 0 else 0

        # --- Lewa strona: Lista rzutów ---
        self.list_widget = QListWidget()
        for shot in shots_data:
            res = "Trafione" if shot["result"] == "hit" else "Pudło"
            self.list_widget.addItem(f"Rzut {shot['shot_id']} - {res}")

        self.list_widget.currentRowChanged.connect(self.show_details)

        # --- Prawa strona: Szczegóły wybranego rzutu ---
        self.details = QLabel("Wybierz rzut z listy, aby zobaczyć szczegóły.")
        self.details.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.details.setStyleSheet("font-size:14px; padding: 10px;")
        self.details.setWordWrap(True)

        # --- Prawy dół: Panel ogólnych statystyk (Standardowy styl Qt) ---
        self.summary_group = QGroupBox("Podsumowanie filmu")
        summary_layout = QVBoxLayout()

        # Dodano czas trwania do tekstu statystyk
        stats_text = (
            f"Czas trwania: {duration}\n"
            f"Łącznie rzutów: {total_shots}\n"
            f"Trafione: {hits}\n"
            f"Skuteczność: {accuracy:.1f}%"
        )
        self.lbl_overall = QLabel(stats_text)
        self.lbl_overall.setStyleSheet("font-size: 15px; font-weight: bold;")

        summary_layout.addWidget(self.lbl_overall)
        self.summary_group.setLayout(summary_layout)

        # --- Przycisk zapisu ---
        self.btn_save = QPushButton("Zapisz do pliku")
        self.btn_save.setMinimumHeight(40)
        self.btn_save.clicked.connect(self.save_data)

        # --- Konstrukcja Layoutu ---
        right_column = QVBoxLayout()
        right_column.addWidget(self.details, 1)
        right_column.addWidget(self.summary_group)

        content_layout = QHBoxLayout()
        content_layout.addWidget(self.list_widget, 1)
        content_layout.addLayout(right_column, 2)

        main_layout = QVBoxLayout()
        main_layout.addLayout(content_layout)
        main_layout.addWidget(self.btn_save)

        self.setLayout(main_layout)

    def show_details(self, index):
        if index < 0 or index >= len(self.shots_data):
            return
        s = self.shots_data[index]
        self.details.setText(self._format_shot_text(s))

    def _format_shot_text(self, s: dict) -> str:
        res_str = "TRAFIONY!" if s['result'] == 'hit' else "PUDŁO"
        # Formatowanie szczegółów pojedynczego rzutu
        text = f"""Rzut nr: {s['shot_id']}
━━━━━━━━━━━━━━━━━━
Wynik: {res_str}

Minimalny dystans: {s['min_distance_px']:.2f} px
Kąt lotu: {s['angle_deg']:.2f}°

Tunel pod obręczą: {"TAK" if s['tunnel'] else "NIE"}
Ruch siatki: {"TAK" if s['net_moved'] else "NIE"}
"""
        if s.get("net_moves_detected") is not None:
            text += f"\nLiczba ruchów siatki: {s['net_moves_detected']}"
            text += f"\nWymagane ruchy: {s['net_moves_required']}"
        return text

    def save_data(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Zapisz raport", "raport_rzutow.txt", "Pliki tekstowe (*.txt)"
        )
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("RAPORT ANALIZY RZUTÓW\n====================\n")
                f.write(self.lbl_overall.text() + "\n\n")
                f.write("SZCZEGÓŁY:\n")
                for shot in self.shots_data:
                    f.write("-" * 25 + "\n")
                    f.write(self._format_shot_text(shot).replace("━━━━━━━━━━━━━━━━━━", "------------------") + "\n")
            QMessageBox.information(self, "Sukces", "Zapisano pomyślnie.")
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Błąd zapisu: {e}")