"""Simple review and rating window for exported event clips."""
from __future__ import annotations

import os
import re
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QInputDialog, QLabel,
                               QListWidget, QPushButton, QVBoxLayout)

from modules.segments.clip_feedback import (add_missed_event, load_feedback,
                                            metrics, save_rating, gemini_usage)


class ClipReviewDialog(QDialog):
    def __init__(self, video_path: str, clips_dir: str, segments: list,
                 metadata: list | None = None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.clips = sorted(Path(clips_dir).glob("clip_*.mp4"),
                            key=lambda path: int(path.stem.split("_")[1]))
        self.segments = segments
        self.metadata = metadata or []
        self.setWindowTitle("Revisar clips gaming")
        self.resize(660, 480)
        layout = QVBoxLayout(self)
        self.list = QListWidget()
        layout.addWidget(self.list)
        self.details = QLabel("Selecciona un clip para abrirlo y calificarlo.")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        row = QHBoxLayout()
        actions = (("Abrir clip", self._open_clip),
                   ("⭐ Excelente", lambda: self._rate("excellent")),
                   ("👍 Bueno", lambda: self._rate("good")),
                   ("👎 Malo", lambda: self._rate("bad")))
        for title, callback in actions:
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self.metrics_label = QLabel()
        layout.addWidget(self.metrics_label)
        missed = QPushButton("Registrar evento importante omitido")
        missed.clicked.connect(self._missed)
        layout.addWidget(missed)
        self.list.currentRowChanged.connect(self._selection)
        self._refresh()
        if self.clips:
            self.list.setCurrentRow(0)

    def _segment(self, index):
        if 0 <= index < len(self.segments):
            return self.segments[index]
        if 0 <= index < len(self.clips):
            match = re.search(r"_(\d+h)?(\d+m)(\d+s)-(\d+h)?(\d+m)(\d+s)$",
                              self.clips[index].stem)
            if match:
                def seconds(hour, minute, second):
                    return ((int(hour[:-1]) if hour else 0) * 3600 +
                            int(minute[:-1]) * 60 + int(second[:-1]))
                return (seconds(*match.groups()[:3]), seconds(*match.groups()[3:]))
        return (0.0, 0.0)

    def _refresh(self):
        data = load_feedback(self.video_path)
        current = self.list.currentRow()
        self.list.clear()
        symbols = {"excellent": "⭐", "good": "👍", "bad": "👎"}
        for index, path in enumerate(self.clips):
            start, end = self._segment(index)
            key = f"{float(start):.3f}-{float(end):.3f}"
            rating = data["clips"].get(key, {}).get("rating")
            self.list.addItem(f"{symbols.get(rating, '•')} {path.name}  ·  {end-start:.0f} s")
        if current >= 0 and current < self.list.count():
            self.list.setCurrentRow(current)
        values = metrics(data)
        usage = gemini_usage(self.video_path)
        precision = (f"{values['perceived_precision']:.0%}"
                     if values["perceived_precision"] is not None else "sin evaluar")
        self.metrics_label.setText(
            f"Evaluados: {values['reviewed']}  ·  Excelentes: {values['excellent']}  ·  "
            f"Buenos: {values['good']}  ·  Malos: {values['bad']}\n"
            f"Precisión percibida: {precision}  ·  Falsos positivos: {values['false_positives']}  ·  "
            f"Omitidos: {values['missed_events']}  ·  "
            f"Duración media: {values['average_duration']:.0f} s\n"
            f"Gemini: {usage['calls']} llamadas · {usage['cache_hits']} reutilizaciones "
            f"de caché · {usage['cached']} respuestas guardadas")

    def _selection(self, index):
        if not 0 <= index < len(self.clips):
            return
        start, end = self._segment(index)
        meta = (self.metadata[index] if index < len(self.metadata) else {}) or {}
        gemini = (meta or {}).get("gemini") or {}
        detail = (f"Origen: {start:.1f}–{end:.1f} s  ·  "
                  f"Calidad local: {meta.get('local_quality', 'sin dato')}  ·  "
                  f"Señales: {', '.join(meta.get('event_signals', [])) or 'sin dato'}")
        if gemini:
            detail += f"\nGemini: {gemini.get('categoria', '')} — {gemini.get('motivo', '')}"
        self.details.setText(detail)

    def _open_clip(self):
        index = self.list.currentRow()
        if 0 <= index < len(self.clips):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.clips[index])))

    def _rate(self, rating):
        index = self.list.currentRow()
        if not 0 <= index < len(self.clips):
            return
        start, end = self._segment(index)
        metadata = self.metadata[index] if index < len(self.metadata) else {}
        save_rating(self.video_path, str(self.clips[index]), start, end,
                    rating, metadata)
        self._refresh()

    def _missed(self):
        second, ok = QInputDialog.getDouble(
            self, "Evento omitido", "Segundo aproximado en el video original:",
            min=0, max=10_000_000, decimals=1)
        if ok:
            note, ok_note = QInputDialog.getText(
                self, "Evento omitido", "¿Qué ocurrió? (opcional)")
            add_missed_event(self.video_path, second, note if ok_note else "")
            self._refresh()
