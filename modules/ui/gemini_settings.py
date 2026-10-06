"""Small Gemini preferences dialog; the key itself lives in Windows Credentials."""
from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSpinBox, QVBoxLayout)

from llm.gemini_credentials import delete_key, get_key, save_key

ORG = "VideoHighlighter"
APP = "Gemini"


def ai_status_label(provider: str, status: dict | None = None) -> str:
    if provider != "gemini":
        return {"ollama": "Ollama local", "llama-cpp": "GGUF local",
                "none": "Ninguna"}.get(provider, "Ninguna")
    status = status or {}
    valid, planned = status.get("valid", 0), status.get("planned", 0)
    if status.get("reason") == "activo":
        return "Gemini — activo"
    if valid:
        return f"Gemini — parcial ({valid}/{planned} candidatos)"
    if status.get("reason") == "desactivado":
        return "Gemini — desactivado"
    if status.get("reason") == "sin iniciar" and not planned:
        return "Gemini — sin candidatos"
    return "Gemini — fallo; ranking local utilizado"


def gemini_preferences() -> dict:
    settings = QSettings(ORG, APP)
    return {
        "model": str(settings.value("model", "gemini-3.5-flash-lite")),
        "mode": str(settings.value("mode", "needed")),
        "max_candidates": max(0, min(40, int(settings.value("max_candidates", 30)))),
        "max_calls": max(0, min(30, int(settings.value("max_calls", 30)))),
    }


class GeminiSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de Gemini")
        self.setMinimumWidth(470)
        saved = gemini_preferences()
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.Password)
        self.key_input.setPlaceholderText("Clave guardada" if get_key() else "API key no configurada")
        key_row = QHBoxLayout()
        key_row.addWidget(self.key_input)
        eye = QCheckBox("Mostrar")
        eye.toggled.connect(lambda visible: self.key_input.setEchoMode(
            QLineEdit.Normal if visible else QLineEdit.Password))
        key_row.addWidget(eye)
        form.addRow("API key:", key_row)

        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.addItems(["gemini-3.5-flash-lite", "gemini-3.5-flash"])
        self.model.setCurrentText(saved["model"])
        form.addRow("Modelo:", self.model)
        self.mode = QComboBox()
        self.mode.addItem("Desactivado", "off")
        self.mode.addItem("Solo cuando sea necesario", "needed")
        self.mode.addItem("Candidatos finales", "finalists")
        self.mode.setCurrentIndex(max(0, self.mode.findData(saved["mode"])))
        form.addRow("Uso de Gemini:", self.mode)
        self.max_candidates = QSpinBox()
        self.max_candidates.setRange(0, 40)
        self.max_candidates.setValue(saved["max_candidates"])
        form.addRow("Máximo candidatos por video:", self.max_candidates)
        self.max_calls = QSpinBox()
        self.max_calls.setRange(0, 30)
        self.max_calls.setValue(saved["max_calls"])
        form.addRow("Máximo llamadas por video:", self.max_calls)
        layout.addLayout(form)
        self.status = QLabel("✅ Clave configurada" if get_key() else "⚠ API key no configurada")
        layout.addWidget(self.status)
        row = QHBoxLayout()
        for title, callback in (("Guardar de forma segura", self._save_key),
                                ("Eliminar clave", self._delete_key),
                                ("Probar conexión", self._test_connection)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        close = QPushButton("Cerrar")
        close.clicked.connect(self.accept)
        layout.addWidget(close)
        for widget, signal in ((self.model, self.model.currentTextChanged),
                               (self.mode, self.mode.currentIndexChanged),
                               (self.max_candidates, self.max_candidates.valueChanged),
                               (self.max_calls, self.max_calls.valueChanged)):
            signal.connect(self._save_preferences)

    def _save_preferences(self, *_):
        settings = QSettings(ORG, APP)
        settings.setValue("model", self.model.currentText().strip())
        settings.setValue("mode", self.mode.currentData())
        settings.setValue("max_candidates", self.max_candidates.value())
        settings.setValue("max_calls", self.max_calls.value())

    def _save_key(self):
        try:
            save_key(self.key_input.text())
            self.key_input.clear()
            self.key_input.setPlaceholderText("Clave guardada")
            self.status.setText("✅ Clave guardada en Windows Credential Manager")
        except (RuntimeError, ValueError) as exc:
            self.status.setText(f"❌ {exc}")

    def _delete_key(self):
        delete_key()
        self.key_input.clear()
        self.key_input.setPlaceholderText("API key no configurada")
        self.status.setText("⚠ API key no configurada")

    def _test_connection(self):
        try:
            from llm.gemini_client import GeminiClient
            client = GeminiClient(self.model.currentText().strip())
            client.load()
            client.query("Responde solo OK.", free_chat_mode=True,
                         max_tokens=16, timeout=15)
            self.status.setText("✅ Gemini conectado")
        except RuntimeError as exc:
            self.status.setText(f"❌ Error de conexión: {exc}")
