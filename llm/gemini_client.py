"""Optional Gemini text adapter for already-computed local analysis results."""
from __future__ import annotations

import os

import requests

from llm.llm_module import VideoContextBuilder, GenerationCancelled


class GeminiClient:
    def __init__(self, model: str = "gemini-3.5-flash-lite"):
        self.model = model
        self._ready = False

    def load(self) -> None:
        if not os.environ.get("GEMINI_API_KEY", "").strip():
            raise RuntimeError("Configura GEMINI_API_KEY en el entorno para usar Gemini.")
        self._ready = True

    def is_loaded(self) -> bool:
        return self._ready

    def unload(self) -> None:
        self._ready = False

    def query(self, user_message: str, analysis_data: dict | None = None,
              free_chat_mode: bool = False, video_path: str = "",
              system_prompt: str | None = None, timeline_context: str = "",
              frame_base64: str | None = None, max_tokens: int = 1024,
              temperature: float = 0.3, stream_callback=None,
              cancellation_token=None, timeout: float = 90) -> str:
        if not self._ready:
            raise RuntimeError("Gemini no está conectado.")
        if frame_base64:
            raise RuntimeError("Gemini recibe solo texto de análisis local; no se envían fotogramas.")
        if cancellation_token and cancellation_token.is_cancelled:
            raise GenerationCancelled()
        parts = ["Responde en español latinoamericano neutro. Usa solo la evidencia proporcionada; "
                 "si falta información, dilo claramente."]
        if system_prompt:
            parts.append(system_prompt[:3000])
        if analysis_data and not free_chat_mode:
            parts.append("Datos del análisis local:\n" +
                         VideoContextBuilder.build(analysis_data, video_path)[:24000])
        if timeline_context:
            parts.append("Contexto de la línea de tiempo:\n" + timeline_context[:4000])
        parts.append("Pregunta:\n" + user_message[:4000])
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("Falta GEMINI_API_KEY.")
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": "\n\n".join(parts)}]}],
                      "generationConfig": {"temperature": temperature,
                                           "maxOutputTokens": max_tokens}},
                timeout=timeout,
            )
            if response.status_code in (401, 403):
                raise RuntimeError("La clave de Gemini no fue aceptada. Revisa GEMINI_API_KEY.")
            if response.status_code == 429:
                raise RuntimeError("Gemini alcanzó su límite de solicitudes. Intenta más tarde.")
            if not response.ok:
                raise RuntimeError(f"Gemini respondió con el código {response.status_code}.")
            data = response.json()
            answer = "".join(part.get("text", "") for candidate in data.get("candidates", [])
                             for part in candidate.get("content", {}).get("parts", []))
            if not answer:
                raise RuntimeError("Gemini no devolvió texto. Intenta reformular la pregunta.")
            if cancellation_token and cancellation_token.is_cancelled:
                raise GenerationCancelled()
            if stream_callback:
                stream_callback(answer)
            return answer
        except requests.RequestException:
            # Never include request or response repr: headers contain the key.
            raise RuntimeError("No se pudo conectar con Gemini. Revisa tu conexión.") from None
