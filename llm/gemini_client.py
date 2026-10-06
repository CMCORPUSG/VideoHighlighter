"""Optional Gemini text adapter for already-computed local analysis results."""
from __future__ import annotations

import json

import requests

from llm.llm_module import VideoContextBuilder, GenerationCancelled
from llm.gemini_credentials import get_key


class GeminiClient:
    def __init__(self, model: str = "gemini-3.5-flash-lite"):
        self.model = model
        self._ready = False

    def load(self) -> None:
        if not get_key():
            raise RuntimeError("Configura la API key de Gemini en Configuración de IA.")
        self._ready = True

    def is_loaded(self) -> bool:
        return self._ready

    def unload(self) -> None:
        self._ready = False

    def evaluate_event(self, context: dict, storyboard: list[dict] | None = None,
                       timeout: float = 35) -> dict:
        """Classify one locally shortlisted event with validated JSON."""
        key = get_key()
        if not key:
            raise RuntimeError("API key de Gemini no configurada.")
        prompt = (
            "Evalúa si este evento de gaming merece estar en un resumen para espectadores. "
            "No supongas kills ni resultados que no sean visibles. Distingue movimiento "
            "rutinario, menús y cargas de tensión, consecuencia, apoyo al equipo, humor, "
            "objetivos o cambios reales en la partida. Si la evidencia es insuficiente, "
            "asigna puntuaciones bajas y explica la duda. Responde solo JSON en español.\n"
            "Datos locales: " + json.dumps(context, ensure_ascii=False)[:5000]
        )
        parts = [{"text": prompt}] + (storyboard or [])[:10]
        schema = {"type": "OBJECT", "properties": {
            "relevante": {"type": "BOOLEAN"},
            "categoria": {"type": "STRING"},
            "importancia": {"type": "NUMBER"},
            "interes_espectador": {"type": "NUMBER"},
            "consecuencia": {"type": "NUMBER"},
            "evento_completo": {"type": "BOOLEAN"},
            "motivo": {"type": "STRING"},
        }, "required": ["relevante", "categoria", "importancia",
                        "interes_espectador", "consecuencia", "evento_completo", "motivo"]}
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json={"contents": [{"parts": parts}],
                      "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512,
                                           "responseMimeType": "application/json",
                                           "responseSchema": schema}},
                timeout=timeout,
            )
        except requests.RequestException:
            raise RuntimeError("No se pudo conectar con Gemini.") from None
        if response.status_code in (401, 403):
            raise RuntimeError("Gemini rechazó la API key.")
        if response.status_code == 429:
            raise RuntimeError("Se alcanzó el límite de solicitudes de Gemini.")
        if not response.ok:
            raise RuntimeError(f"Gemini respondió con el código {response.status_code}.")
        try:
            data = response.json()
            raw = "".join(part.get("text", "")
                          for candidate in data.get("candidates", [])
                          for part in candidate.get("content", {}).get("parts", []))
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError()
            for name in ("relevante", "evento_completo"):
                if type(result.get(name)) is not bool:
                    raise ValueError()
            for name in ("importancia", "interes_espectador", "consecuencia"):
                value = result.get(name)
                if type(value) not in (float, int) or not 0 <= value <= 1:
                    raise ValueError()
            if not isinstance(result.get("categoria"), str) or not isinstance(result.get("motivo"), str):
                raise ValueError()
            result["categoria"] = result["categoria"][:80]
            result["motivo"] = result["motivo"][:500]
            return result
        except (ValueError, TypeError, KeyError):
            raise RuntimeError("Gemini devolvió una evaluación inválida; se usará el ranking local.") from None

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
        key = get_key()
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
