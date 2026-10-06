"""Optional Gemini text adapter for already-computed local analysis results."""
from __future__ import annotations

import json

import requests

from llm.llm_module import VideoContextBuilder, GenerationCancelled
from llm.gemini_credentials import get_key


class GeminiEvaluationError(RuntimeError):
    """Safe diagnostic; never contains request headers, body or API key."""

    def __init__(self, kind: str, description: str, *, retryable: bool = False):
        self.kind = kind
        self.description = description
        self.retryable = retryable
        super().__init__(f"{kind}: {description}")


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
                       timeout: float = 90) -> dict:
        """Classify one locally shortlisted event with validated JSON."""
        key = get_key()
        if not key:
            raise RuntimeError("API key de Gemini no configurada.")
        prompt = (
            "Evalúa si este evento de gaming merece estar en un resumen para espectadores. "
            "No supongas kills ni resultados que no sean visibles. Distingue movimiento "
            "rutinario, menús y cargas de tensión, consecuencia, apoyo al equipo, humor, "
            "objetivos o cambios reales en la partida. Si la evidencia es insuficiente, "
            "asigna puntuaciones bajas y explica la duda. Decide si el contexto "
            "y el desenlace se ven suficientemente. Los tres valores numéricos "
            "importancia, interes_espectador y consecuencia DEBEN estar en la "
            "escala decimal 0.0–1.0 (nunca 0–100). Responde solo JSON en español.\n"
            "Datos locales: " + json.dumps(context, ensure_ascii=False)[:5000]
        )
        parts = [{"text": prompt}] + (storyboard or [])[:14]
        schema = {"type": "OBJECT", "properties": {
            "relevante": {"type": "BOOLEAN"},
            "categoria": {"type": "STRING"},
            "importancia": {"type": "NUMBER", "minimum": 0, "maximum": 1},
            "interes_espectador": {"type": "NUMBER", "minimum": 0, "maximum": 1},
            "consecuencia": {"type": "NUMBER", "minimum": 0, "maximum": 1},
            "contexto_suficiente": {"type": "BOOLEAN"},
            "evento_completo": {"type": "BOOLEAN"},
            "motivo": {"type": "STRING"},
        }, "required": ["relevante", "categoria", "importancia",
                        "interes_espectador", "consecuencia", "contexto_suficiente",
                        "evento_completo", "motivo"]}
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
        except requests.exceptions.Timeout:
            raise GeminiEvaluationError("timeout", "La respuesta multimedia tardó demasiado.",
                                        retryable=True) from None
        except requests.exceptions.SSLError:
            raise GeminiEvaluationError("TLS", "Falló la conexión segura.") from None
        except requests.exceptions.ConnectionError:
            raise GeminiEvaluationError("red", "No se pudo establecer la conexión.",
                                        retryable=True) from None
        except requests.RequestException:
            raise GeminiEvaluationError("red", "Falló la solicitud de red.",
                                        retryable=True) from None
        if response.status_code in (401, 403):
            raise GeminiEvaluationError(f"HTTP {response.status_code}",
                                        "Gemini rechazó la clave o el acceso al modelo.")
        if response.status_code == 404:
            raise GeminiEvaluationError("HTTP 404", "Modelo no disponible para esta clave.")
        if response.status_code == 400:
            raise GeminiEvaluationError("HTTP 400", "Solicitud o esquema rechazado por Gemini.")
        if response.status_code == 429:
            raise GeminiEvaluationError("HTTP 429", "Límite temporal alcanzado.",
                                        retryable=True)
        if response.status_code in (500, 502, 503, 504):
            raise GeminiEvaluationError(f"HTTP {response.status_code}",
                                        "Servicio temporalmente no disponible.",
                                        retryable=True)
        if not response.ok:
            raise GeminiEvaluationError(f"HTTP {response.status_code}",
                                        "Solicitud rechazada por Gemini.")
        try:
            data = response.json()
        except ValueError:
            raise GeminiEvaluationError("parsing", "La respuesta HTTP no contiene JSON válido.") from None
        try:
            raw = "".join(part.get("text", "")
                          for candidate in data.get("candidates", [])
                          for part in candidate.get("content", {}).get("parts", []))
            if not raw:
                raise GeminiEvaluationError("respuesta vacía", "Gemini no devolvió texto evaluable.")
            try:
                result = json.loads(raw)
            except ValueError:
                raise GeminiEvaluationError("JSON inválido", "No se pudo interpretar la respuesta.") from None
            if not isinstance(result, dict):
                raise GeminiEvaluationError("schema inválido", "Se esperaba un objeto JSON.")
            required = ("relevante", "categoria", "importancia", "interes_espectador",
                        "consecuencia", "contexto_suficiente", "evento_completo", "motivo")
            missing = [name for name in required if name not in result]
            if missing:
                raise GeminiEvaluationError("schema inválido",
                                            "Faltan campos: " + ", ".join(missing))
            for name in ("relevante", "contexto_suficiente", "evento_completo"):
                if type(result.get(name)) is not bool:
                    raise GeminiEvaluationError("schema inválido", f"{name} no es booleano.")
            for name in ("importancia", "interes_espectador", "consecuencia"):
                value = result.get(name)
                if type(value) not in (float, int) or not 0 <= value <= 1:
                    raise GeminiEvaluationError("schema inválido", f"{name} debe estar entre 0 y 1.")
            if not isinstance(result.get("categoria"), str) or not isinstance(result.get("motivo"), str):
                raise GeminiEvaluationError("schema inválido", "Categoría o motivo no es texto.")
            result["categoria"] = result["categoria"][:80]
            result["motivo"] = result["motivo"][:500]
            return result
        except GeminiEvaluationError:
            raise
        except (ValueError, TypeError, KeyError):
            raise GeminiEvaluationError("schema inválido", "Faltan campos o valores válidos.") from None

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
