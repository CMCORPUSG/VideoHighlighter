"""Bounded, observable Gemini review of locally generated gameplay candidates."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from llm.gemini_client import GeminiClient, GeminiEvaluationError
from llm.gemini_credentials import get_key
from modules.media.event_storyboard import storyboard_parts
from modules.media.video_cache import atomic_write_json

SCHEMA_VERSION = 2
PROMPT_VERSION = 2


def semantic_approved(result: dict) -> bool:
    return bool(result.get("relevante") and result.get("contexto_suficiente")
                and result.get("evento_completo")
                and result.get("interes_espectador", 0) >= 0.55)


class GeminiEventReview:
    def __init__(self, video_path: str, *, identity_path: str | None = None,
                 model: str = "gemini-3.5-flash-lite",
                 mode: str = "needed", max_candidates: int = 30,
                 max_calls: int = 30, cache_dir: str = "./cache/gemini_events",
                 log_fn=print, cancel_flag=None, context_fn=None,
                 sleep_fn=time.sleep):
        self.video_path = video_path
        identity_path = identity_path or video_path
        self.model = model
        self.mode = mode if mode in ("off", "needed", "finalists") else "off"
        self.max_candidates = max(0, min(40, int(max_candidates)))
        self.max_calls = max(0, min(30, int(max_calls)))
        self.log_fn = log_fn
        self.cancel_flag = cancel_flag
        self.context_fn = context_fn
        self.sleep_fn = sleep_fn
        stat = os.stat(identity_path)
        fingerprint = hashlib.sha256(
            f"{os.path.abspath(identity_path)}|{stat.st_size}|{stat.st_mtime_ns}".encode()
        ).hexdigest()
        self.path = Path(cache_dir) / f"{fingerprint}.json"
        try:
            old = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            old = {}
        self.cache = {
            "schema": SCHEMA_VERSION,
            "calls": int(old.get("calls", 0)),
            "cache_hits": int(old.get("cache_hits", 0)),
            "reviews": old.get("reviews", {}) if old.get("schema") == SCHEMA_VERSION else {},
        }
        self.calls_this_run = 0
        self.cache_hits = 0
        self.status = {"local": 0, "planned": 0, "sent": 0, "calls": 0,
                       "cached": 0, "valid": 0, "failed": 0, "approved": 0,
                       "rejected": 0, "reason": "sin iniciar"}

    def _candidate_key(self, item: dict, context: dict, frames: list[dict]) -> str:
        payload = {"model": self.model, "prompt": PROMPT_VERSION,
                   "schema": SCHEMA_VERSION, "start": round(item["start"], 3),
                   "end": round(item["end"], 3), "context": context,
                   "storyboard": frames}
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         ensure_ascii=False).encode()).hexdigest()

    def _log_error(self, exc: GeminiEvaluationError, attempt: int, retry: bool):
        delay = (10 if exc.kind == "HTTP 429" else 2) * (2 ** (attempt - 1))
        action = (f"reintento {attempt + 1}/3 en {delay} s" if retry
                  else "ranking local para este candidato")
        self.log_fn(f"❌ Gemini | Tipo: {exc.kind} | Modelo: {self.model} | "
                    f"Descripción: {exc.description} | Acción: {action}.")
        return delay

    def _summary(self):
        s = self.status
        self.log_fn("GEMINI — RESUMEN\n"
                    f"Candidatos locales: {s['local']} | Preparados: {s['planned']} | "
                    f"Enviados: {s['sent']}\n"
                    f"Llamadas nuevas: {s['calls']} | Caché Gemini: {s['cached']} | "
                    f"Respuestas válidas: {s['valid']} | Fallos: {s['failed']}\n"
                    f"Descartados por Gemini: {s['rejected']} | "
                    f"Aprobados por Gemini: {s['approved']}")

    def review(self, candidates: list[dict]) -> dict:
        s = self.status
        s["local"] = len(candidates)
        if self.mode == "off":
            s["reason"] = "desactivado"
            return s
        if not get_key():
            s["reason"] = "API key ausente"
            self.log_fn("❌ Gemini | Tipo: API key ausente | Configura la clave; "
                        "continúa el ranking local.")
            self._summary()
            return s
        self.log_fn("🤖 Gemini: preparando candidatos...")
        pool = sorted(candidates, key=lambda item: item["local_quality"],
                      reverse=True)[:max(20, self.max_candidates * 2)]
        if self.mode == "needed":
            pool.sort(key=lambda item: (
                not 0.45 <= item["local_quality"] <= 0.8,
                -item["local_quality"]))
        pool = pool[:self.max_candidates]
        s["planned"] = len(pool)
        self.log_fn(f"🤖 Candidatos preparados: {len(pool)}")
        client = GeminiClient(model=self.model)
        try:
            client.load()
        except RuntimeError:
            s["reason"] = "API key ausente"
            self._summary()
            return s
        for index, item in enumerate(pool, 1):
            if self.cancel_flag is not None and self.cancel_flag.is_set():
                s["reason"] = "cancelado"
                break
            context = {"inicio_s": round(item["start"], 2),
                       "fin_s": round(item["end"], 2),
                       "duracion_s": round(item["end"] - item["start"], 2),
                       "score_local": item["local_quality"],
                       "senales": item.get("signals", []),
                       "evidencia": item.get("reason", ""),
                       "picos_s": item.get("peak_seconds", [])}
            try:
                if self.context_fn:
                    context.update(self.context_fn(item) or {})
                frames = storyboard_parts(self.video_path, item["start"], item["end"],
                                          extra_seconds=item.get("peak_seconds", []))
                if sum("inlineData" in part for part in frames) < 3:
                    raise ValueError("storybook incompleto")
            except Exception:
                s["failed"] += 1
                item["gemini_failed"] = True
                self.log_fn(f"❌ Gemini: no se pudo preparar el candidato {index}; "
                            "se usará ranking local para este candidato.")
                continue
            key = self._candidate_key(item, context, frames)
            cached = self.cache["reviews"].get(key)
            if cached:
                result = cached
                self.cache_hits += 1
                s["cached"] += 1
                self.cache["cache_hits"] += 1
                self.log_fn(f"🤖 Candidato {index}/{len(pool)}: respuesta de caché.")
            else:
                if self.cache["calls"] >= self.max_calls:
                    s["reason"] = "límite de llamadas"
                    self.log_fn("ℹ️ Límite de llamadas Gemini alcanzado para este video.")
                    break
                s["sent"] += 1
                self.log_fn(f"🤖 Evaluando candidato {index}/{len(pool)}...")
                result = None
                stop_after_failure = False
                for attempt in range(1, 4):
                    if self.cancel_flag is not None and self.cancel_flag.is_set():
                        s["reason"] = "cancelado"
                        break
                    if self.cache["calls"] >= self.max_calls:
                        s["reason"] = "límite de llamadas"
                        break
                    # Every network attempt consumes the persistent per-video cap.
                    self.cache["calls"] += 1
                    self.calls_this_run += 1
                    s["calls"] += 1
                    atomic_write_json(self.path, self.cache)
                    try:
                        result = client.evaluate_event(context, frames)
                        break
                    except GeminiEvaluationError as exc:
                        retry = exc.retryable and attempt < 3 and self.cache["calls"] < self.max_calls
                        delay = self._log_error(exc, attempt, retry)
                        if retry:
                            self.sleep_fn(delay)
                        else:
                            s["reason"] = exc.kind
                            stop_after_failure = exc.kind not in (
                                "JSON inválido", "schema inválido", "respuesta vacía", "parsing")
                            break
                    except Exception:
                        self.log_fn("❌ Gemini | Tipo: excepción interna | "
                                    "Se usará ranking local para este candidato.")
                        s["reason"] = "excepción interna"
                        stop_after_failure = True
                        break
                if result is None:
                    s["failed"] += 1
                    item["gemini_failed"] = True
                    if stop_after_failure:
                        self.log_fn("↩ Se detienen las llamadas Gemini; "
                                    "el resto seguirá con ranking local.")
                        break
                    continue
                self.cache["reviews"][key] = result
                atomic_write_json(self.path, self.cache)
            item["gemini"] = result
            s["valid"] += 1
            if semantic_approved(result):
                s["approved"] += 1
            else:
                s["rejected"] += 1
            self.log_fn(f"✅ Respuesta válida | ⭐ interés: {result['interes_espectador']:.2f} | "
                        f"🏷 categoría: {result['categoria']} | "
                        f"Decisión: {'aprobar' if semantic_approved(result) else 'descartar'}")
        if self.cache_hits:
            atomic_write_json(self.path, self.cache)
        if s["valid"] and s["valid"] == s["planned"]:
            s["reason"] = "activo"
        elif s["valid"]:
            s["reason"] = "parcial"
        elif s["reason"] == "sin iniciar":
            s["reason"] = "fallo"
        self._summary()
        return s
