"""Bounded, cached Gemini review of locally shortlisted gameplay events."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from llm.gemini_client import GeminiClient
from llm.gemini_credentials import get_key
from modules.media.event_storyboard import storyboard_parts
from modules.media.video_cache import atomic_write_json

SCHEMA_VERSION = 1


class GeminiEventReview:
    def __init__(self, video_path: str, *, model: str = "gemini-3.5-flash-lite",
                 mode: str = "needed", max_candidates: int = 30,
                 max_calls: int = 30, cache_dir: str = "./cache/gemini_events",
                 log_fn=print, cancel_flag=None, context_fn=None):
        self.video_path = video_path
        self.model = model
        self.mode = mode if mode in ("off", "needed", "finalists") else "off"
        self.max_candidates = max(0, min(100, int(max_candidates)))
        self.max_calls = max(0, min(100, int(max_calls)))
        self.log_fn = log_fn
        self.cancel_flag = cancel_flag
        self.context_fn = context_fn
        stat = os.stat(video_path)
        fingerprint = hashlib.sha256(
            f"{os.path.abspath(video_path)}|{stat.st_size}|{stat.st_mtime_ns}".encode()
        ).hexdigest()
        self.path = Path(cache_dir) / f"{fingerprint}.json"
        try:
            self.cache = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            self.cache = {"schema": SCHEMA_VERSION, "calls": 0, "reviews": {}}
        if self.cache.get("schema") != SCHEMA_VERSION:
            self.cache = {"schema": SCHEMA_VERSION, "calls": 0, "reviews": {}}
        self.cache.setdefault("cache_hits", 0)
        self.calls_this_run = 0
        self.cache_hits = 0

    def _candidate_key(self, item: dict) -> str:
        return hashlib.sha256(
            f"{self.model}|{item['start']:.3f}|{item['end']:.3f}|{SCHEMA_VERSION}".encode()
        ).hexdigest()

    def review(self, finalists: list[dict]) -> None:
        if self.mode == "off" or not get_key():
            if self.mode != "off":
                self.log_fn("ℹ️ Gemini omitido: API key no configurada; continúa el ranking local.")
            return
        # Local shortlist first. Needed mode favours uncertain motion-only
        # candidates, but still reviews strong finalists when budget remains.
        pool = sorted(finalists, key=lambda item: item["local_quality"],
                      reverse=True)[:self.max_candidates * 2]
        if self.mode == "needed":
            pool.sort(key=lambda item: (
                not 0.45 <= item["local_quality"] <= 0.8,
                -item["local_quality"]))
        pool = pool[:self.max_candidates]
        client = GeminiClient(model=self.model)
        client.load()
        for item in pool:
            if self.cancel_flag is not None and self.cancel_flag.is_set():
                break
            key = self._candidate_key(item)
            cached = self.cache["reviews"].get(key)
            if cached:
                item["gemini"] = cached
                self.cache_hits += 1
                self.cache["cache_hits"] += 1
                continue
            if self.cache["calls"] >= self.max_calls:
                self.log_fn("ℹ️ Límite de llamadas Gemini alcanzado para este video.")
                break
            context = {"inicio_s": round(item["start"], 2),
                       "fin_s": round(item["end"], 2),
                       "duracion_s": round(item["end"] - item["start"], 2),
                       "score_local": item["local_quality"],
                       "senales": item.get("signals", []),
                       "evidencia": item.get("reason", "")}
            if self.context_fn:
                context.update(self.context_fn(item) or {})
            frames = storyboard_parts(self.video_path, item["start"], item["end"])
            # Count before the network call, including failures and invalid
            # responses. A restart cannot silently exceed the user's cap.
            self.cache["calls"] += 1
            self.calls_this_run += 1
            atomic_write_json(self.path, self.cache)
            try:
                result = client.evaluate_event(context, frames)
            except Exception as exc:
                detail = str(exc) if isinstance(exc, RuntimeError) else "Error interno"
                self.log_fn(f"⚠️ Revisión Gemini detenida: {detail}. Continúa el ranking local.")
                break
            self.cache["reviews"][key] = result
            atomic_write_json(self.path, self.cache)
            item["gemini"] = result
        if self.cache_hits:
            atomic_write_json(self.path, self.cache)
        self.log_fn(f"Gemini: {self.calls_this_run} llamadas nuevas, "
                    f"{self.cache_hits} respuestas de caché, "
                    f"{self.cache['calls']}/{self.max_calls} llamadas del video.")
