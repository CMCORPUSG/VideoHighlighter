"""Two-candidate Gemini smoke test using an existing local analysis cache."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from llm.gemini_event_review import GeminiEventReview, semantic_approved
from modules.segments.event_rank import combine_semantic


def cached_candidates(video_path: str, cache_dir: str = "./cache") -> list[dict]:
    source = Path(video_path)
    stat = source.stat()
    video_hash = hashlib.sha256(
        f"{source.absolute()}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()
    files = sorted(Path(cache_dir).glob(f"{video_hash}.*.cache.json"),
                   key=lambda path: path.stat().st_mtime, reverse=True)
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("video_hash") != video_hash or not data.get("cache_complete"):
                continue
            segments = data.get("highlight_segments") or []
            meta = (data.get("highlight_metadata") or {}).get("segments_metadata") or []
            candidates = []
            for index, segment in enumerate(segments):
                start, end = float(segment[0]), float(segment[1])
                if end <= start:
                    continue
                info = meta[index] if index < len(meta) else {}
                candidates.append({"start": start, "end": end,
                                   "local_quality": float(info.get("local_quality") or 0.6),
                                   "signals": info.get("event_signals") or [],
                                   "reason": "Candidato del análisis local en caché",
                                   "peak_seconds": []})
            if candidates:
                return sorted(candidates, key=lambda item: item["local_quality"],
                              reverse=True)[:2]
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return []


class GeminiDiagnosticWorker(QThread):
    log = Signal(str)
    result = Signal(dict)

    def __init__(self, video_path: str, model: str, max_calls: int = 30,
                 cache_dir: str = "./cache",
                 parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.model = model
        self.max_calls = max_calls
        self.cache_dir = cache_dir

    def run(self):
        try:
            candidates = cached_candidates(self.video_path, self.cache_dir)
            if not candidates:
                self.log.emit("⚠️ No hay candidatos en la caché de este video. "
                              "Analízalo primero y vuelve a probar Gemini.")
                self.result.emit({"reason": "sin candidatos"})
                return
            self.log.emit(f"🧪 Diagnóstico Gemini: {len(candidates)} candidatos del análisis local.")
            reviewer = GeminiEventReview(
                self.video_path, model=self.model, mode="finalists",
                max_candidates=2, max_calls=self.max_calls,
                cache_dir=str(Path(self.cache_dir) / "gemini_events"),
                log_fn=self.log.emit)
            status = reviewer.review(candidates)
            for index, item in enumerate(candidates, 1):
                evaluation = item.get("gemini")
                if evaluation:
                    after = combine_semantic(item["local_quality"], evaluation)
                    decision = "conservar" if semantic_approved(evaluation) else "descartar"
                    self.log.emit(
                        f"🧪 Candidato {index}: local {item['local_quality']:.3f} → "
                        f"semántico {after:.3f}; decisión: {decision}.\n"
                        "JSON validado: " + json.dumps(evaluation, ensure_ascii=False))
                else:
                    self.log.emit(f"🧪 Candidato {index}: sin evaluación válida; "
                                  "se conserva el ranking local como fallback.")
            self.result.emit(status)
        except Exception:
            self.log.emit("❌ Diagnóstico Gemini: excepción interna al preparar la prueba.")
            self.result.emit({"reason": "excepción interna"})
