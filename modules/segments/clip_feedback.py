"""Local review records for exported clips; no training or remote upload."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from modules.media.video_cache import atomic_write_json


def feedback_path(video_path: str, cache_dir: str = "./cache") -> Path:
    try:
        stat = os.stat(video_path)
        identity = f"{os.path.abspath(video_path)}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        identity = os.path.abspath(video_path)
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return Path(cache_dir) / "clip_feedback" / f"{digest}.json"


def load_feedback(video_path: str, cache_dir: str = "./cache") -> dict:
    path = feedback_path(video_path, cache_dir)
    try:
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema") == 1:
            return data
    except (OSError, ValueError):
        pass
    return {"schema": 1, "video": os.path.abspath(video_path),
            "clips": {}, "missed_events": []}


def save_rating(video_path: str, clip_path: str, start: float, end: float,
                rating: str, metadata: dict | None = None,
                cache_dir: str = "./cache") -> dict:
    if rating not in ("excellent", "good", "bad"):
        raise ValueError("Calificación desconocida")
    data = load_feedback(video_path, cache_dir)
    key = f"{float(start):.3f}-{float(end):.3f}"
    metadata = metadata or {}
    data["clips"][key] = {
        "clip": os.path.abspath(clip_path), "start": float(start),
        "end": float(end), "duration": float(end) - float(start),
        "rating": rating,
        "local_quality": metadata.get("local_quality"),
        "final_quality": metadata.get("final_quality"),
        "signals": metadata.get("event_signals", []),
        "category": (metadata.get("gemini") or {}).get("categoria"),
        "gemini_used": bool(metadata.get("gemini")),
        "gemini_result": metadata.get("gemini"),
    }
    atomic_write_json(feedback_path(video_path, cache_dir), data)
    return data


def add_missed_event(video_path: str, timestamp: float, note: str = "",
                     cache_dir: str = "./cache") -> dict:
    data = load_feedback(video_path, cache_dir)
    data["missed_events"].append({"timestamp": float(timestamp),
                                  "note": note[:500]})
    atomic_write_json(feedback_path(video_path, cache_dir), data)
    return data


def metrics(data: dict) -> dict:
    clips = list(data.get("clips", {}).values())
    counts = {rating: sum(c.get("rating") == rating for c in clips)
              for rating in ("excellent", "good", "bad")}
    n = len(clips)
    return {"reviewed": n, **counts,
            "perceived_precision": ((counts["excellent"] + counts["good"]) / n
                                    if n else None),
            "false_positives": counts["bad"],
            "missed_events": len(data.get("missed_events", [])),
            "average_duration": (sum(c.get("duration", 0) for c in clips) / n
                                 if n else 0.0)}


def gemini_usage(video_path: str, cache_dir: str = "./cache") -> dict:
    path = feedback_path(video_path, cache_dir)
    candidate = path.parent.parent / "gemini_events" / path.name
    try:
        import json
        data = json.loads(candidate.read_text(encoding="utf-8"))
        return {"calls": int(data.get("calls", 0)),
                "cached": len(data.get("reviews", {})),
                "cache_hits": int(data.get("cache_hits", 0))}
    except (OSError, ValueError, TypeError):
        return {"calls": 0, "cached": 0, "cache_hits": 0}
