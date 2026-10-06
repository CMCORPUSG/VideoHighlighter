"""Conservative, explainable local evidence for gameplay candidates."""
from __future__ import annotations

import numpy as np


SIGNALS = ("motion_peak", "audio", "loudness_burst", "object", "action", "keyword")


def assess_candidate(start: float, end: float, score, signal_curves=None,
                     labels=()) -> dict:
    values = np.asarray(score, dtype=float)
    a, b = max(0, int(start)), min(len(values), int(np.ceil(end)))
    window = values[a:b]
    if not len(window):
        return {"local_quality": 0.0, "signals": [], "reason": "Sin datos"}
    baseline = float(np.median(values)) if len(values) else 0.0
    scale = max(1.0, float(np.percentile(values, 95)) - baseline)
    peak = float(np.percentile(window, 98))
    contrast = min(1.0, max(0.0, (peak - baseline) / scale))
    active = window > baseline + max(0.5, scale * 0.2)
    thirds = np.array_split(active, 3)
    arc = sum(bool(np.any(part)) for part in thirds) / 3.0
    density = min(1.0, float(np.mean(active)) * 3.0)
    tail = float(bool(np.any(thirds[-1])))

    present = []
    for name in SIGNALS:
        curve = (signal_curves or {}).get(name)
        if curve is not None and np.any(np.asarray(curve)[a:b] > 0):
            present.append(name)
    if not present:
        present = sorted({"action" if str(label).startswith("action:") else
                          "keyword" if str(label).startswith("keyword:") else
                          "loudness_burst" if label == "loudness" else
                          "motion_peak" for label in labels})
    support = min(1.0, len(present) / 3.0)
    quality = (.39 * contrast + .21 * support + .20 * arc +
               .10 * density + .10 * tail)
    if end - start < 20:
        quality *= 0.7
    if present == ["motion_peak"] and arc < 0.67:
        quality *= 0.65
    peak_seconds = []
    for offset in np.argsort(window)[::-1]:
        second = a + int(offset)
        if all(abs(second - old) >= 8 for old in peak_seconds):
            peak_seconds.append(second)
        if len(peak_seconds) == 2:
            break
    return {"local_quality": round(max(0.0, min(1.0, quality)), 4),
            "signals": present,
            "peak_seconds": sorted(peak_seconds),
            "reason": f"Contraste {contrast:.2f}; señales {', '.join(present) or 'ninguna'}; desarrollo {arc:.2f}"}


def combine_semantic(local_quality: float, evaluation: dict | None) -> float:
    if not evaluation:
        return float(local_quality)
    semantic = (0.4 * float(evaluation["importancia"]) +
                0.35 * float(evaluation["interes_espectador"]) +
                0.25 * float(evaluation["consecuencia"]))
    combined = 0.3 * local_quality + 0.7 * semantic
    if not evaluation["relevante"]:
        combined *= 0.4
    if not evaluation["evento_completo"]:
        combined *= 0.75
    if not evaluation.get("contexto_suficiente", True):
        combined *= 0.5
    return round(max(0.0, min(1.0, combined)), 4)
