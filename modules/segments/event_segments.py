"""Select complete gameplay events from local signal intervals.

The duration target is a budget for choosing events, never a clip length. Scene
cuts can add score, but cannot start or end an event on their own.
"""
from __future__ import annotations

from math import sqrt

import numpy as np


def _time(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _event_intervals(action_sequences, keyword_matches, loudness_bursts):
    intervals = []
    for action in action_sequences or ():
        if len(action) >= 2:
            start, end = _time(action[0]), _time(action[1])
            if end > start:
                label = str(action[4]).lower() if len(action) >= 5 else "action"
                intervals.append((start, end, {f"action:{label}"}))
    for match in keyword_matches or ():
        segment = match.get("main_segment", {})
        start, end = _time(segment.get("start")), _time(segment.get("end"))
        if end > start:
            intervals.append((start, end, {f"keyword:{match.get('keyword', '')}"}))
    for burst in loudness_bursts or ():
        if isinstance(burst, dict):
            start, end = _time(burst.get("start")), _time(burst.get("end"))
        else:
            start, end = _time(burst[0]), _time(burst[1])
        if end > start:
            intervals.append((start, end, {"loudness"}))
    return sorted(intervals, key=lambda x: x[0])


def _salient_intervals(score, scenes):
    """Find activity that stands out within its part of the VOD."""
    values = np.asarray(score, dtype=float)
    scene_starts = {int(round(start)) for start, _ in scenes or ()}
    hot = []
    for offset in range(0, len(values), 180):
        block = values[offset:offset + 180]
        positive = block[block > 0]
        if not len(positive):
            continue
        dense = len(positive) > len(block) / 2
        if float(np.max(positive)) == float(np.min(positive)) and dense:
            continue
        median = float(np.median(positive))
        threshold = (max(float(np.percentile(positive, 80)),
                         median + max(1.0, median * 0.1))
                     if dense else float(np.percentile(positive, 80)))
        hot.extend(int(i + offset) for i in np.flatnonzero(block >= threshold)
                   if not (i + offset in scene_starts and block[i] <= 2))
    # A scene-cut-only score is insufficient evidence of an event.
    if hot and all(i in scene_starts for i in hot):
        return []
    return [(float(i), float(i + 1), {"activity"}) for i in hot]


def _merge_activity(intervals, gap=15.0,
                    semantic_gap=50.0):
    """Connect nearby evidence; longer gaps need a shared specific label."""
    if not intervals:
        return []
    merged = [[intervals[0][0], intervals[0][1], set(intervals[0][2])]]
    for start, end, labels in intervals[1:]:
        prev = merged[-1]
        distance = start - prev[1]
        shared = prev[2] & labels
        specific = any(label.startswith(("action:", "keyword:", "object:"))
                       for label in shared)
        # A long run gets a boundary only at an actual lull, not at a clock
        # tick. Continuous action can legitimately remain longer than 5 min.
        long_lull = prev[1] - prev[0] >= 240 and distance >= 12 and not specific
        if not long_lull and (distance <= gap or (distance <= semantic_gap and specific)):
            prev[1] = max(prev[1], end)
            prev[2].update(labels)
        else:
            merged.append([start, end, set(labels)])
    return merged


def build_event_segments(*, video_duration, score, scenes=None,
                         motion_events=None, motion_peaks=None, audio_peaks=None,
                         object_detections=None, action_sequences=None,
                         keyword_matches=None, loudness_bursts=None,
                         target_duration=900, log_fn=print):
    """Return non-overlapping, variable-length event clips in source order.

    A 15-second context margin is a starting point. Nearby evidence extends the
    event before margins are applied. The target is approximate: an event is
    never truncated to consume the last seconds of the budget.
    """
    duration = max(0.0, float(video_duration))
    values = np.asarray(score, dtype=float)
    intervals = _event_intervals(action_sequences, keyword_matches,
                                 loudness_bursts)
    salient = _salient_intervals(values, scenes)
    intervals.extend(salient)
    ranking_values = values
    if not salient and (motion_peaks or audio_peaks):
        # Old profiles may score only scene cuts or object classes absent from
        # this video. Motion/audio measurements still provide local evidence.
        fallback = np.zeros(len(values), dtype=float)
        for timestamps, points in ((motion_peaks, 5.0), (audio_peaks, 3.0)):
            for t in timestamps or ():
                i = int(round(_time(t)))
                if 0 <= i < len(fallback):
                    fallback[i] += points
        intervals.extend(_salient_intervals(fallback, ()))
        ranking_values = values + fallback
        if intervals:
            log_fn("ℹ️ Se usaron picos de movimiento/audio: la puntuación configurada no produjo eventos.")
    intervals.sort(key=lambda item: item[0])
    objects = {int(sec): {str(name).lower() for name in names}
               for sec, names in (object_detections or {}).items()}
    generic = {"person", "human", "player", "screen"}
    enriched = []
    for start, end, labels in intervals:
        labels = set(labels)
        for sec in range(max(0, int(start) - 2), min(int(end) + 3, int(start) + 8)):
            labels.update(f"object:{name}" for name in objects.get(sec, ())
                          if name not in generic)
        enriched.append((start, end, labels))
    intervals = enriched
    groups = _merge_activity(intervals)
    if not groups:
        log_fn("ℹ️ No hay evidencia temporal suficiente para formar eventos completos.")
        return [], []

    # Context may overlap between *distinct* activity groups. Divide that
    # quiet gap instead of merging the clips by transitive padding: otherwise
    # an hour of gameplay can become one continuous candidate.
    # Scene changes are deliberately absent from the boundary rule.
    candidates = []
    for index, (start, end, labels) in enumerate(groups):
        if start >= duration:
            continue
        span = end - start
        before = 12.0 if span < 15 else 18.0
        after = 15.0 if span < 15 else 20.0
        clip_start = max(0.0, start - before)
        clip_end = min(duration, end + after)
        if index and clip_start < candidates[-1][1]:
            boundary = (groups[index - 1][1] + start) / 2.0
            candidates[-1][1] = min(candidates[-1][1], boundary)
            clip_start = max(clip_start, boundary)
        candidates.append([clip_start, clip_end, labels])
    ranked = []
    for start, end, labels in candidates:
        if end <= start:
            continue
        a, b = max(0, int(start)), min(len(values), int(np.ceil(end)))
        strength = float(np.sum(ranking_values[a:b])) / sqrt(max(1.0, end - start))
        ranked.append((strength, start, end, labels))
    ranked.sort(key=lambda item: item[0], reverse=True)

    budget = max(0.0, float(target_duration))
    selected = []
    used = 0.0
    for _, start, end, labels in ranked:
        length = end - start
        if budget and used + length > budget:
            # Select an entire event only if it is closer to the target than
            # stopping here. Never manufacture a partial ending.
            if selected and used + length - budget >= budget - used:
                continue
        selected.append((start, end, labels))
        used += length
        if budget and used >= budget:
            break
    selected.sort(key=lambda item: item[0])
    segments = [(start, end) for start, end, _ in selected]
    log_fn(f"🎮 {len(segments)} eventos completos; {used:.1f} s "
           f"(objetivo aproximado: {budget:.1f} s)")
    return segments, selected
