"""Small in-memory storyboard for a candidate event; never uploads a VOD."""
from __future__ import annotations

import base64

import cv2


def storyboard_parts(video_path: str, start: float, end: float,
                     count: int = 5, extra_seconds=()) -> list[dict]:
    """Return a temporal arc plus up to two evidence peaks as small JPEGs."""
    count = max(1, min(5, int(count)))
    if end <= start:
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return []
    parts = []
    try:
        fractions = [0.03, 0.25, 0.5, 0.75, 0.97] if count == 5 else [
            (i + 0.5) / count for i in range(count)]
        times = [float(start) + (float(end) - float(start)) * fraction
                 for fraction in fractions]
        times.extend(float(t) for t in list(extra_seconds)[:2]
                     if start < float(t) < end and
                     all(abs(float(t) - old) >= 3 for old in times))
        for i, second in enumerate(sorted(times)):
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            if width > 320:
                frame = cv2.resize(frame, (320, max(1, round(height * 320 / width))),
                                   interpolation=cv2.INTER_AREA)
            ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 62])
            if not ok or len(data) > 180_000:
                continue
            parts.append({"text": f"Fotograma {i + 1}/{len(times)} en {second:.1f} s"})
            parts.append({"inlineData": {"mimeType": "image/jpeg",
                                          "data": base64.b64encode(data).decode("ascii")}})
    finally:
        cap.release()
    return parts
