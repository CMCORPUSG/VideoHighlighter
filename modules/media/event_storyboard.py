"""Small in-memory storyboard for a candidate event; never uploads a VOD."""
from __future__ import annotations

import base64

import cv2


def storyboard_parts(video_path: str, start: float, end: float,
                     count: int = 5) -> list[dict]:
    """Return at most five labelled JPEG frames, bounded to 320 px wide."""
    count = max(1, min(5, int(count)))
    if end <= start:
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return []
    parts = []
    try:
        for i in range(count):
            fraction = (i + 0.5) / count
            second = float(start) + (float(end) - float(start)) * fraction
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
            parts.append({"text": f"Fotograma {i + 1}/{count} en {second:.1f} s"})
            parts.append({"inlineData": {"mimeType": "image/jpeg",
                                          "data": base64.b64encode(data).decode("ascii")}})
    finally:
        cap.release()
    return parts
