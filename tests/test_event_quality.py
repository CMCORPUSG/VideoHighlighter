import json

import numpy as np
import pytest

from modules.segments.clip_feedback import (add_missed_event, load_feedback,
                                            metrics, save_rating)
from modules.segments.event_rank import assess_candidate, combine_semantic


def test_local_rank_penalizes_flat_activity_and_rewards_multiple_signals():
    flat = np.full(300, 4.0)
    weak = assess_candidate(60, 120, flat, {"motion_peak": flat})
    scored = flat.copy()
    scored[60:120] = 16
    motion = np.zeros(300)
    audio = np.zeros(300)
    action = np.zeros(300)
    motion[60:120] = 1
    audio[85:100] = 1
    action[95:110] = 1
    strong = assess_candidate(50, 135, scored, {
        "motion_peak": motion, "audio": audio, "action": action})
    assert strong["local_quality"] > weak["local_quality"]
    assert set(strong["signals"]) == {"motion_peak", "audio", "action"}


def test_semantic_result_is_additional_signal_not_absolute_decision():
    result = {"relevante": False, "importancia": 0.1,
              "interes_espectador": 0.1, "consecuencia": 0.1,
              "evento_completo": True}
    assert combine_semantic(0.9, result) < 0.5
    assert combine_semantic(0.9, None) == 0.9


def test_gemini_key_uses_credential_store(monkeypatch):
    import keyring
    from llm.gemini_credentials import delete_key, get_key, save_key
    stored = {}
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(keyring, "set_password", lambda service, account, value:
                        stored.update({(service, account): value}))
    monkeypatch.setattr(keyring, "get_password", lambda service, account:
                        stored.get((service, account)))
    monkeypatch.setattr(keyring, "delete_password", lambda service, account:
                        stored.pop((service, account), None))
    save_key("secret-for-test")
    assert get_key() == "secret-for-test"
    delete_key()
    assert get_key() == ""


def test_gemini_structured_response_and_invalid_fallback(monkeypatch):
    from llm.gemini_client import GeminiClient
    monkeypatch.setenv("GEMINI_API_KEY", "secret-for-test")
    captured = {}
    answer = {"relevante": True, "categoria": "rescate", "importancia": .9,
              "interes_espectador": .8, "consecuencia": .7,
              "evento_completo": True, "motivo": "Salva al equipo"}

    class Response:
        status_code = 200
        ok = True
        def json(self):
            return {"candidates": [{"content": {"parts":
                    [{"text": json.dumps(answer)}]}}]}

    def post(url, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("llm.gemini_client.requests.post", post)
    client = GeminiClient()
    result = client.evaluate_event({"inicio_s": 10},
                                   [{"inlineData": {"mimeType": "image/jpeg", "data": "abc"}}])
    assert result["categoria"] == "rescate"
    assert captured["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert len(captured["json"]["contents"][0]["parts"]) == 2
    assert "secret-for-test" not in str(captured["json"])
    answer["importancia"] = 1.5
    with pytest.raises(RuntimeError, match="evaluación inválida"):
        client.evaluate_event({"inicio_s": 10})


def test_gemini_limits_and_cache_survive_restart(tmp_path, monkeypatch):
    from llm.gemini_event_review import GeminiEventReview
    monkeypatch.setenv("GEMINI_API_KEY", "secret-for-test")
    monkeypatch.setattr("llm.gemini_event_review.storyboard_parts", lambda *a: [])
    calls = []
    evaluation = {"relevante": True, "categoria": "objetivo", "importancia": .9,
                  "interes_espectador": .8, "consecuencia": .8,
                  "evento_completo": True, "motivo": "Objetivo completado"}
    monkeypatch.setattr("llm.gemini_event_review.GeminiClient.evaluate_event",
                        lambda self, context, storyboard: calls.append(context) or evaluation)
    video = tmp_path / "game.mp4"
    video.write_bytes(b"video")
    items = [{"start": 10.0, "end": 55.0, "local_quality": .7,
              "signals": ["motion_peak"], "reason": "test"},
             {"start": 100.0, "end": 160.0, "local_quality": .6,
              "signals": ["audio"], "reason": "test"}]
    first = GeminiEventReview(str(video), cache_dir=str(tmp_path / "gemini"),
                              max_candidates=2, max_calls=1, log_fn=lambda _: None)
    first.review(items)
    assert len(calls) == 1 and first.calls_this_run == 1
    assert first.path.exists()
    assert "secret-for-test" not in first.path.read_text(encoding="utf-8")
    second = GeminiEventReview(str(video), cache_dir=str(tmp_path / "gemini"),
                               max_candidates=2, max_calls=1, log_fn=lambda _: None)
    second.review(items)
    assert len(calls) == 1 and second.cache_hits == 1
    assert json.loads(second.path.read_text(encoding="utf-8"))["cache_hits"] == 1


def test_clip_feedback_and_metrics_are_local(tmp_path):
    video = tmp_path / "game.mp4"
    video.write_bytes(b"game")
    clip = tmp_path / "clip.mp4"
    data = save_rating(str(video), str(clip), 10, 70, "good",
                       {"local_quality": .8, "event_signals": ["motion_peak"]},
                       cache_dir=str(tmp_path))
    add_missed_event(str(video), 200, "rescate", cache_dir=str(tmp_path))
    loaded = load_feedback(str(video), cache_dir=str(tmp_path))
    values = metrics(loaded)
    assert values["reviewed"] == 1
    assert values["perceived_precision"] == 1.0
    assert values["missed_events"] == 1
    assert values["average_duration"] == 60
