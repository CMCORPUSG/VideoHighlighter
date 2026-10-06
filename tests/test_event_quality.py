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
              "contexto_suficiente": True, "evento_completo": True,
              "motivo": "Salva al equipo"}

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
    with pytest.raises(RuntimeError, match="schema inválido"):
        client.evaluate_event({"inicio_s": 10})


def test_gemini_limits_and_cache_survive_restart(tmp_path, monkeypatch):
    from llm.gemini_event_review import GeminiEventReview
    monkeypatch.setenv("GEMINI_API_KEY", "secret-for-test")
    monkeypatch.setattr("llm.gemini_event_review.storyboard_parts",
                        lambda *a, **k: [{"inlineData": {"data": "frame"}}] * 3)
    calls = []
    evaluation = {"relevante": True, "categoria": "objetivo", "importancia": .9,
                  "interes_espectador": .8, "consecuencia": .8,
                  "contexto_suficiente": True, "evento_completo": True,
                  "motivo": "Objetivo completado"}
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


@pytest.mark.parametrize("failure,kind,retryable", [
    ("timeout", "timeout", True), (429, "HTTP 429", True),
    (400, "HTTP 400", False), (401, "HTTP 401", False),
    (403, "HTTP 403", False), (404, "HTTP 404", False),
    (503, "HTTP 503", True),
])
def test_gemini_safe_error_categories(monkeypatch, failure, kind, retryable):
    import requests
    from llm.gemini_client import GeminiClient, GeminiEvaluationError
    monkeypatch.setenv("GEMINI_API_KEY", "secret-for-test")

    class Response:
        status_code = failure
        ok = False

    def post(*args, **kwargs):
        if failure == "timeout":
            raise requests.exceptions.ReadTimeout("secret-for-test")
        return Response()

    monkeypatch.setattr("llm.gemini_client.requests.post", post)
    with pytest.raises(GeminiEvaluationError) as caught:
        GeminiClient().evaluate_event({})
    assert caught.value.kind == kind
    assert caught.value.retryable is retryable
    assert "secret-for-test" not in str(caught.value)


def test_gemini_retry_is_bounded_and_cached(tmp_path, monkeypatch):
    from llm.gemini_client import GeminiEvaluationError
    from llm.gemini_event_review import GeminiEventReview
    monkeypatch.setenv("GEMINI_API_KEY", "secret-for-test")
    monkeypatch.setattr("llm.gemini_event_review.storyboard_parts",
                        lambda *a, **k: [{"inlineData": {"data": "frame"}}] * 3)
    calls = []
    evaluation = {"relevante": False, "categoria": "actividad_normal",
                  "importancia": .1, "interes_espectador": .1, "consecuencia": .1,
                  "contexto_suficiente": True, "evento_completo": True,
                  "motivo": "Caminar"}

    def evaluate(self, context, frames):
        calls.append(1)
        if len(calls) == 1:
            raise GeminiEvaluationError("HTTP 429", "Límite temporal.", retryable=True)
        return evaluation

    monkeypatch.setattr("llm.gemini_event_review.GeminiClient.evaluate_event", evaluate)
    video = tmp_path / "game.mp4"
    video.write_bytes(b"video")
    item = {"start": 10., "end": 60., "local_quality": .8,
            "signals": ["motion_peak"], "reason": "test"}
    waits = []
    logs = []
    reviewer = GeminiEventReview(str(video), cache_dir=str(tmp_path / "gemini"),
                                  max_candidates=1, max_calls=2,
                                  sleep_fn=waits.append, log_fn=logs.append)
    status = reviewer.review([item])
    assert len(calls) == 2 and waits == [10]
    assert status["calls"] == 2 and status["rejected"] == 1
    assert any("HTTP 429" in log and "reintento" in log for log in logs)
    restarted = GeminiEventReview(str(video), cache_dir=str(tmp_path / "gemini"),
                                   max_candidates=1, max_calls=2,
                                   log_fn=lambda _: None)
    assert restarted.review([item])["cached"] == 1
    assert len(calls) == 2
    assert restarted._candidate_key(item, {"x": 1}, []) != restarted._candidate_key(
        item, {"x": 2}, [])
    assert restarted._candidate_key(item, {"x": 1}, []) != restarted._candidate_key(
        item, {"x": 1}, [{"inlineData": {"data": "different"}}])


def test_semantic_rejection_prevents_local_refill():
    from modules.segments.event_segments import build_event_segments
    score = np.zeros(240)
    score[50:60] = 10
    score[160:170] = 10

    def reject_all(items):
        for item in items:
            item["gemini"] = {"relevante": False, "categoria": "caminar",
                              "importancia": .1, "interes_espectador": .1,
                              "consecuencia": .1, "contexto_suficiente": True,
                              "evento_completo": True, "motivo": "Rutina"}

    clips, _ = build_event_segments(video_duration=240, score=score,
                                    semantic_review=reject_all,
                                    log_fn=lambda _: None)
    assert clips == []


def test_provider_labels_and_narration_gate():
    from modules.narration.story_run import provider_allows_narration
    from modules.ui.gemini_settings import ai_status_label
    assert ai_status_label("none") == "Ninguna"
    assert ai_status_label("ollama") == "Ollama local"
    assert ai_status_label("gemini", {"reason": "activo", "valid": 2,
                                      "planned": 2}) == "Gemini — activo"
    assert ai_status_label("gemini", {"reason": "parcial", "valid": 1,
                                      "planned": 2}) == "Gemini — parcial (1/2 candidatos)"
    assert "fallo" in ai_status_label("gemini", {"reason": "HTTP 503"})
    assert not provider_allows_narration({"ai_provider": "none"})
    assert not provider_allows_narration({"ai_provider": "gemini"})
    assert provider_allows_narration({"ai_provider": "ollama"})


def test_two_candidate_diagnostic_reads_only_matching_video_cache(tmp_path):
    import hashlib
    from modules.ui.gemini_diagnostic import cached_candidates
    video = tmp_path / "game.mp4"
    video.write_bytes(b"game")
    stat = video.stat()
    digest = hashlib.sha256(
        f"{video.absolute()}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()
    cache = {"video_hash": digest, "cache_complete": True,
             "highlight_segments": [[10, 40], [50, 100], [120, 180]],
             "highlight_metadata": {"segments_metadata": [
                 {"local_quality": .2}, {"local_quality": .9},
                 {"local_quality": .8}]}}
    (tmp_path / f"{digest}.signature.cache.json").write_text(
        json.dumps(cache), encoding="utf-8")
    selected = cached_candidates(str(video), str(tmp_path))
    assert len(selected) == 2
    assert [item["start"] for item in selected] == [50, 120]
    other = tmp_path / "other.mp4"
    other.write_bytes(b"other")
    assert cached_candidates(str(other), str(tmp_path)) == []
