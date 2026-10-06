from modules.segments.simple_run import apply_gaming_run
from modules.media.video_cache import VideoAnalysisCache, build_analysis_cache_params
from modules.system import encoder_select


def test_gaming_preset_prioritizes_visual_signals_and_keeps_clips():
    settings = {"highlight_objects": None, "interesting_actions": None,
                "render_mode": "auto"}
    apply_gaming_run(settings, 15)
    assert settings["max_duration"] == 900
    assert settings["motion_peak_points"] > settings["audio_peak_points"]
    assert settings["transcript_points"] == 0
    assert settings["use_transcript"] is False
    assert settings["object_points"] == 0
    assert settings["action_points"] == 0
    assert settings["export_separate_clips"] is True
    assert settings["render_mode"] == "auto"
    assert settings["clip_time"] == 0
    assert settings["event_mode"] is True


def test_gaming_event_keeps_fight_result_and_reaction_across_scene_cuts():
    import numpy as np
    from modules.segments.event_segments import build_event_segments
    score = np.zeros(500)
    activity = list(range(100, 235, 6)) + [240, 247, 255]
    score[activity] = 8
    segments, _ = build_event_segments(
        video_duration=500, score=score,
        scenes=[(0, 125), (125, 220), (220, 245), (245, 500)],
        motion_peaks=activity, target_duration=180, log_fn=lambda _: None)
    assert len(segments) == 1
    assert segments[0][0] <= 88
    assert segments[0][1] >= 270


def test_gaming_events_split_at_real_lull_and_never_trim_for_budget():
    import numpy as np
    from modules.segments.event_segments import build_event_segments
    score = np.zeros(600)
    first = list(range(100, 180, 5))
    second = list(range(350, 475, 5))
    score[first + second] = 10
    segments, _ = build_event_segments(
        video_duration=600, score=score,
        motion_peaks=first + second, target_duration=190,
        log_fn=lambda _: None)
    assert len(segments) == 1
    assert segments[0][0] <= 338
    assert segments[0][1] >= 490
    assert segments[0][1] - segments[0][0] > 140


def test_gaming_event_without_temporal_evidence_is_not_invented():
    import numpy as np
    from modules.segments.event_segments import build_event_segments
    segments, _ = build_event_segments(
        video_duration=200, score=np.ones(200) * 5,
        scenes=[(0, 100), (100, 200)], target_duration=120,
        log_fn=lambda _: None)
    assert segments == []


def test_gaming_dense_background_does_not_join_whole_vod():
    import numpy as np
    from modules.segments.event_segments import build_event_segments
    score = np.full(1200, 4.0)
    score[100:150] = 20
    score[650:730] = 24
    segments, _ = build_event_segments(
        video_duration=1200, score=score, target_duration=240,
        log_fn=lambda _: None)
    assert len(segments) == 2
    assert segments[0][1] < 250
    assert segments[1][0] > 550


def test_event_mode_uses_motion_peaks_when_saved_scoring_only_has_scene_cuts():
    import numpy as np
    from modules.segments.event_segments import build_event_segments
    score = np.zeros(500)
    score[[0, 100, 200, 300, 400]] = 1
    peaks = list(range(120, 180, 4))
    segments, _ = build_event_segments(
        video_duration=500, score=score,
        scenes=[(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)],
        motion_peaks=peaks, target_duration=120, log_fn=lambda _: None)
    assert len(segments) == 1
    assert segments[0][0] < 120 < segments[0][1]


def test_event_swap_preserves_variable_boundaries_and_undo():
    import numpy as np
    from modules.segments.highlight_swap import SwapSession
    score = np.zeros(500)
    score[80:125] = 12
    score[310:390] = 10
    session = SwapSession(score, [(68, 140)], video_duration=500,
                          clip_time=10, segmentation_mode="events")
    assert session.swap(0)
    assert len(session.segments) == 1
    start, end = session.segments[0]
    assert start < 310 < 390 < end
    assert end - start > 80
    assert session.undo()
    assert session.segments == [(68.0, 140.0)]


def test_stage_checkpoint_requires_same_video_and_parameters(tmp_path):
    video = tmp_path / "game.mp4"
    video.write_bytes(b"first video")
    cache = VideoAnalysisCache(cache_dir=str(tmp_path / "cache"))
    params = {"detector": "n", "sample_rate": 5}
    cache.save_stage(str(video), params, "motion", {"motion_peaks": [1.0]})
    assert cache.load_stages(str(video), params)["motion"]["motion_peaks"] == [1.0]
    assert cache.load_stages(str(video), {**params, "sample_rate": 8}) == {}
    video.write_bytes(b"second video, longer")
    assert cache.load_stages(str(video), params) == {}


def test_analysis_signature_tracks_detector_backend():
    base = {"interesting_actions": ["victory"], "action_backend": "auto",
            "object_confidence": 0.3}
    first = build_analysis_cache_params(base, {}, 5, 3600)
    second = build_analysis_cache_params({**base, "action_backend": "r3d_cpu"}, {}, 5, 3600)
    third = build_analysis_cache_params({**base, "object_confidence": 0.6}, {}, 5, 3600)
    assert first != second
    assert first != third


def test_nvenc_mode_only_attempts_nvidia_then_cpu(monkeypatch):
    encoder_select._chain_cache.clear()
    monkeypatch.setattr(encoder_select, "probe_video_size", lambda *a: (1920, 1080))
    monkeypatch.setattr(encoder_select, "_available_encoders", lambda *a: "h264_nvenc h264_qsv h264_amf")
    monkeypatch.setattr(encoder_select, "preferred_gpu_vendor", lambda: "intel")
    names = [name for name, _ in encoder_select.encoder_chain("game.mp4", mode="nvenc")]
    assert names == ["h264_nvenc", "libx264"]


def test_ui_translation_keeps_combo_data_and_can_switch_back():
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QApplication, QComboBox, QPushButton, QWidget
    from modules.ui.i18n import translate_tree
    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    button = QPushButton("Analyze", parent)
    button.setToolTip("Click a face to find all segments where they appear.")
    combo = QComboBox(parent)
    combo.addItem("Full video", "full")
    action = QAction("Export", parent)
    translate_tree(parent, "es")
    assert button.text() == "Analizar"
    assert button.toolTip().startswith("Haz clic en un rostro")
    assert combo.itemText(0) == "Video completo"
    assert combo.itemData(0) == "full"
    assert action.text() == "Exportar"
    translate_tree(parent, "en")
    assert button.text() == "Analyze"
    assert button.toolTip().startswith("Click a face")
    assert combo.itemText(0) == "Full video"
    assert combo.itemData(0) == "full"
    assert action.text() == "Export"


def test_fixed_mode_status_translates_with_duration():
    from PySide6.QtWidgets import QApplication, QLabel
    from modules.ui.i18n import translate_tree
    app = QApplication.instance() or QApplication([])
    label = QLabel("✂️ Fixed mode: each highlight clip will be 15s long.")
    translate_tree(label, "es")
    assert label.text() == "✂️ Modo fijo: cada clip durará 15 s."


def test_gemini_sends_only_bounded_text_context(monkeypatch):
    import pytest
    from llm.gemini_client import GeminiClient
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-value")
    captured = {}

    class Response:
        status_code = 200
        ok = True

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "Resumen listo"}]}}]}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("llm.gemini_client.requests.post", fake_post)
    client = GeminiClient()
    client.load()
    assert client.query("Resume", analysis_data={"scenes": []}) == "Resumen listo"
    assert captured["headers"]["x-goog-api-key"] == "secret-test-value"
    assert "secret-test-value" not in str(captured["json"])
    with pytest.raises(RuntimeError, match="no se envían fotogramas"):
        client.query("Describe el fotograma", frame_base64="fake-frame")
