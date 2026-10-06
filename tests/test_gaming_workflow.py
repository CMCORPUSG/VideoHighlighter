from modules.segments.simple_run import apply_gaming_run
from modules.media.video_cache import VideoAnalysisCache
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


def test_nvenc_mode_only_attempts_nvidia_then_cpu(monkeypatch):
    encoder_select._chain_cache.clear()
    monkeypatch.setattr(encoder_select, "probe_video_size", lambda *a: (1920, 1080))
    monkeypatch.setattr(encoder_select, "_available_encoders", lambda *a: "h264_nvenc h264_qsv h264_amf")
    monkeypatch.setattr(encoder_select, "preferred_gpu_vendor", lambda: "intel")
    names = [name for name, _ in encoder_select.encoder_chain("game.mp4", mode="nvenc")]
    assert names == ["h264_nvenc", "libx264"]


def test_ui_translation_keeps_combo_data_and_can_switch_back():
    from PySide6.QtWidgets import QApplication, QComboBox, QPushButton, QWidget
    from modules.ui.i18n import translate_tree
    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    button = QPushButton("Analyze", parent)
    combo = QComboBox(parent)
    combo.addItem("Full video", "full")
    translate_tree(parent, "es")
    assert button.text() == "Analizar"
    assert combo.itemText(0) == "Video completo"
    assert combo.itemData(0) == "full"
    translate_tree(parent, "en")
    assert button.text() == "Analyze"
    assert combo.itemText(0) == "Full video"
    assert combo.itemData(0) == "full"


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
