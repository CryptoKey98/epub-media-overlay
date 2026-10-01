import types
import pytest
import transcription_backend as tb
import pipeline_core as core


def test_asr_cache_reuses_and_invalidates(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"first")
    calls = []
    monkeypatch.setattr(tb, "runtime_identity", lambda model: {"model": model})
    monkeypatch.setattr(tb, "transcribe_segments", lambda *args: calls.append(args) or {"segments": []})
    tb._cached_cpp_segments(str(audio), "small", "en")
    tb._cached_cpp_segments(str(audio), "small", "en")
    assert len(calls) == 1
    audio.write_bytes(b"other")
    tb._cached_cpp_segments(str(audio), "small", "en")
    tb._cached_cpp_segments(str(audio), "small", "fr")
    assert len(calls) == 3


def test_stamp_alignment_selection(monkeypatch):
    monkeypatch.setattr(tb, "runtime_identity", lambda model: {})
    info = {"backend": "whispercpp", "model": "small"}
    auto = core.transcript_compatibility_stamp(info, {})
    assert auto == core.transcript_compatibility_stamp(dict(info, alignment_backend="cpu"), {})
    assert auto != core.transcript_compatibility_stamp(dict(info, alignment_backend="directml"), {})


def test_directml_failure_falls_back_once():
    adapter = object.__new__(tb._DirectMLAlignment)
    adapter.failed = False
    calls = []
    def fail(*args):
        calls.append(1)
        raise RuntimeError("GPU unavailable")
    adapter.session = types.SimpleNamespace(run=fail)
    adapter.model = lambda waveform, lengths=None: ("cpu", None)
    import torch
    wave = torch.zeros(1, 16000)
    assert adapter(wave) == ("cpu", None)
    assert adapter(wave) == ("cpu", None)
    assert len(calls) == 1


def test_whisperx_cpu_override_keeps_asr_device(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(load_audio=lambda p: [], load_model=lambda *a, **kw: types.SimpleNamespace(transcribe=lambda *a, **kw: {"segments": []}))
    monkeypatch.setattr(tb.importlib, "import_module", lambda name: fake)
    monkeypatch.setattr(tb, "_import_torch", lambda: types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(tb, "_WHISPERX_MODELS", {})
    monkeypatch.setattr(tb, "_align_result", lambda *args: calls.append(args[-1]) or {})
    tb.transcribe_file("unused", "small", "en", "whisperx", 1, "cpu")
    tb.transcribe_file("unused", "small", "en", "whisperx", 1, "auto")
    assert calls == ["cpu", "cuda"]
    assert ("small", "cuda", "float16") in tb._WHISPERX_MODELS


@pytest.mark.parametrize("backend", ["whispercpp", "whisperx"])
@pytest.mark.parametrize("option", [None, "auto", "cpu", "directml"])
def test_cli_alignment_choices(tmp_path, monkeypatch, option, backend):
    import sys
    import generate_epub_overlay as cli
    audio = tmp_path / "audio.wav"
    epub = tmp_path / "book.epub"
    audio.touch()
    epub.touch()
    monkeypatch.setattr(tb, "resolve_model", lambda *args, **kwargs: tmp_path / "model.bin")
    args = ["program", "--audio", str(audio), "--epub", str(epub), "--backend", backend, "--language", "en"]
    if option is not None:
        args += ["--alignment-backend", option]
    monkeypatch.setattr(sys, "argv", args)
    assert cli.parse_args().alignment_backend == (option or "auto")


@pytest.mark.parametrize("cuda", [False, True])
def test_whisperx_directml_preserves_transcription_device(monkeypatch, cuda):
    calls = []
    fake = types.SimpleNamespace(load_audio=lambda p: [], load_model=lambda *a, **kw: types.SimpleNamespace(transcribe=lambda *a, **kw: {"segments": []}))
    monkeypatch.setattr(tb.importlib, "import_module", lambda name: fake)
    monkeypatch.setattr(tb, "_import_torch", lambda: types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda)))
    monkeypatch.setattr(tb, "_WHISPERX_MODELS", {})
    monkeypatch.setattr(tb, "_align_result", lambda *args: calls.append(args[-2:]) or {})
    tb.transcribe_file("unused", "small", "en", "whisperx", 1, "directml")
    assert calls == [("cpu", "directml")]
    assert ("small", "cuda" if cuda else "cpu", "float16" if cuda else "int8") in tb._WHISPERX_MODELS


@pytest.mark.parametrize("backend,language", [("mlx", "en"), ("whisperx", "fr")])
def test_directml_rejects_unsupported_combinations(backend, language):
    with pytest.raises(ValueError, match="DirectML"):
        tb.transcribe_file("unused", "small", language, backend, 1, "directml")
