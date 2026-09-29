"""Vulkan adapter contracts, failures, alignment and cache compatibility."""
import json
from pathlib import Path
import subprocess
import sys
import types
import wave

import pytest
import transcription_backend as cpp
import transcription_backend as tb
import pipeline_core as pc


def _payload():
    return {"result": {"language": "en"}, "transcription": [
        {"text": " Hello world.", "offsets": {"from": 250, "to": 1250}}]}


def test_offsets_are_milliseconds_and_clamped_to_audio():
    result = cpp.parse_segments(_payload(), 1.0)
    assert result == {"language": "en", "segments": [
        {"text": "Hello world.", "start": 0.25, "end": 1.0}]}
    assert cpp.parse_segments({"transcription": []}, 1)["segments"] == []


@pytest.mark.parametrize("payload", [{}, {"transcription": None}, {"transcription": [{}]},
    {"transcription": [{"text": "hi", "offsets": {"from": -1, "to": 4}}]},
    {"transcription": [{"text": "hi", "offsets": {"from": 5, "to": 4}}]},
    {"transcription": [{"text": "hi", "offsets": {"from": 0, "to": float('nan')}}]}])
def test_reject_invalid_json_contract(payload):
    with pytest.raises(ValueError):
        cpp.parse_segments(payload, 2)


def _runtime(tmp_path, monkeypatch):
    binary = tmp_path / "tool with spaces.exe"
    binary.write_bytes(b"binary")
    model = tmp_path / "ggml-small.bin"
    model.write_bytes(b"model")
    monkeypatch.setenv("WHISPERCPP_BINARY", str(binary))
    monkeypatch.setenv("WHISPERCPP_MODEL_DIR", str(tmp_path))
    return binary, model


def test_model_resolution_and_runtime_cache_identity(tmp_path, monkeypatch):
    binary, model = _runtime(tmp_path, monkeypatch)
    assert cpp.resolve_model("small") == model
    assert cpp.resolve_model(str(model)) == model
    with pytest.raises(FileNotFoundError):
        cpp.resolve_model("missing")
    book = {"backend": "whispercpp", "model": "small", "language": "en"}
    first = pc.transcript_compatibility_stamp(book, {})
    model.write_bytes(b"different model")
    second = pc.transcript_compatibility_stamp(book, {})
    assert first != second
    binary.write_bytes(b"new executable")
    assert second != pc.transcript_compatibility_stamp(book, {})
    assert "whispercpp_runtime" not in pc.transcript_compatibility_stamp({"backend": "whisperx"}, {})


@pytest.mark.parametrize("diagnostics,valid", [
    ("ggml_vulkan: Found 1 Vulkan devices:\nggml_vulkan: 0 = AMD Radeon", True),
    ("CPU only", False), ("ggml_vulkan: Found 0 Vulkan devices", False)])
def test_preflight_requires_vulkan_device(tmp_path, monkeypatch, diagnostics, valid):
    _runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(cpp, "_run", lambda *a, **k: subprocess.CompletedProcess([], 0, "", diagnostics))
    if valid:
        cpp.preflight("small")
    else:
        with pytest.raises(RuntimeError, match="Vulkan"):
            cpp.preflight("small")


@pytest.mark.parametrize("gpu,write_json", [(True, True), (False, True), (True, False)])
def test_transcription_requires_gpu_and_output_and_cleans_temp(tmp_path, monkeypatch, gpu, write_json):
    binary, model = _runtime(tmp_path, monkeypatch)
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        if args[0] == "ffmpeg":
            with wave.open(args[-1], "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b'\0' * 64000)
            return subprocess.CompletedProcess(args, 0, "", "")
        if write_json:
            Path(args[args.index("-of")+1]+".json").write_text(json.dumps(_payload()))
        diagnostics = "whisper_backend_init_gpu: using Vulkan0 backend" if gpu else "using CPU backend"
        return subprocess.CompletedProcess(args, 0, "", diagnostics)
    monkeypatch.setattr(cpp, "_run", run)
    if gpu and write_json:
        result = cpp.transcribe_segments("audio with spaces.m4a", "small", "en")
        assert result["segments"][0]["start"] == 0.25
    else:
        with pytest.raises(RuntimeError):
            cpp.transcribe_segments("audio with spaces.m4a", "small", "en")
    assert calls[0][calls[0].index("-i")+1] == "audio with spaces.m4a"
    assert calls[1][0] == str(binary)
    assert not Path(calls[0][-1]).exists()


def test_process_failures_are_reported(monkeypatch):
    monkeypatch.setattr(cpp.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 7, "", "device lost"))
    with pytest.raises(RuntimeError, match="device lost"):
        cpp._run(["whisper-cli"])
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("whisper-cli", 3)
    monkeypatch.setattr(cpp.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="Could not run"):
        cpp._run(["whisper-cli"])


def test_cpp_uses_cpu_alignment_and_shared_cache(monkeypatch):
    calls = []
    def load_align_model(language_code, device):
        calls.append((language_code, device))
        return object(), {}
    def align(segments, model, metadata, audio, device, return_char_alignments):
        assert device == "cpu"
        assert segments[0]["start"] == 0.25
        return {"segments": [{"words": [{"word": "Hello", "start": .25, "end": .8}]}]}
    fake = types.SimpleNamespace(load_audio=lambda p: [], load_align_model=load_align_model, align=align)
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    monkeypatch.setattr(tb, "_WHISPERX_ALIGN_MODELS", {})
    monkeypatch.setattr(cpp, "transcribe_segments", lambda *a: cpp.parse_segments(_payload(), 2))
    for _ in range(2):
        result = tb.transcribe_file("audio.m4a", "small", "en", "whispercpp", 1)
        assert result["word_segments"][0]["word"] == "Hello"
    assert calls == [("en", "cpu")]


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        tb.transcribe_file("audio", "small", "en", "typo", 1)


def test_cli_accepts_backend_and_resolves_model_path(tmp_path, monkeypatch):
    import generate_epub_overlay as geo
    _, model = _runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["generate_epub_overlay.py", "--audio", "sample.m4b",
        "--epub", "sample.epub", "--backend", "whispercpp", "--model", "small"])
    config = geo.parse_args()
    assert config.backend == "whispercpp"
    assert config.model == str(model)


def test_vulkan_execution_is_logged_to_pipeline_logger(tmp_path, monkeypatch, caplog):
    import logging
    _runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(cpp, "_run", lambda *a, **k: subprocess.CompletedProcess([], 0, "",
        "ggml_vulkan: Found 1 Vulkan devices:\nggml_vulkan: 0 = AMD Radeon"))
    with caplog.at_level(logging.INFO, logger="generate_epub_overlay"):
        cpp.preflight("small")
    assert "AMD Radeon" in caplog.text


def test_model_cache_and_legacy_resume(tmp_path, monkeypatch):
    import transcription_backend as setup
    cached = tmp_path / "hub" / "ggml-small.bin"
    cached.parent.mkdir()
    cached.write_bytes(b"model")
    monkeypatch.delenv("WHISPERCPP_MODEL_DIR", raising=False)
    monkeypatch.setattr(setup, "ensure_model", lambda model: cached)
    assert cpp.resolve_model("small") == cached
    legacy = tmp_path / "legacy" / "ggml-small.bin"
    legacy.parent.mkdir()
    legacy.write_bytes(b"previous model")
    assert cpp.resolve_model("small", previous_model=str(legacy)) == legacy
    assert cpp.resolve_model("base", previous_model=str(legacy)) == cached


def test_huggingface_cached_model_does_not_need_network(tmp_path, monkeypatch):
    import transcription_backend as setup
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["local_files_only"] is True
        return str(model)
    monkeypatch.setattr(setup, "hf_hub_download", download)
    assert setup.ensure_model("small") == model
    assert len(calls) == 1
    assert calls[0]["revision"] == setup.MODEL_REVISION


def test_huggingface_first_use_download(tmp_path, monkeypatch):
    import transcription_backend as setup
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise setup.LocalEntryNotFoundError("not cached")
        return str(model)
    monkeypatch.setattr(setup, "hf_hub_download", download)
    assert setup.ensure_model("small") == model
    assert len(calls) == 2


def test_legacy_model_uses_new_upstream_runtime(tmp_path, monkeypatch):
    import transcription_backend as setup
    monkeypatch.delenv("WHISPERCPP_BINARY", raising=False)
    monkeypatch.setattr(cpp, "_ROOT", tmp_path)
    binary = tmp_path / "new-upstream.exe"
    monkeypatch.setattr(setup, "ensure_runtime", lambda: binary)
    assert cpp.resolve_binary(str(tmp_path / "models" / "ggml-small.bin")) == binary


def test_legacy_stamp_reuse_requires_same_inputs_model_and_known_binary(tmp_path, monkeypatch):
    import copy
    import transcription_backend as setup
    monkeypatch.setattr(cpp, '_ROOT', tmp_path / 'legacy')
    monkeypatch.setattr(setup, 'ensure_runtime', lambda: tmp_path / 'installed' / 'whisper-cli.exe')
    legacy = cpp._ROOT / 'bin' / 'whisper-cli.exe'
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b'legacy')
    stat = legacy.stat()
    old = {'kind':'transcript', 'backend':'whispercpp', 'model':'small', 'language':'en',
           'chunk_stamp':{'id':1}, 'whispercpp_runtime':{'adapter_version':1, 'alignment_device':'cpu',
           'model':{'path':'model.bin','size':3,'mtime_ns':42},
           'executable':{'path':str(legacy),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}}}
    new = copy.deepcopy(old)
    new['whispercpp_runtime']['executable'] = {'path':str(setup.ensure_runtime())}
    assert not cpp.legacy_stamp_compatible(old, new), 'Unknown binary must not be accepted'
    monkeypatch.setattr(cpp.hashlib, 'sha256', lambda data: types.SimpleNamespace(hexdigest=lambda:
        '2f62973b38f6daeba67f6ed2cd77d52ec8642385baa528a819b35a7e05487b02'))
    assert cpp.legacy_stamp_compatible(old, new)
    changed = copy.deepcopy(new)
    changed['language'] = 'fr'
    assert not cpp.legacy_stamp_compatible(old, changed)
    changed = copy.deepcopy(new)
    changed['whispercpp_runtime']['model']['size'] = 4
    assert not cpp.legacy_stamp_compatible(old, changed)
    assert not cpp.legacy_stamp_compatible(None, new)


def test_upstream_static_build_probes_gpu_after_help(tmp_path, monkeypatch):
    _runtime(tmp_path, monkeypatch)
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        if '--help' in args:
            return subprocess.CompletedProcess(args, 0, 'usage', '')
        assert Path(args[args.index('-f') + 1]).is_file()
        return subprocess.CompletedProcess(args, 0, '', 'ggml_vulkan: Found 1 Vulkan devices:')
    monkeypatch.setattr(cpp, '_run', run)
    cpp.preflight('small')
    assert len(calls) == 2
