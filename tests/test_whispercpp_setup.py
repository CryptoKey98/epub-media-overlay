"""Pip runtime discovery, missing dependencies, and explicit overrides."""
import sys
import types
import pytest
import transcription_backend as setup


def windows(monkeypatch):
    monkeypatch.setattr(setup.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup.platform, "machine", lambda: "AMD64")
    monkeypatch.delenv("WHISPERCPP_BINARY", raising=False)


def test_installed_runtime_without_vendor_or_cache(tmp_path, monkeypatch):
    windows(monkeypatch)
    binary = tmp_path / "whisper-cli.exe"
    for name in (binary.name, "ggml-base.dll", "ggml-cpu.dll", "ggml-vulkan.dll", "ggml.dll", "whisper.dll"):
        (tmp_path / name).write_bytes(b"runtime")
    monkeypatch.setitem(sys.modules, "whispercpp_runtime", types.SimpleNamespace(binary_path=lambda: binary))
    assert setup.resolve_binary() == binary


def test_missing_package_has_install_instructions(monkeypatch):
    windows(monkeypatch)
    monkeypatch.setitem(sys.modules, "whispercpp_runtime", None)
    with pytest.raises(RuntimeError, match="pip install -r requirements.txt"):
        setup.resolve_binary()


def test_missing_dll_has_reinstall_instructions(tmp_path, monkeypatch):
    windows(monkeypatch)
    binary = tmp_path / "whisper-cli.exe"
    binary.write_bytes(b"exe")
    monkeypatch.setitem(sys.modules, "whispercpp_runtime", types.SimpleNamespace(binary_path=lambda: binary))
    with pytest.raises(RuntimeError, match="Incomplete"):
        setup.resolve_binary()


def test_other_platform_requires_explicit_runtime(monkeypatch):
    windows(monkeypatch)
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="WHISPERCPP_BINARY"):
        setup.resolve_binary()


def test_explicit_binary_does_not_require_package(tmp_path, monkeypatch):
    binary = tmp_path / "custom.exe"
    binary.write_bytes(b"custom")
    monkeypatch.setenv("WHISPERCPP_BINARY", str(binary))
    monkeypatch.setitem(sys.modules, "whispercpp_runtime", None)
    assert setup.resolve_binary() == binary
