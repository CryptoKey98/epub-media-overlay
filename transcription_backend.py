"""ASR adapters: MLX, WhisperX, and optional whisper.cpp Vulkan + CPU alignment.

Everything the pipeline knows about the speech engines lives here so
`pipeline_core.transcribe_audio` only calls `transcribe_file`. Models are cached for
the life of the process and released with `release_models()` after the transcribe
stage; whisperx used to reload its ASR and alignment models for every audio chunk.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import wave

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
import importlib
import os
import platform
from typing import Any

# Environment for the ML libraries. These must be set before torch / transformers /
# matplotlib are imported by the backends, which happens lazily inside this module.
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # no HuggingFace fork-parallelism warnings/deadlocks
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # legacy checkpoint loads under newer torch defaults
os.environ["MPLBACKEND"] = "Agg"  # never open a GUI window from a dependency

BACKEND_MLX = "mlx"
BACKEND_WHISPERX = "whisperx"
BACKEND_WHISPERCPP = "whispercpp"
BACKENDS = (BACKEND_MLX, BACKEND_WHISPERX, BACKEND_WHISPERCPP)

DEFAULT_MODEL_BY_BACKEND = {
    BACKEND_MLX: "mlx-community/whisper-large-v3-mlx",
    BACKEND_WHISPERX: "small",
    BACKEND_WHISPERCPP: "small",
}
REQUIRED_MODULE_BY_BACKEND = {
    BACKEND_MLX: "mlx_whisperx",
    BACKEND_WHISPERX: "whisperx",
    BACKEND_WHISPERCPP: "whisperx",  # CPU forced alignment only
}


def detect_transcription_backend() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return BACKEND_MLX
    return BACKEND_WHISPERX


def default_model_for_backend(backend: str) -> str:
    return DEFAULT_MODEL_BY_BACKEND[backend]


def transcribe_file(
    file_path: str,
    model: str,
    language: str,
    backend: str,
    batch_size: int,
) -> dict[str, Any]:
    if backend == BACKEND_MLX:
        return _transcribe_with_mlx(file_path, model, language, batch_size)
    if backend == BACKEND_WHISPERCPP:
        result = transcribe_segments(file_path, model, language)
        whisperx = importlib.import_module("whisperx")
        audio = whisperx.load_audio(file_path)
        return _align_result(whisperx, result, audio, language, "cpu")
    if backend == BACKEND_WHISPERX:
        return _transcribe_with_whisperx(file_path, model, language, batch_size)
    raise ValueError(f"Unknown transcription backend: {backend}")


def apply_mlx_cache_limit(backend: str, cache_gb: float | None) -> float | None:
    """Cap the mlx Metal buffer cache so freed GPU memory is not retained unbounded.

    Returns the applied limit in GB, or None when nothing was applied (non-mlx
    backend, no limit requested, or the installed mlx build lacks the setter).
    """
    if backend != BACKEND_MLX or cache_gb is None:
        return None
    try:
        mx = importlib.import_module("mlx.core")
    except ImportError:
        return None
    limit_bytes = int(cache_gb * 1024**3)
    setter = getattr(mx, "set_cache_limit", None) or getattr(
        getattr(mx, "metal", None), "set_cache_limit", None
    )
    if setter is None:
        return None
    setter(limit_bytes)
    return cache_gb


def _transcribe_with_mlx(
    file_path: str,
    model: str,
    language: str,
    batch_size: int,
) -> dict[str, Any]:
    mlx_whisperx = importlib.import_module("mlx_whisperx")
    return mlx_whisperx.transcribe(
        file_path,
        model=model,
        language=language,
        beam_size=1,
        batch_size=batch_size,
    )


# whisperx model caches, keyed so a different model/device/language loads afresh.
_WHISPERX_MODELS: dict[tuple[str, str, str], Any] = {}
_WHISPERX_ALIGN_MODELS: dict[tuple[str, str], tuple[Any, Any]] = {}


def _import_torch():
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _transcribe_with_whisperx(
    file_path: str,
    model: str,
    language: str,
    batch_size: int,
) -> dict[str, Any]:
    whisperx = importlib.import_module("whisperx")
    torch = _import_torch()

    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model_key = (model, device, compute_type)
    model_obj = _WHISPERX_MODELS.get(model_key)
    if model_obj is None:
        model_obj = whisperx.load_model(model, device, compute_type=compute_type)
        _WHISPERX_MODELS[model_key] = model_obj

    audio = whisperx.load_audio(file_path)
    result = model_obj.transcribe(audio, batch_size=batch_size, language=language)

    return _align_result(whisperx, result, audio, language, device)


def _align_result(whisperx, result, audio, language, device):
    """Share word alignment and its model cache across transcription engines."""
    align_language = result.get("language") or language
    if not result.get("segments"):
        return {"segments": [], "word_segments": [], "language": align_language}
    align_key = (align_language, device)
    if align_key not in _WHISPERX_ALIGN_MODELS:
        _WHISPERX_ALIGN_MODELS[align_key] = whisperx.load_align_model(
            language_code=align_language, device=device
        )
    model_a, metadata = _WHISPERX_ALIGN_MODELS[align_key]

    aligned = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    aligned.setdefault("language", align_language)

    if "word_segments" not in aligned:
        aligned["word_segments"] = [
            word
            for segment in aligned.get("segments", [])
            for word in segment.get("words", [])
        ]
    return aligned


def release_models() -> None:
    """Drop cached models and return their memory. Call once after transcription."""
    _WHISPERX_MODELS.clear()
    _WHISPERX_ALIGN_MODELS.clear()
    gc.collect()
    torch = _import_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# whisper.cpp Vulkan adapter and first-use runtime/model setup.
_LOG = logging.getLogger("generate_epub_overlay.whispercpp")
_ROOT = Path(__file__).resolve().parent / ".whispercpp"


def resolve_binary(model: str | None = None) -> Path:
    configured = os.environ.get("WHISPERCPP_BINARY")
    if configured:
        resolved = shutil.which(configured)
        path = Path(resolved or configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"WHISPERCPP_BINARY does not exist: {path}")
        return path
    return ensure_runtime()


def resolve_model(model: str, previous_model: str | None = None) -> Path:
    path = Path(model).expanduser()
    if path.is_file():
        return path.resolve()
    if path.suffix == ".bin" or path.is_absolute() or len(path.parts) > 1:
        raise FileNotFoundError(f"whisper.cpp model not found: {model}")
    directory = os.environ.get("WHISPERCPP_MODEL_DIR")
    if directory:
        path = Path(directory).expanduser() / f"ggml-{model}.bin"
        if not path.is_file():
            raise FileNotFoundError(f"whisper.cpp model not found in WHISPERCPP_MODEL_DIR: {path}")
        return path.resolve()
    if previous_model:
        previous = Path(previous_model)
        if previous.name == f"ggml-{model}.bin" and previous.is_file():
            return previous.resolve()
    return ensure_model(model)


def runtime_identity(model: str) -> dict:
    """Bind cached transcripts to local model/binary identity and adapter version."""
    def identify(path):
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"adapter_version": 1, "model": identify(resolve_model(model)),
            "executable": identify(resolve_binary(model)), "alignment_device": "cpu"}


def legacy_stamp_compatible(actual: dict, expected: dict) -> bool:
    """Reuse verified old transcripts during the one-time upstream-runtime upgrade.

    New transcription always uses the upstream runtime. This only retains already
    saved words/timestamps when all input/model/alignment identities still match.
    """
    if not isinstance(actual, dict) or expected.get("backend") != "whispercpp":
        return False
    old = actual.get("whispercpp_runtime", {})
    new = expected.get("whispercpp_runtime", {})
    if {k: v for k, v in actual.items() if k != "whispercpp_runtime"} != {
            k: v for k, v in expected.items() if k != "whispercpp_runtime"}:
        return False
    if {k: v for k, v in old.items() if k != "executable"} != {
            k: v for k, v in new.items() if k != "executable"}:
        return False
    target = ensure_runtime()
    if new.get("executable", {}).get("path") != str(target):
        return False
    legacy = _ROOT / "bin" / "whisper-cli.exe"
    if not legacy.is_file():
        return False
    stat = legacy.stat()
    if old.get("executable") != {"path": str(legacy.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
        return False
    # Only the exact originally tested legacy executable is eligible.
    return hashlib.sha256(legacy.read_bytes()).hexdigest() == "2f62973b38f6daeba67f6ed2cd77d52ec8642385baa528a819b35a7e05487b02"


def _run(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not run {Path(args[0]).name}: {exc}") from exc
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} exited with {result.returncode}:\n{result.stderr[-4000:]}")
    return result


def preflight(model: str, logger=None) -> None:
    logger = logger or _LOG
    binary = resolve_binary(model)
    model_path = resolve_model(model)
    result = _run([str(binary), "--help"], timeout=30)
    diagnostics = result.stdout + result.stderr
    if not re.search(r"ggml_vulkan:.*Found [1-9]\d* Vulkan devices", diagnostics):
        # Upstream static builds initialize Vulkan when loading a model, not on --help.
        with tempfile.TemporaryDirectory(prefix="epub-vulkan-probe-") as temporary:
            probe = Path(temporary) / "probe.wav"
            with wave.open(str(probe), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\0" * 3200)
            result = _run([str(binary), "-m", str(model_path), "-f", str(probe),
                           "-d", "1", "-l", "en", "-t", "1"], timeout=120)
            diagnostics = result.stdout + result.stderr
    if not re.search(r"ggml_vulkan:.*Found [1-9]\d* Vulkan devices", diagnostics):
        raise RuntimeError("whispercpp requires a Vulkan build and a detected Vulkan GPU. "
                           "CPU fallback is disabled.\n" + diagnostics[-2000:])
    for line in diagnostics.splitlines():
        if "ggml_vulkan:" in line:
            logger.info("%s", line)
    logger.info("whispercpp: Vulkan transcription; WhisperX word alignment on CPU; batch-size is unused")


def parse_segments(payload: dict, duration: float) -> dict:
    """Convert CLI millisecond offsets to the seconds expected by WhisperX."""
    if not isinstance(payload, dict) or not isinstance(payload.get("transcription"), list):
        raise ValueError("Invalid whisper.cpp JSON: missing transcription list")
    segments = []
    for item in payload["transcription"]:
        try:
            text = item["text"].strip()
            start = float(item["offsets"]["from"]) / 1000
            end = float(item["offsets"]["to"]) / 1000
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise ValueError("Invalid whisper.cpp segment") from exc
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end < start:
            raise ValueError("Invalid whisper.cpp segment offsets")
        end = min(end, duration)
        if text and start < end:
            segments.append({"text": text, "start": start, "end": end})
    return {"segments": segments, "language": payload.get("result", {}).get("language")}


def transcribe_segments(file_path: str, model: str, language: str) -> dict:
    model_path = resolve_model(model)
    binary = resolve_binary(str(model_path))
    with tempfile.TemporaryDirectory(prefix="epub-whispercpp-") as temporary:
        root = Path(temporary)
        wav, output = root / "audio.wav", root / "transcript"
        _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(file_path),
              "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
        with wave.open(str(wav), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        if duration <= 0:
            raise ValueError("Cannot transcribe empty audio")
        result = _run([str(binary), "-m", str(model_path), "-f", str(wav), "-l", language,
                       "-oj", "-of", str(output), "-t", str(min(8, os.cpu_count() or 1))])
        diagnostics = result.stdout + result.stderr
        # Merely detecting a device does not prove that inference selected it.
        if not re.search(r"using Vulkan\d+ backend", diagnostics):
            raise RuntimeError("whisper.cpp did not confirm Vulkan GPU execution; CPU fallback is disabled.\n"
                               + diagnostics[-3000:])
        for line in diagnostics.splitlines():
            if "ggml_vulkan:" in line or "using Vulkan" in line or "total time" in line:
                _LOG.info("%s", line, extra={"file_only": True})
        json_path = output.with_suffix(".json")
        if not json_path.is_file():
            raise RuntimeError("whisper.cpp produced no transcript JSON.\n" + diagnostics[-3000:])
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("Could not read whisper.cpp transcript JSON") from exc
        return parse_segments(payload, duration)


MODEL_REPO = "ggerganov/whisper.cpp"
MODEL_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"


def ensure_runtime() -> Path:
    """Locate the pip-installed runtime; never download or extract executables here."""
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("The packaged whisper.cpp runtime supports Windows x64. "
                           "On other systems, build whisper-cli with Vulkan and set WHISPERCPP_BINARY.")
    try:
        runtime = importlib.import_module("whispercpp_runtime")
    except ModuleNotFoundError as exc:
        if exc.name != "whispercpp_runtime":
            raise
        raise RuntimeError("whisper.cpp runtime is not installed in this Python environment. "
                           "Run python -m pip install -r requirements.txt using the same environment.") from exc
    binary = Path(runtime.binary_path()).resolve()
    required = (binary, *(binary.parent / name for name in
                         ("ggml-base.dll", "ggml-cpu.dll", "ggml-vulkan.dll", "ggml.dll", "whisper.dll")))
    if any(not path.is_file() for path in required):
        raise RuntimeError("Incomplete pip-installed whisper.cpp runtime. Reinstall "
                           "whispercpp-vulkan-runtime from the URL in requirements.txt with --force-reinstall.")
    return binary


def ensure_model(model: str) -> Path:
    model = {"large": "large-v3", "turbo": "large-v3-turbo"}.get(model, model)
    allowed = {"tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
               "large-v1", "large-v2", "large-v3", "large-v3-turbo"}
    allowed |= {name + suffix for name in tuple(allowed) for suffix in ("-q5_0", "-q5_1", "-q8_0")}
    if model not in allowed:
        raise ValueError(f"Unknown whisper.cpp model {model!r}; use a Whisper model name or an existing GGML .bin path.")
    args = dict(repo_id=MODEL_REPO, filename=f"ggml-{model}.bin", revision=MODEL_REVISION)
    try:
        # Completed setup works offline without a network metadata lookup.
        return Path(hf_hub_download(**args, local_files_only=True)).resolve()
    except LocalEntryNotFoundError:
        _LOG.warning("Downloading whisper.cpp model %s to the Hugging Face cache (once)", model)
        try:
            return Path(hf_hub_download(**args)).resolve()
        except Exception as exc:
            raise RuntimeError(f"Could not download GGML model {model!r}. Check the connection, "
                               "or pass an existing .bin file with --model.") from exc
