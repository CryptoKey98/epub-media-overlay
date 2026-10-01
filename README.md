# epub-media-overlay

Generate a Media Overlay EPUB from an audiobook file and a source `.epub`.

This project provides a command-line pipeline that automatically resumes prior work when possible and:

- prepares a working EPUB copy
- splits audiobook input into audio chunks
- transcribes each chunk with a platform-appropriate backend
- matches transcript chunks to EPUB HTML files
- injects stable segment ids into HTML
- generates SMIL overlays
- packages the final Media Overlay EPUB
- validates the generated result

## Requirements and installation

- Python 3.10 or later, subject to the selected backend's supported Python versions.
- FFmpeg, including both `ffmpeg` and `ffprobe`, installed on `PATH`.
- Internet access on first use to download transcription/alignment models and NLTK data.
- Enough disk space for models, split audio, temporary WAV conversion, and the output EPUB.

All three backends are integrated in `transcription_backend.py`:

| Backend option | Transcription | Word alignment | Selection |
| --- | --- | --- | --- |
| `whisperx` | NVIDIA CUDA when available, otherwise CPU | WhisperX on the selected device; experimental DirectML available | Default outside Apple Silicon |
| `mlx` | Apple Silicon using MLX | mlx-whisperx | Default on Apple Silicon |
| `whispercpp` | Vulkan GPU, including AMD | WhisperX on CPU by default; experimental DirectML available | Select explicitly |

For the pip-installed whispercpp runtime, use **Windows x64, an AVX2/FMA/F16C-capable CPU, a Vulkan-capable GPU with its graphics driver installed, and the Microsoft Visual C++ v14 x64 Redistributable**. CUDA, a compiler, and the Vulkan SDK are not needed to run it. Automatic runtime installation is Windows x64 only; other systems require a compatible Vulkan executable through `WHISPERCPP_BINARY` and the WhisperX alignment dependency.

Install Python dependencies into your virtual environment **from the repository root**
(the requirements include a bundled dependency wheel):

```bash
python -m pip install -r requirements.txt
```

For a fresh Windows checkout, from the repository directory (PowerShell):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ffmpeg -version
ffprobe -version
```

For an existing Windows environment with standard ONNX Runtime installed, follow
the [runtime migration steps](#experimental-directml-setup) before reinstalling. `requirements.txt` includes `huggingface-hub` for whispercpp model downloads and the Windows x64 `whispercpp-vulkan-runtime` wheel for the executable and DLLs. **Keep WhisperX installed when using whispercpp:** it supplies the word alignment model and timestamp processing. FFmpeg and GPU drivers are system requirements, not Python packages.

Pip installs the whispercpp executable and DLLs into the active environment and creates `Scripts/whispercpp.exe`. Models download separately into the Hugging Face cache on first use. No vendor archive or separate runtime installer is required. See [AMD GPU setup and storage](#amd-gpu-transcription-with-whispercpp-optional) below.

The pipeline downloads the required NLTK tokenizer data automatically on first run and caches it under `~/.cache/epub-media-overlay/nltk_data`.

## Usage

### Examples

Basic run:

```bash
python generate_epub_overlay.py \
  --audio /path/to/book.m4b \
  --epub /path/to/book.epub
```

Parameter template with every CLI option shown:

```bash
  python generate_epub_overlay.py \
  --audio /path/to/book.m4b \
  --epub /path/to/book.epub \
  --backend mlx \
  --output-dir /path/to \
  --work-dir /path/to/.book.epubmo \
  --model mlx-community/whisper-small-mlx \
  --language en \
  --audio-extension .m4a \
  --audio-codec aac \
  --audio-bitrate 64k \
  --audio-sample-rate 24000 \
  --audio-channels 1 \
  --split-jobs 4 \
  --chunk-seconds 600 \
  --batch-size 1 \
  --mlx-cache-gb 8
```

### Defaults

- transcription backend:
  - Apple Silicon macOS: `mlx-whisperx`
  - other platforms: `whisperx`
- available backend overrides:
  - `mlx`
  - `whisperx`
  - `whispercpp` (Vulkan GPU; opt-in)
- transcription model:
  - Apple Silicon macOS: `mlx-community/whisper-large-v3-mlx`
  - other platforms: `small`
- language: `en`
- audio extension: `.m4a`
- split audio codec: `copy`
- AAC split audio bitrate: `64k` when `--audio-codec aac`
- AAC split audio sample rate: `24000` when `--audio-codec aac`
- AAC split audio channels: `1` (mono) when `--audio-codec aac`
- split jobs with `--audio-codec copy`: `1`
- split jobs with `--audio-codec aac`: `max(1, cpu_count - 2)`, capped at CPU count
- transcription batch size: `1`

With `--audio-codec copy`:

- split audio preserves the source audio stream without re-encoding
- packaged Media Overlay playback keeps source-quality audio
- splitting is usually faster than AAC re-encoding
- final EPUB size may increase because the source-quality audio is packaged

The `--language` setting is used for both transcription and HTML sentence segmentation.

### Parameters

`--audio`

- Required.
- Path to the source audiobook file.
- Any audio extension is accepted as long as `ffprobe` and `ffmpeg` can read it.

`--epub`

- Required.
- Path to the source ebook file.
- Must point to an `.epub` file.

`--output-dir`

- Optional.
- Directory where the final `<book-stem>.media-overlay.epub` file is written.
- Default: the source EPUB directory.

`--work-dir`

- Optional.
- Directory used for persistent state, logs, transcripts, split audio, and intermediate EPUB artifacts.
- Default: `<output-dir>/.<book-stem>.epubmo`

`--model`

- Optional.
- Transcription model identifier.
- Default:
  - Apple Silicon macOS: `mlx-community/whisper-large-v3-mlx`
  - other platforms: `small`
- Valid model names depend on the active backend.
- Common Apple Silicon macOS values:
  - `mlx-community/whisper-tiny-mlx`
  - `mlx-community/whisper-tiny.en-mlx`
  - `mlx-community/whisper-base-mlx`
  - `mlx-community/whisper-base.en-mlx`
  - `mlx-community/whisper-small-mlx`
  - `mlx-community/whisper-small.en-mlx`
  - `mlx-community/whisper-medium-mlx`
  - `mlx-community/whisper-medium.en-mlx`
  - `mlx-community/whisper-large-mlx`
  - `mlx-community/whisper-large-v1-mlx`
  - `mlx-community/whisper-large-v2-mlx`
  - `mlx-community/whisper-large-v3-mlx`
  - `mlx-community/whisper-large-v3-turbo`
- Common values on other platforms:
  - `tiny`
  - `tiny.en`
  - `base`
  - `base.en`
  - `small`
  - `small.en`
  - `medium`
  - `medium.en`
  - `large`
  - `large-v1`
  - `large-v2`
  - `large-v3`
  - `turbo`
- `--model` can also point to a compatible local model path instead of one of the common identifiers above.

`--backend`

- Optional.
- Overrides automatic backend detection.
- Supported values: `mlx`, `whisperx`, `whispercpp`
- Default:
  - Apple Silicon macOS: `mlx`
  - other platforms: `whisperx`

`--alignment-backend`

- Optional; controls word alignment independently of the transcription backend.
- Supported values: `auto`, `cpu`, `directml`
- Default: `auto`. Omitting the flag is equivalent to explicitly selecting `auto`.
- `auto` preserves the existing backend behavior: WhisperX uses CUDA when available, otherwise CPU; whisper.cpp uses CPU alignment; MLX uses its existing alignment.
- `cpu` forces CPU word alignment for WhisperX or whisper.cpp without changing the transcription device.
- `directml` enables experimental GPU alignment on Windows; currently requires `--backend whispercpp` or `--backend whisperx`, with `--language en` and the optional DirectML dependencies.
- MLX supports only `auto`.
- See [Experimental DirectML setup](#experimental-directml-setup) for dependencies, limitations, and a command example.

`--language`

- Optional.
- Language code used for transcription and HTML sentence segmentation.
- Default: `en`

`--audio-extension`

- Optional.
- Filename extension used for split audio chunks.
- Default: `.m4a`

`--audio-codec`

- Optional.
- Split audio codec mode.
- Supported values: `copy`, `aac`
- Default: `copy`
- `copy` preserves the source audio stream.
- `aac` re-encodes split audio and enables the quality controls below.
- With `--audio-codec aac`, the default bitrate is `64k` and the default sample rate is `24000` unless overridden.

`--audio-bitrate`

- Optional.
- AAC bitrate for split audio chunks, such as `64k`, `96k`, or `128k`.
- Default: `64k` when `--audio-codec aac` is used.
- Only valid when `--audio-codec aac` is used.

`--audio-sample-rate`

- Optional.
- AAC sample rate in Hz, such as `24000` or `44100`.
- Default: `24000` when `--audio-codec aac` is used.
- Only valid when `--audio-codec aac` is used.

`--audio-channels`

- Optional.
- AAC channel count, such as `1` for mono or `2` for stereo.
- Default: `1` (mono) when `--audio-codec aac` is used.
- Only valid when `--audio-codec aac` is used.

`--chunk-seconds`

- Optional.
- Fixed chunk length, in seconds, used only when the source audio has no chapter metadata.
- Default: `600`

`--split-jobs`

- Optional.
- Number of parallel `ffmpeg` jobs used during the split stage.
- Default with `--audio-codec copy`: `1`
- Default with `--audio-codec aac`: `max(1, cpu_count - 2)`
- The effective value is capped at the machine CPU count.
- Higher values may help AAC splitting, but can be slower on external drives or with `--audio-codec copy`.

`--batch-size`

- Optional.
- Number of audio windows decoded together per transcription batch.
- Default: `1`
- Higher values can increase transcription speed but raise peak memory (notably GPU/unified memory on the `mlx` backend).
- Ignored by `whispercpp`, which processes one audio chunk per invocation.

`--mlx-cache-gb`

- Optional.
- Cap, in GB, for the `mlx` Metal buffer cache during transcription, limiting how much freed GPU memory `mlx` retains across chunks.
- Default: unset (no cap).
- Ignored for non-`mlx` backends.

### Run behavior

- if compatible work already exists, the pipeline resumes automatically
- if the inputs or transcription-relevant options changed, derived artifacts (match, segmentation, SMIL, packaged and final EPUB) are rebuilt; split audio chunks and transcripts are kept and reused whenever their recorded configuration still matches
- if no work exists yet, the pipeline starts from the beginning
- if `--output-dir` is omitted, the final EPUB is written next to the source EPUB

## Outputs

Final output:

- `<book-stem>.media-overlay.epub`
- default location: the source EPUB folder

Working directory default:

- `<resolved-output-dir>/.<book-stem>.epubmo`

Important working artifacts:

- `state.json`
- `matched_list.json`
- `segmented.epub3`
- `validation.json`
- `logs/pipeline.log`

## Notes

- The source `.epub` is not modified directly.
- If the source audio has chapter metadata, the pipeline splits by chapter.
- If the source audio has no chapter metadata, the pipeline splits into fixed-size chunks.
- Split audio defaults to stream copy so packaged playback preserves source quality.
- Validation checks packaging, SMIL clip quality, transcript coverage, and OPF media-overlay wiring.
- The console output and `logs/pipeline.log` include detailed stage-by-stage progress and timings.

## Files

- `generate_epub_overlay.py`: CLI wrapper and state manager with automatic resume
- `pipeline_core.py`: EPUB/audio matching, SMIL generation, packaging, and validation logic
- `mark_sentence.py`: HTML segmentation logic used for overlay targets
- `epub_reference_index.py`: builds the set of referenced `class`/`id` names so tag cleanup only removes provably-unused markup
- `transcription_backend.py`: MLX, WhisperX, and whispercpp adapters, shared word alignment, and whispercpp runtime/model setup
- `requirements.txt`: Python dependencies for the pipeline and backends
- [whisper.cpp runtime repository](https://github.com/CryptoKey98/whispercpp-vulkan-runtime): runtime source packaging, build script, releases, and licenses

## Troubleshooting

If `ffmpeg` or `ffprobe` are missing, install them and ensure they are on `PATH`.

If the first run cannot download NLTK data automatically, make sure the machine has network access and write permission for `~/.cache/epub-media-overlay/nltk_data`.

If a run is interrupted, rerun the same command with the same work directory. Completed compatible chunk transcripts are reused; an unfinished chunk restarts from its beginning. With whispercpp, the progress bar advances when a chunk finishes, so a long chapter can remain at 0% while transcription and word alignment are running.

For whispercpp GPU detection errors, check your graphics driver and Vulkan support. The backend reports an error when GPU execution cannot be confirmed. For a missing runtime package, run `python -m pip install -r requirements.txt` in the same Python environment used to run the program. The pipeline log contains per-chunk GPU diagnostics.


## AMD GPU transcription with whisper.cpp (optional)

Select `--backend whispercpp --model small` for Vulkan GPU transcription followed
by WhisperX forced word alignment on CPU by default. Experimental DirectML alignment
is available for English on Windows (see below). The remaining matching, SMIL, packaging,
and validation stages use the existing pipeline. Default backend selection is
unchanged; this backend is opt-in. `--batch-size` does not affect whisper.cpp.

### Setup and storage

Install the normal Python requirements with `python -m pip install -r requirements.txt`.
On Windows x64, pip installs the `whispercpp-vulkan-runtime` wheel named in `requirements.txt` from the [latest standalone runtime release](https://github.com/CryptoKey98/whispercpp-vulkan-runtime/releases/latest). The URL currently requires the `1.8.5.1` wheel asset; update the filename when changing package versions.

- Runtime and DLLs: `.venv/Lib/site-packages/whispercpp_runtime/bin/`.
- Command launcher: `.venv/Scripts/whispercpp.exe`.
- GGML models: standard Hugging Face cache, normally `~/.cache/huggingface/hub/models--ggerganov--whisper.cpp/`.
- WhisperX CPU alignment models: downloaded when needed and reused from their existing cache.

Package revision `1.8.5.1` contains an MSVC/shared-library build of upstream commit `f24588a272ae8e23280d9c220536437164e6ed28` (reported version 1.8.5). This is a community build, not an executable published by OpenAI or ggml-org. Build provenance and third-party licenses are included in the installed package; build instructions live in the separate runtime repository.

No compiler, Vulkan SDK, or manual model download is required for end users. A Vulkan-capable graphics driver, FFmpeg, and the Microsoft Visual C++ v14 x64 Redistributable are required. Pip installation and initial model downloads need network access; installed runtimes and cached models work offline. Other platforms can supply their own compatible Vulkan CLI.

Storage overrides:

- `HF_HOME` / `HF_HUB_CACHE`: standard Hugging Face model cache controls.
- `WHISPERCPP_BINARY`: an explicit compatible Vulkan `whisper-cli` executable.
- `WHISPERCPP_MODEL_DIR`: a manually populated model directory.
- `--model`: an existing GGML `.bin` file or a supported Whisper model name.

`WHISPERCPP_CACHE_DIR` is no longer used. Upgrading from the extracted runtime changes the executable path, so existing transcript stamps may trigger retranscription. Keep a separate work directory if you want to preserve old results for comparison. The old runtime cache is not used or automatically deleted.

GGML models are downloaded from https://huggingface.co/ggerganov/whisper.cpp,
pinned at revision `5359861c739e955e79d9a303bcbc70fb988958b1`. Existing faster-whisper
model files cannot be used as GGML models. The adapter checks for a Vulkan GPU
before prepare and requires confirmation of GPU execution for each chunk; it
will report an error instead of silently accepting CPU fallback.

Example from the repository directory (PowerShell):

```powershell
.\.venv\Scripts\python.exe generate_epub_overlay.py `
  --audio "..\Audiobooks\Example.m4b" `
  --epub "..\EPUB\Example.epub" `
  --backend whispercpp --model small --language en `
  --work-dir "..\Overlay Work\Example-amd" `
  --output-dir "..\Media Overlays AMD"
```

Use a separate work/output folder when comparing backends to retain both sets of
results. Cache stamps distinguish the backend, model, and local executable/model
file identities. An identical rerun can reuse completed AMD transcripts. Replacing
the executable or model with a file of different size or modification time invalidates
those transcript stamps. Runtime DLL changes alone are not detected: use a new work
folder when replacing only companion DLLs.

### Behavior and validation

Audio chunks are converted to temporary 16 kHz mono PCM WAV files for whisper.cpp.
Its JSON millisecond offsets become second-based segments, then the existing
WhisperX alignment model supplies word timestamps. Alignment models are cached
across chunks; whisper.cpp currently starts a new process and loads its model for
each chunk. All temporary conversion files are cleaned up on success or failure.
GPU selection is shown once at startup. Per-chunk GPU diagnostics and timings go
only to the pipeline log, so the console progress bar updates in place.

Different transcription backends can produce different text and segment boundaries with the same model size.


### Word alignment selection

Use `--alignment-backend` to choose how word timestamps are produced independently
of speech recognition. Omitting it is equivalent to `--alignment-backend auto`.

| Option | Behavior |
| --- | --- |
| `auto` (default) | Preserves existing behavior: WhisperX uses CUDA when available, otherwise CPU; whisper.cpp uses CPU alignment; MLX uses its existing alignment. |
| `cpu` | Forces CPU alignment for WhisperX or whisper.cpp, without changing the transcription device. |
| `directml` | Experimental GPU alignment for English with whisper.cpp or WhisperX on Windows. |

MLX currently accepts only `auto`. DirectML itself is not language-specific; this
integration currently supports only the English alignment model. A compatible
DirectX 12 GPU and driver are required. DirectML supports AMD, NVIDIA, and Intel
GPUs; this project has not validated performance on every supported device.

### Experimental DirectML setup

On Windows x64, `requirements.txt` installs `onnxruntime-directml==1.23.0`
as the single ONNX Runtime distribution, plus `onnx==1.23.1` for model export.
DirectML alignment remains opt-in; the default is still `auto`.

The repository includes a dependency-only patch of faster-whisper, version
`1.2.1+directml1`, so its Windows dependency accepts the DirectML runtime.
Its inference code, models, and upstream licenses are unchanged. The DirectML
runtime supplies CPU execution for faster-whisper's ONNX voice detection;
WhisperX's normal CPU/CUDA alignment continues to use PyTorch.
See [package provenance and rebuild instructions](packages/README.md).
NVIDIA CUDA operation with this package combination has not been hardware-tested.

For a fresh environment, run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

For an existing environment that has standard `onnxruntime`, remove both runtime
distributions before reinstalling. They own overlapping files, so uninstalling
only one can leave the other incomplete. Stop running pipeline processes first.

```powershell
python -m pip uninstall -y onnxruntime onnxruntime-directml
python -m pip install -r requirements.txt
python -m pip check
```

Use this repository's requirements when updating dependencies. Independently
upgrading faster-whisper to an upstream version can bring standard `onnxruntime`
back into the environment. Other platforms retain the standard runtime dependency.

Run from the repository directory:

```powershell
python generate_epub_overlay.py `
  --audio "..\Audiobooks\Example.m4b" `
  --epub "..\EPUB\Example.epub" `
  --backend whispercpp `
  --alignment-backend directml `
  --model small `
  --language en `
  --output-dir "..\Media Overlays" `
  --work-dir "..\Overlay Work\Example-directml"
```

To use WhisperX transcription with the same DirectML alignment, replace
`--backend whispercpp` with `--backend whisperx`. WhisperX still selects CUDA
when available, otherwise CPU, for transcription; alignment uses DirectML with
CPU timestamp processing. Separate ASR caching described below applies only to
whisper.cpp, so changing alignment for WhisperX repeats transcription.

PowerShell displays `>>` as its continuation prompt; do not paste those characters.
Use a separate work/output directory when comparing results you want to retain.

On first use, the program exports and caches the English alignment model under
`~/.cache/epub-media-overlay/alignment`. DirectML accelerates acoustic model
inference; timestamp processing and unsupported operations still use CPU. Very
short segments that require a length mask also use CPU. An inference error logs
a warning and switches that alignment model to CPU for the remainder of the run.
Missing dependencies, model export errors, or session initialization errors stop
the run rather than silently accepting an unavailable DirectML backend.

### Alignment caching and resume

New whisper.cpp transcriptions save an `.asr.json` beside each audio chunk before
word alignment starts. This allows a different alignment selection, or a restart
after an alignment interruption, to reuse completed speech recognition when the
audio, language, model, and runtime identities still match.

The aligned transcript cache distinguishes DirectML from default CPU alignment.
Older transcripts without a separate ASR cache require one new transcription
when changing alignment. Work within an interrupted ASR chunk is repeated; an
interrupted alignment step also restarts rather than resuming at an individual word.
