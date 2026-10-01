# Windows faster-whisper dependency patch

This directory bundles `faster-whisper 1.2.1+directml1`, a dependency-only repack
of the official 1.2.1 wheel. Python code, models and upstream license files are
unchanged. This is a project-maintained package, not an official upstream release.

On Windows its metadata requires `onnxruntime-directml==1.23.0` instead of
`onnxruntime`. On other platforms the original ONNX Runtime requirement remains.
The local version suffix makes the patch identifiable. All other dependencies
remain unchanged, and the wheel RECORD hashes are regenerated.

Rebuild from the repository root with:

```powershell
python packages/build_faster_whisper.py
```

The builder downloads only the official PyPI 1.2.1 wheel, verifies its SHA-256,
and uses fixed ZIP timestamps. No compiler is required. Requirements refer to this
bundled wheel on Windows x64, so no separate repository or unpublished release URL
is required. Run pip from the repository root. Do not upgrade faster-whisper
independently: an upstream wheel may reinstall the overlapping standard runtime.

Upstream: https://github.com/SYSTRAN/faster-whisper
Original wheel SHA-256: `79a66ad50688c0b794dd501dc340a736992a6342f7f95e5811be60b5224a26a7`

NVIDIA CUDA execution has not been tested on hardware for this patched setup.
