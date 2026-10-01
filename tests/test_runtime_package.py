"""Check the bundled dependency patch's metadata and integrity."""
from pathlib import Path
import base64
import csv
import hashlib
import io
import zipfile
from email.parser import Parser
from packaging.requirements import Requirement


def test_dependency_patch_metadata_and_records():
    wheel = Path(__file__).resolve().parents[1] / "packages/faster_whisper-1.2.1+directml1-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        assert metadata["Version"] == "1.2.1+directml1"
        requirements = [Requirement(r) for r in metadata.get_all("Requires-Dist")]
        for platform, expected in [("win32", "onnxruntime-directml"), ("linux", "onnxruntime")]:
            active = [r.name for r in requirements if r.name.startswith("onnxruntime") and (not r.marker or r.marker.evaluate({"sys_platform": platform}))]
            assert active == [expected]
        record_name = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
        for name, digest, size in csv.reader(io.StringIO(archive.read(record_name).decode())):
            if not digest:
                assert name == record_name
                continue
            contents = archive.read(name)
            assert int(size) == len(contents)
            assert digest == "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).decode().rstrip("=")
        assert any("LICENSE" in n.upper() for n in archive.namelist())
