"""Build a clearly versioned, dependency-only faster-whisper wheel patch.
Upstream code and license are preserved byte-for-byte; RECORD is regenerated.
"""
from pathlib import Path
import zipfile,hashlib,base64,csv,io,json,tempfile,subprocess,sys
root=Path(__file__).resolve().parent
download=tempfile.TemporaryDirectory()
subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:', '--index-url', 'https://pypi.org/simple', 'faster-whisper==1.2.1', '-d', download.name], check=True)
src=Path(download.name)/'faster_whisper-1.2.1-py3-none-any.whl'
if hashlib.sha256(src.read_bytes()).hexdigest() != "79a66ad50688c0b794dd501dc340a736992a6342f7f95e5811be60b5224a26a7":
 raise RuntimeError('Upstream wheel checksum mismatch')
dst=root/'faster_whisper-1.2.1+directml1-py3-none-any.whl'
old='faster_whisper-1.2.1.dist-info/'
new='faster_whisper-1.2.1+directml1.dist-info/'
records=[]
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED) as zout:
 for name in zin.namelist():
  if name.endswith('/RECORD'):continue
  data=zin.read(name)
  target=name.replace(old,new)
  if name==old+'METADATA':
   metadata=data.decode()
   assert 'Requires-Dist: onnxruntime<2,>=1.14' in metadata
   metadata=metadata.replace('Version: 1.2.1\n','Version: 1.2.1+directml1\n',1)
   metadata=metadata.replace('Requires-Dist: onnxruntime<2,>=1.14','Requires-Dist: onnxruntime-directml==1.23.0; sys_platform == "win32"\nRequires-Dist: onnxruntime<2,>=1.14; sys_platform != "win32"',1)
   data=metadata.encode()
  info=zipfile.ZipInfo(target, (2025,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
  zout.writestr(info,data)
  records.append([target,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('='),str(len(data))])
 records.append([new+'RECORD','',''])
 output=io.StringIO(newline='');csv.writer(output).writerows(records)
 info=zipfile.ZipInfo(new+'RECORD', (2025,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
 zout.writestr(info,output.getvalue())
print(dst)
