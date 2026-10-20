"""Convert a RIFE v4.6 checkpoint (PyTorch zip, e.g. flownet_v4.6.pkl) into the extension's weight file.

The checkpoint is read with a restricted unpickler that only rebuilds tensors (no arbitrary code runs),
so torch is not needed: python3 export_weights.py flownet_v4.6.pkl ../extension/models/rife-v4.6.bin

Output format (little endian):
  magic "NNFW", u32 version (1), u32 header length, header JSON, padding to 4 bytes, fp16 data.
  header = {"model": ..., "tensors": [{"name", "shape", "offset"}]}, offset = element index into the data.
"""
import json
import pickle
import struct
import sys
import zipfile
from collections import OrderedDict

import numpy as np

DTYPES = {'FloatStorage': np.float32, 'HalfStorage': np.float16, 'BFloat16Storage': None}


class _Storage:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor_v2(storage, offset, size, stride, *_):
    arr = storage
    if not size:
        return arr[offset:offset + 1].reshape(())
    item = arr.itemsize
    return np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size), strides=tuple(s * item for s in stride)).copy()


def load_state_dict(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    prefix = names[0].split('/')[0]
    pkl = z.read(f'{prefix}/data.pkl')

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'collections' and name == 'OrderedDict':
                return OrderedDict
            if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                return _rebuild_tensor_v2
            if module == 'torch' and name in DTYPES and DTYPES[name] is not None:
                return _Storage(name)
            raise pickle.UnpicklingError(f'refusing to load {module}.{name}')

        def persistent_load(self, pid):
            kind, storage, key, _location, _numel = pid
            assert kind == 'storage'
            data = z.read(f'{prefix}/data/{key}')
            return np.frombuffer(data, dtype=DTYPES[storage.name])

    sd = Restricted(__import__('io').BytesIO(pkl)).load()
    if 'state_dict' in sd:
        sd = sd['state_dict']
    return OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())


def main():
    src, dst = sys.argv[1], sys.argv[2]
    sd = load_state_dict(src)
    tensors = []
    chunks = []
    off = 0
    for name, arr in sd.items():
        a = np.asarray(arr, dtype=np.float32)
        tensors.append({'name': name, 'shape': list(a.shape), 'offset': off})
        chunks.append(a.astype(np.float16).ravel())
        off += a.size
    header = json.dumps({'model': 'rife-v4.6', 'tensors': tensors}, separators=(',', ':')).encode()
    header += b' ' * (-(12 + len(header)) % 4)
    with open(dst, 'wb') as f:
        f.write(b'NNFW' + struct.pack('<II', 1, len(header)) + header)
        f.write(np.concatenate(chunks).astype('<f2').tobytes())
    print(f'{len(tensors)} tensors, {off} parameters -> {dst}')


if __name__ == '__main__':
    main()
