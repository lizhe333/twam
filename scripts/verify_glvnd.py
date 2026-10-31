"""Check the prepared x86-64 GLVND dispatch layout without initializing a GPU."""
import argparse
import ctypes
import json
from pathlib import Path


def dispatch_slot(pointer):
    # The pinned x86-64 GLVND entry stubs jump through the TLS dispatch table.
    code = ctypes.string_at(pointer, 32)
    start = code.find(b'\x41\xff\xa3')
    if start < 0:
        raise RuntimeError('Unrecognized x86-64 GLVND dispatch stub')
    return int.from_bytes(code[start + 3:start + 7], 'little')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library-dir', type=Path, required=True)
    args = parser.parse_args()
    names = ('libGL.so.1', 'libOpenGL.so.0', 'libGLESv1_CM.so.1', 'libGLESv2.so.2',
             'libGLdispatch.so.0', 'libGLX.so.0', 'libEGL.so.1')
    libs = {name: ctypes.CDLL(name) for name in names}
    get_proc = libs['libGLdispatch.so.0'].__glDispatchGetProcAddress
    get_proc.argtypes, get_proc.restype = [ctypes.c_char_p], ctypes.c_void_p
    for symbol in ('eglGetProcAddress', 'eglGetDisplay', 'eglInitialize', 'eglMakeCurrent',
                   'eglChooseConfig', 'eglBindAPI', 'eglCreateContext', 'eglDestroyContext', 'eglTerminate'):
        getattr(libs['libEGL.so.1'], symbol)
    layouts = {}
    for symbol in ('glGetString', 'glGetError', 'glClear', 'glViewport'):
        expected = dispatch_slot(get_proc(symbol.encode()))
        actual = {name: dispatch_slot(ctypes.cast(getattr(libs[name], symbol), ctypes.c_void_p).value)
                  for name in names[:4]}
        if any(offset != expected for offset in actual.values()):
            raise RuntimeError(f'Mixed GLVND dispatch table for {symbol}: expected {expected:#x}, got {actual}')
        layouts[symbol] = hex(expected)
    mapped = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
              if '/' in line and Path(line.split()[-1]).name in names}
    expected_paths = {(args.library_dir / name).resolve() for name in names}
    if mapped != expected_paths:
        raise RuntimeError(f'GLVND libraries loaded outside prepared directory: {sorted(map(str, mapped))}')
    print(json.dumps(dict(glvnd_dispatch='matched', slots=layouts, library_dir=str(args.library_dir),
                          gpu_initialized=False)), flush=True)


if __name__ == '__main__':
    main()
