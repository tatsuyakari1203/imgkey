# -*- mode: python ; coding: utf-8 -*-

# GPU runtime packaging flavor: compact native CUDA DLL backend.

import glob
import os
from pathlib import Path
import subprocess
import struct
import sys


GPU_RUNTIME_EXCLUDES = [
    'torch',
    'torch' + 'vision',
    'torch' + 'audio',
    'torch' + 'text',
    'triton',
    'nvidia',
    'cupy',
    'pycuda',
    'pyopencl',
    'trans' + 'formers',
    't' + 'imm',
    'kor' + 'nia',
    'ei' + 'nops',
    'accel' + 'erate',
    'hugging' + 'face_hub',
    'safe' + 'tensors',
    'ski' + 'mage',
    'diff' + 'users',
    'peft',
    'token' + 'izers',
    'sentence' + 'piece',
    'tensorflow',
    'keras',
    'jax',
    'jaxlib',
    'flax',
    'ultra' + 'lytics',
    'onnx',
    'onnxruntime',
    'onnxruntime_gpu',
    'pymatting',
    'scipy',
    'numba',
    'corridor' + 'key',
    'Corridor' + 'Key',
]


CUDA_DLL_NAME = 'imgkey_cuda.dll'
MSVC_RUNTIME_DLLS = ('MSVCP140.dll', 'VCRUNTIME140.dll', 'VCRUNTIME140_1.dll')


def _repo_root():
    return Path(SPECPATH).resolve()


def _resolve_cuda_dll():
    candidates = []
    env_path = os.environ.get('IMGKEY_CUDA_DLL')
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(_repo_root() / 'native' / 'imgkey_cuda' / 'build' / CUDA_DLL_NAME)

    checked = []
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        checked.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError(
        f'{CUDA_DLL_NAME} was not found. Run native/imgkey_cuda/build.ps1 first '
        f'or set IMGKEY_CUDA_DLL. Checked: {checked}'
    )


def _vs_install_roots():
    roots = []
    for value in (os.environ.get('VSINSTALLDIR'), os.environ.get('VCINSTALLDIR')):
        if value:
            roots.append(Path(value))

    vswhere = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / 'Microsoft Visual Studio' / 'Installer' / 'vswhere.exe'
    if vswhere.is_file():
        try:
            output = subprocess.check_output(
                [
                    str(vswhere),
                    '-latest',
                    '-products',
                    '*',
                    '-requires',
                    'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                    '-property',
                    'installationPath',
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if output:
                roots.append(Path(output))
        except Exception:
            pass

    roots.extend(
        [
            Path(r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'),
            Path(r'C:\Program Files\Microsoft Visual Studio\2022\BuildTools'),
            Path(r'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools'),
        ]
    )

    unique = []
    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        key = str(resolved).casefold()
        if key not in seen and resolved.exists():
            seen.add(key)
            unique.append(resolved)
    return unique


def _msvc_runtime_search_dirs():
    dirs = [Path(sys.executable).resolve().parent, Path(sys.base_prefix).resolve()]
    for root in _vs_install_roots():
        patterns = [
            root / 'VC' / 'Redist' / 'MSVC' / '*' / 'x64' / 'Microsoft.VC*.CRT',
            root / 'VC' / 'Tools' / 'MSVC' / '*' / 'bin' / 'Hostx64' / 'x64',
            root / 'Common7' / 'IDE' / 'VC' / 'VCPackages',
            root / 'Common7' / 'IDE',
        ]
        for pattern in patterns:
            dirs.extend(Path(path) for path in glob.glob(str(pattern)))
    for raw in os.environ.get('PATH', '').split(os.pathsep):
        if raw:
            dirs.append(Path(raw))

    unique = []
    seen = set()
    for directory in dirs:
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        key = str(resolved).casefold()
        if key not in seen and resolved.is_dir():
            seen.add(key)
            unique.append(resolved)
    return unique


def _resolve_runtime_dll(name):
    for directory in _msvc_runtime_search_dirs():
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'{name} was not found; install the Visual C++ runtime or Visual Studio Build Tools.')


def _split_env_paths(value):
    if not value:
        return []
    parts = []
    for raw in value.split(os.pathsep):
        raw = raw.strip().strip('"')
        if raw:
            parts.append(Path(raw))
    return parts


def _read_c_string(blob, offset):
    end = blob.find(b'\0', offset)
    if end < 0:
        end = len(blob)
    return blob[offset:end].decode('ascii', errors='ignore')


def _pe_imported_dll_names(path):
    data = Path(path).read_bytes()
    if len(data) < 0x40 or data[:2] != b'MZ':
        return set()
    pe_offset = struct.unpack_from('<I', data, 0x3C)[0]
    if pe_offset + 24 >= len(data) or data[pe_offset:pe_offset + 4] != b'PE\0\0':
        return set()

    file_header_offset = pe_offset + 4
    section_count = struct.unpack_from('<H', data, file_header_offset + 2)[0]
    optional_size = struct.unpack_from('<H', data, file_header_offset + 16)[0]
    optional_offset = file_header_offset + 20
    magic = struct.unpack_from('<H', data, optional_offset)[0]
    data_directory_offset = optional_offset + (112 if magic == 0x20B else 96)
    section_offset = optional_offset + optional_size

    sections = []
    for index in range(section_count):
        offset = section_offset + index * 40
        if offset + 40 > len(data):
            break
        virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from('<IIII', data, offset + 8)
        sections.append((virtual_address, max(virtual_size, raw_size), raw_pointer, raw_size))

    def rva_to_offset(rva):
        for virtual_address, virtual_size, raw_pointer, raw_size in sections:
            if virtual_address <= rva < virtual_address + virtual_size:
                delta = rva - virtual_address
                if delta < raw_size:
                    return raw_pointer + delta
        return None

    def directory(index):
        offset = data_directory_offset + index * 8
        if offset + 8 > len(data):
            return 0, 0
        return struct.unpack_from('<II', data, offset)

    names = set()

    import_rva, _ = directory(1)
    import_offset = rva_to_offset(import_rva) if import_rva else None
    if import_offset is not None:
        while import_offset + 20 <= len(data):
            descriptor = struct.unpack_from('<IIIII', data, import_offset)
            if descriptor == (0, 0, 0, 0, 0):
                break
            name_offset = rva_to_offset(descriptor[3])
            if name_offset is not None:
                names.add(_read_c_string(data, name_offset))
            import_offset += 20

    delay_rva, _ = directory(13)
    delay_offset = rva_to_offset(delay_rva) if delay_rva else None
    if delay_offset is not None:
        while delay_offset + 32 <= len(data):
            descriptor = struct.unpack_from('<IIIIIIII', data, delay_offset)
            if descriptor == (0, 0, 0, 0, 0, 0, 0, 0):
                break
            name_offset = rva_to_offset(descriptor[1])
            if name_offset is not None:
                names.add(_read_c_string(data, name_offset))
            delay_offset += 32

    return {name for name in names if name}


def _find_cuda_runtime_candidate(name, cuda_dll):
    candidates = list(_split_env_paths(os.environ.get('IMGKEY_CUDA_RUNTIME_DLLS')))
    candidates.extend(cuda_dll.parent.glob(name))
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        candidates.append(Path(cuda_path) / 'bin' / name)
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path.is_file() and path.name.casefold() == name.casefold():
            return path
    raise FileNotFoundError(
        f'{name} is imported by {cuda_dll.name} but was not found beside the DLL, '
        f'in IMGKEY_CUDA_RUNTIME_DLLS, or under CUDA_PATH\\bin.'
    )


def _explicit_cuda_runtime_binaries(cuda_dll):
    # The normal build uses -cudart static, so cudart64_*.dll should not be in
    # the import table. Bundle a dynamic CUDA runtime only when the native DLL
    # imports it; this prevents stale adjacent toolkit DLLs from being shipped.
    imported = {name.casefold(): name for name in _pe_imported_dll_names(cuda_dll)}
    needed = [original for lowered, original in imported.items() if lowered.startswith('cudart64_') and lowered.endswith('.dll')]
    return [(str(_find_cuda_runtime_candidate(name, cuda_dll)), '.') for name in sorted(needed)]


def cuda_binaries():
    cuda_dll = _resolve_cuda_dll()
    binaries = [(str(cuda_dll), '.')]
    binaries.extend((str(_resolve_runtime_dll(name)), '.') for name in MSVC_RUNTIME_DLLS)
    binaries.extend(_explicit_cuda_runtime_binaries(cuda_dll))
    print('ImgKey-GPU.spec: bundled native GPU binaries:')
    for source, dest in binaries:
        print(f'  {source} -> {dest}')
    return binaries


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=cuda_binaries(),
    datas=[],
    hiddenimports=[
        'gpu_runtime',
        'gpu_accel',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py'],
    excludes=GPU_RUNTIME_EXCLUDES,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
splash = Splash(
    'packaging/imgkey_splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(36, 268),
    text_size=12,
    text_color='#E7ECF3',
    text_default='Extracting ImgKey GPU bundle…',
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    a.binaries,
    a.datas,
    splash.binaries,
    [],
    name='ImgKey-GPU',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
