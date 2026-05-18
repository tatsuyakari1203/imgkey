# -*- mode: python ; coding: utf-8 -*-

# GPU BiRefNet packaging flavor: PyTorch CUDA + BiRefNet-only worker path.
# Model weights are not pulled from the network. A local snapshot can be used at
# runtime through IMGKEY_BIREFNET_MODEL. To bundle a snapshot, set
# IMGKEY_BIREFNET_BUNDLE_MODEL and IMGKEY_BIREFNET_BUNDLE_LICENSE_OK=1; the
# manifest must contain SHA256 hashes for all required files or the build stops.

import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


FORBIDDEN_MODEL_EXCLUDES = [
    'onnxruntime',
    'onnxruntime_gpu',
    'pymatting',
    'corridorkey',
    'CorridorKey',
    'segment_anything',
    'sam2',
    'u2net',
    'modnet',
    'vitmatte',
]


def cuda_binaries():
    binaries = []
    for package in ('torch', 'torchvision', 'nvidia'):
        try:
            binaries += collect_dynamic_libs(package)
        except Exception as exc:
            print(f'ImgKey-GPU-BiRefNet.spec: could not collect dynamic libs for {package}: {exc}')
    return binaries


def runtime_datas():
    datas = [('ai_backends/birefnet_manifest.json', 'ai_backends')]
    for package in ('transformers', 'huggingface_hub'):
        try:
            datas += collect_data_files(package)
        except Exception as exc:
            print(f'ImgKey-GPU-BiRefNet.spec: could not collect data files for {package}: {exc}')
    model_dir = os.environ.get('IMGKEY_BIREFNET_BUNDLE_MODEL', '').strip()
    if not model_dir:
        print('ImgKey-GPU-BiRefNet.spec: no bundled model; set IMGKEY_BIREFNET_MODEL at runtime or provide IMGKEY_BIREFNET_BUNDLE_MODEL for a gated bundle.')
        return datas
    if os.environ.get('IMGKEY_BIREFNET_BUNDLE_LICENSE_OK') != '1':
        raise RuntimeError('Refusing to bundle BiRefNet model without IMGKEY_BIREFNET_BUNDLE_LICENSE_OK=1 after license/notice review.')
    from ai_backends.birefnet_adapter import validate_model_path

    validation = validate_model_path(model_dir, verify_hashes=True, require_hashes=True)
    datas.append((validation['model_path'], 'models/BiRefNet'))
    print(f"ImgKey-GPU-BiRefNet.spec: bundled BiRefNet snapshot {validation['model_path']}")
    return datas


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=cuda_binaries(),
    datas=runtime_datas(),
    hiddenimports=[
        'gpu_runtime',
        'ai_worker',
        'ai_backends',
        'ai_backends.birefnet_adapter',
        'torch',
        'torch.cuda',
        'torchvision',
        'transformers',
        'transformers.models.auto',
        'safetensors',
        'huggingface_hub',
        'timm',
        'einops',
        'kornia',
        'accelerate',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py'],
    excludes=FORBIDDEN_MODEL_EXCLUDES,
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
    text_default='Extracting ImgKey GPU BiRefNet bundle…',
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
    name='ImgKey-GPU-BiRefNet',
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
