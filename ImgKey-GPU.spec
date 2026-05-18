# -*- mode: python ; coding: utf-8 -*-

# GPU runtime packaging flavor: PyTorch CUDA runtime and GPU probe support,
# but no BiRefNet/Transformers model stack and no model weights.

from PyInstaller.utils.hooks import collect_dynamic_libs


MODEL_STACK_EXCLUDES = [
    'transformers',
    'timm',
    'kornia',
    'einops',
    'accelerate',
    'huggingface_hub',
    'safetensors',
    'skimage',
    'onnxruntime',
    'onnxruntime_gpu',
    'pymatting',
    'corridorkey',
    'CorridorKey',
]


def cuda_binaries():
    binaries = []
    for package in ('torch', 'torchvision', 'nvidia'):
        try:
            binaries += collect_dynamic_libs(package)
        except Exception as exc:
            print(f'ImgKey-GPU.spec: could not collect dynamic libs for {package}: {exc}')
    return binaries


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=cuda_binaries(),
    datas=[],
    hiddenimports=[
        'gpu_runtime',
        'torch',
        'torch.cuda',
        'torchvision',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py'],
    excludes=MODEL_STACK_EXCLUDES,
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
