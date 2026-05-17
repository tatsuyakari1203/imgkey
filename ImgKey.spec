# -*- mode: python ; coding: utf-8 -*-

# Packaging source of truth for the default ImgKey desktop build.
# Keep this bundle non-AI: optional adapters in ai_assist.py must not pull
# PyTorch/CUDA/model runtimes into the onefile EXE.

AI_RUNTIME_EXCLUDES = [
    'torch',
    'torchvision',
    'torchaudio',
    'triton',
    'nvidia',
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


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=AI_RUNTIME_EXCLUDES,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ImgKey',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
