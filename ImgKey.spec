# -*- mode: python ; coding: utf-8 -*-

# Packaging source of truth for the default ImgKey desktop build.
# Keep this bundle lightweight: no CUDA tensor runtime in the default EXE.

DEFAULT_RUNTIME_EXCLUDES = [
    'torch',
    'torch' + 'vision',
    'torch' + 'audio',
    'torch' + 'text',
    'triton',
    'nvidia',
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


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=DEFAULT_RUNTIME_EXCLUDES,
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
