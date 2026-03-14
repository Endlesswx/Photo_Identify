# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

project_root = Path('.').resolve()
src_root = project_root / 'src'

hiddenimports = [
    'jieba',
    'pillow_heif',
    'sentence_transformers',
    'insightface',
    'onnxruntime',
    'sklearn.cluster',
]

datas = []
datas += collect_data_files('photo_identify')
datas += collect_data_files('jieba')
datas += collect_data_files('pillow_heif')
datas += [
    ('src/video_edit/video_compression.py', 'src/video_edit'),
    ('src/data_migration/lvip_decompression.py', 'src/data_migration'),
]

block_cipher = None


a = Analysis(
    ['src/photo_identify/__main__.py'],
    pathex=[str(src_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='photo_identify',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='photo_identify',
)
