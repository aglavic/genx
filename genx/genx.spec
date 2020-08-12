# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
import os

for pi in os.environ['PATH'].split(';'):
    if os.path.exists(os.path.join(pi, 'libiomp5md.dll')):
        dll_path=os.path.join(pi, 'libiomp5md.dll')
        break

a = Analysis(['scripts\\genx'],
             pathex=[os.path.abspath(os.path.curdir)],
             binaries=[(dll_path, '.')],
             datas=[('genx', 'genx')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt5', ],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='genx',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=False , icon='windows_build\\genx.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               upx_exclude=[],
               name='genx')
