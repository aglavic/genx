# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
import os

for pathi in os.environ['PATH'].split(';'):
    if os.path.exists(os.path.join(pathi, 'libiomp5md.dll')):
        dll_path=pathi
        break

a = Analysis(['scripts\\genx'],
             pathex=[os.path.abspath(os.path.curdir)],
             binaries=[(os.path.join(dll_path, 'libiomp5md.dll'), '.'),
                       (os.path.join(dll_path, 'nvvm64*.dll'), 'DLLs'), # For CUDA toolkit
                       (os.path.join(dll_path, 'libdevice*'), 'DLLs'),
                       (os.path.join(dll_path, 'nvvm64*.dll'), 'Library/bin'),
                       ],
             datas=[('genx', 'genx')],
             hiddenimports=['pymysql', 'numba', 'numba.cuda', 
			                'vtk', 'vtkmodules', 'vtkmodules.all','vtkmodules.util.colors',
							'vtkmodules.util','vtkmodules.util.numpy_support'],
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
