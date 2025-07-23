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
                       (os.path.join(dll_path, 'mkl_*.dll'), '.'), # For CUDA toolkit
                       (os.path.join(dll_path, 'nvvm64*.dll'), 'DLLs'), # For CUDA toolkit
                       (os.path.join(dll_path, 'libdevice*'), 'DLLs'),
                       (os.path.join(dll_path, 'nvvm64*.dll'), 'Library/bin'),
                       (os.path.join(dll_path, 'cudart64*.dll'), 'Library/bin'),
                       ],
             datas=[('genx', 'genx')],
             hiddenimports=['pymysql', 'numba', 'numba.cuda', 
							'scipy._lib.array_api_compat.numpy.fft', 'scipy.special._special_ufuncs',
			                'vtk', 'vtkmodules', 'vtkmodules.all','vtkmodules.util.colors',
							'vtkmodules.util','vtkmodules.util.numpy_support',
                            'scipy.special.cython_special', 'xml.dom.minidom',
                            'docutils.parsers.null', 'genx.plugins.add_ons.LayerGraphics',
                            'genx.gui.plotpanel', 'genx.gui.plotpanel_wx'],
             hooksconfig={
                 "matplotlib": {
                     "backends": "WxAgg",
                     },},
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
exe_console = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='genx_console',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=True , icon='windows_build\\genx.ico')
coll = COLLECT(exe,
               exe_console,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               upx_exclude=[],
               name='genx')
