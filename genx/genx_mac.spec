# -*- mode: python ; coding: utf-8 -*-


block_cipher = None
import os
from PyInstaller.utils.hooks import collect_submodules
from glob import glob
genx_modules = [si[:-3].replace('/', '.') for si in glob('genx/plugins/*/*.py')]
genx_modules += [si[:-3].replace('/', '.') for si in glob('genx/plugins/*/help_modules/*.py')]
genx_modules += [si[:-3].replace('/', '.') for si in glob('genx/models/*.py')]

a = Analysis(['scripts/genx_mac'],
             pathex=[os.path.abspath(os.path.curdir)],
             binaries=[],
             datas=[('genx', 'genx_source/genx')],
             hiddenimports= genx_modules+[
                 'pymysql', 'numba', 'wx._core.ArtProvider'
                  ] + collect_submodules('h5py', 
                     filter=lambda name: not name.startswith('h5py.tests')),
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
# remove genx modules from distribution to make sure to import from genx_source folder on execution
a.pure = [pi for pi in a.pure if not pi[0].startswith('genx')]
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
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None , icon='mac_build/genx.icns')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=False,
               upx_exclude=[],
               name='genx')
app = BUNDLE(coll,
             name='genx.app',
             icon='mac_build/genx.icns',
             bundle_identifier=None,
             info_plist={
             'NSPrincipalClass': 'NSApplication',
             'NSAppleScriptEnabled': False,
             'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'GenX Model File',
                    'CFBundleTypeIconFile': 'mac_build/genx.icns',
                     'CFBundleTypeExtensions': ['hgx', 'gx'],
                    'LSHandlerRank': 'Owner'
                    }
                ]
            },
            )
