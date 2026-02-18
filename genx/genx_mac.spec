# -*- mode: python ; coding: utf-8 -*-


block_cipher = None
import os
from PyInstaller.utils.hooks import collect_submodules
from glob import glob
genx_modules = [si[:-3].replace('/', '.') for si in glob('genx/plugins/*/*.py')]
genx_modules += [si[:-3].replace('/', '.') for si in glob('genx/plugins/*/help_modules/*.py')]
genx_modules += [si[:-3].replace('/', '.') for si in glob('genx/models/*.py')]
genx_modules += ['genx.gui.plotpanel', 'genx.gui.plotpanel_wx']

a = Analysis(['scripts/genx_mac'],
             pathex=[os.path.abspath(os.path.curdir)],
             binaries=[("/usr/local/Cellar/libomp/21.1.8/lib/libomp.dylib", ".")],
             datas=[('genx', 'genx_source/genx'),
                    ('mac_build/genx.icns', '.'),
                    ('mac_build/orso.icns', '.')],
             hiddenimports= genx_modules+[
                 'numpy', 'pymysql', 'numba', 'requests',
                 'scipy._lib.array_api_compat.numpy.fft', 'scipy.special._special_ufuncs',
                 'scipy.special.cython_special', 'xml.dom.minidom',
                 'vtk', 'vtkmodules', 'vtkmodules.all', 'vtkmodules.util.colors',
                 'vtkmodules.util', 'vtkmodules.util.numpy_support',
                 'docutils.parsers.null', 'genx.plugins.add_ons.LayerGraphics',
                 'genx.gui.plotpanel', 'genx.gui.plotpanel_wx',
                 ] + collect_submodules('h5py',
                     filter=lambda name: not name.startswith('h5py.tests')
                  ) + collect_submodules('docutils.parsers'),
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
             bundle_identifier='com.isa.genx',
             info_plist={
             'NSPrincipalClass': 'NSApplication',
             'NSAppleScriptEnabled': False,
             'CFBundlePackageType': 'APPL',
             'LSApplicationCategoryType': 'public.app-category.education',
             'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'GenX Model File',
                    'CFBundleTypeIconFile': 'genx.icns',
                     'CFBundleTypeExtensions': ['hgx', 'gx'],
                    'LSHandlerRank': 'Owner',
                    'CFBundleTypeRole' : 'Editor',
                    'CFBundleIdentifier': 'com.isa.genx',
                },
                {
                    'CFBundleTypeName': 'ORSO reflectivity for GenX',
                    'CFBundleTypeIconFile': 'orso.icns',
                     'CFBundleTypeExtensions': ['ort', 'orb'],
                    'LSHandlerRank': 'Default',
                    'CFBundleTypeRole' : 'Viewer',
                    'CFBundleIdentifier': 'com.isa.genx',
                }
                ]
            },
            )
