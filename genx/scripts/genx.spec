# -*- mode: python -*-
a = Analysis([os.path.join(HOMEPATH,'support/_mountzlib.py'),
              os.path.join(HOMEPATH,'support/useUnicode.py'), 
              'genx',
              ],
             pathex=['/Users/Matts/workspace/GenX/trunk/genx/scripts',
                     '/Users/Matts/workspace/GenX/trunk/genx'])
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          name=os.path.join('build/pyi.darwin/genx', 'genx'),
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT( exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name=os.path.join('dist', 'genx'))
