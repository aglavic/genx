# Update the genx version in .iss file
import sys
import os

folder=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(folder))
from genx import version

txt=open(os.path.join(folder, 'genx_template.iss'), 'r').read()
txt=txt.replace('{version}', version.__version__)
open(os.path.join(folder, 'genx.iss'), 'w').write(txt)
