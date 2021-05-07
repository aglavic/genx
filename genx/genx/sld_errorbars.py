import argparse
import re
import os
import time
import sys

import numpy as np

from . import version
from .gui_logging import iprint

def sld_mc(args):
    """ Function to start fitting from the command line.

    :param args:
    :return:
    """
    from . import model
    from . import diffev
    from . import filehandling as io

    mod=model.Model()
    config=io.Config()
    config.load_default(os.path.split(os.path.abspath(__file__))[0]+'genx.conf')
    opt=diffev.DiffEv()

    iprint('Loading model %s...'%args.infile)
    io.load_file(args.infile, mod, opt, config)
    io.load_opt_config(opt, config)
    # Hack the script so that it simulates the SLD instead of the data
    mod.script=re.sub(r"SimSpecular\((.*)\,(.*)\)", r"SimSLD(data[0].x, None,\2)", mod.script)
    iprint("Hacking the model script. Resulting script:")
    iprint(mod.script)

    # Simulate, this will also compile the model script
    iprint('Compiling model...')
    mod.compile_script()

    (funcs, vals, minvals, maxvals)=mod.get_fit_pars()
    vals=np.array(vals)
    boundaries=[row[5] for row in mod.parameters.data if not row[0]=='' and row[2]]
    iprint(boundaries)
    boundaries=np.array([eval(s) for s in boundaries])
    minvals, maxvals=boundaries[:, 0]+vals, boundaries[:, 1]+vals
    min_SLD=[]
    max_SLD=[]
    z=np.arange(args.min, args.max, args.step)
    # Change the x-data so that it contain the z values instead.
    for d in mod.data:
        d.x=z

    def extreme_dict(cur, extreme, func):
        """Makes a comparison of cur and extreme through func (np.min or np.max) and returns the result as a dict"""
        return_dic={}
        for key in extreme:
            if key not in ['z', 'SLD unit']:
                # print cur[key].shape
                return_dic[key]=func(np.c_[cur[key], extreme[key]], axis=1)
            else:
                return_dic[key]=cur[key]
        return return_dic

    iprint("Calculating sld's...")
    missed=0
    for i in range(args.runs):
        current_vals=minvals+(maxvals-minvals)*np.random.random_sample(len(funcs))
        [func(val) for func, val in zip(funcs, current_vals)]

        mod.script_module._sim=False
        current_sld=mod.script_module.Sim(mod.data)
        same_shape=all([sld['z'].shape==msld['z'].shape for sld, msld in zip(current_sld, min_SLD)])
        if i==0:
            min_SLD=[sld for sld in current_sld]
            max_SLD=[sld for sld in current_sld]
        elif same_shape:
            min_SLD=[extreme_dict(sld, msld, np.min) for sld, msld in zip(current_sld, min_SLD)]
            max_SLD=[extreme_dict(sld, msld, np.max) for sld, msld in zip(current_sld, max_SLD)]
        else:
            missed+=1

        sys.stdout.write("\r Progress: %d%%"%(float(i)*100/float(args.runs)))
        sys.stdout.flush()

    iprint(' ')
    iprint(missed, " simulations was discarded due to wrong size.")
    iprint("Saving the data to file...")
    for sim in range(len(min_SLD)):
        new_filename=(args.outfile+'%03d'%sim+'.dat')
        save_array=np.array([min_SLD[sim]['z']])
        header='z\t'
        for key in min_SLD[sim]:
            if key!='z' and key!='SLD unit':
                save_array=np.r_[save_array, [min_SLD[sim][key].real], [min_SLD[sim][key].imag],
                                 [max_SLD[sim][key].real], [max_SLD[sim][key].imag]]
                header+='min(%s.real)\tmin(%s.imag)\tmax(%s.real)\tmax(%s.imag)\t'%(key, key, key, key)
        f=open(new_filename, 'w')
        f.write(
            "# Monte Carlo estimation of SLD bounds with script sld_errorbars.py model taken from file: %s\n"%args.infile)
        f.write("# File created: %s\n"%time.ctime())
        f.write("# Simulated SLD for data set: %s\n"%mod.data[sim].name)
        f.write("# Headers: \n")
        f.write('#'+header+'\n')
        np.savetxt(f, save_array.transpose(), delimiter='\t')
        f.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description="sld_errorbars %s, creates a boundary sld profile from given errorbar"
                                               " values in a GenX file. Note this is experimental software!"%version.__version__,
                                   epilog="For support, manuals and bug reporting see http://genx.sf.net"
                                   )

    parser.add_argument('--runs', type=int, default=1, help='Number of Monte Carlo evaluations')
    parser.add_argument('--min', type=float, default=0.0, help='Minimum position to evaluate SLD')
    parser.add_argument('--max', type=float, default=1000.0, help='Maximum position to evaluate SLD')
    parser.add_argument('--step', type=float, default=0.5, help='Step size for the gridding of the SLD')

    parser.add_argument('infile', default='', help='The .gx or .hgx file to load')
    parser.add_argument('outfile', default='out', help='The output base file name (no extension)')

    args=parser.parse_args()
    sld_mc(args)
