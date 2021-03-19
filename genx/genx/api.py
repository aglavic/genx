"""
Scripting interface for GenX for use in python scripting or Jupyter Notebooks.
"""

__all__=[]

import sys
from genx import models as _models
if not "models" in sys.modules:
    sys.modules["models"]=_models

from genx.model import Model
from genx.diffev import DiffEv
from genx import filehandling as io
_config=io.Config()

_fit_output=[]
def text_output_api(text):
    _fit_output.append(text)

def load(fname, compile=True):
    model=Model()
    optimizer=DiffEv()
    io.load_file(fname, model, optimizer, _config)
    if compile:
        model.compile_script()
    optimizer.model=model
    optimizer.text_output=text_output_api
    return model, optimizer

def save(fname, model, optimizer):
    io.save_file(fname, model, optimizer, _config)

def fit_notebook(model, optimizer):
    """
    Function to fit a GenX model while giving feedback on a Jupyter notebook.
    """
    global _fit_output
    _fit_output=[]
    optimizer.start_fit(model)
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output
    from numpy import array
    from time import sleep

    fig=plt.figure(figsize=(14, 5))
    plt.suptitle("To stop fit, interrupt the Kernel")

    plt.subplot(121)
    ax1=plt.gca()
    line=plt.semilogy([0., 1.], [0.1, 1.])[0]
    plt.xlabel('Generation')
    plt.ylabel('FOM')
    t1=plt.title('FOM:')

    plt.subplot(122)
    t2=plt.title('Data display')
    ax2=plt.gca()
    refls=[]
    for i, ds in enumerate(model.data.items):
        refls.append(plt.semilogy(ds.x, ds.y,
                              color=ds.data_color,
                              lw=ds.data_linethickness, ls=ds.data_linetype,
                              marker=ds.data_symbol, ms=ds.data_symbolsize,
                              label='data-%i: %s'%(i, ds.name))[0])
        if ds.y_sim.shape==ds.y.shape:
            refls.append(plt.semilogy(ds.x, ds.y_sim,
                                  color=ds.sim_color,
                                  lw=ds.sim_linethickness, ls=ds.sim_linetype,
                                  marker=ds.sim_symbol, ms=ds.sim_symbolsize,
                                  label='model-%i: %s'%(i, ds.name))[0])

    plt.xlabel('x')
    plt.ylabel('I')

    plt.draw()

    last=2
    while optimizer.running:
        try:
            sleep(0.1)
            if len(optimizer.fom_log)<=last:
                continue
            x, y=array(optimizer.fom_log).T
            last=len(x)
            #t1.set_text('FOM: %.4e'%optimizer.best_fom)
            t1.set_text(_fit_output[-1])
            #vec=optimizer.best_vec
            #list(map(lambda func, value: func(value), model.get_fit_pars()[0], vec))
            #model.evaluate_sim_func()
            j=0
            for i, ds in enumerate(model.data.items):
                refls[j].set_ydata(ds.y)
                if ds.y_sim.shape==ds.y.shape:
                    j=+1
                    refls[j].set_ydata(ds.y_sim)
                j+=1

            line.set_xdata(x)
            line.set_ydata(y)

            ax1.set_xlim(0, x[-1])
            ax1.set_ylim(y.min()*0.9, y.max()*1.1)
            plt.draw()
            clear_output(wait=True)
            display(fig)
        except KeyboardInterrupt:
            optimizer.stop_fit()
    plt.close()

    print(_fit_output[-1])
    print("If you want to update the model with the fit results, call api.fit_update(model, optimizer)")

def fit_update(model, optimizer):
    pnames=model.parameters.get_fit_pars()[1]
    for di in model.parameters.data:
        if di[0] in pnames:
            di[1]=optimizer.best_vec[pnames.index(di[0])]
    model.simulate()
