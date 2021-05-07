"""
Scripting interface for GenX for use in python scripting or Jupyter Notebooks.
"""

__all__=[]

import os
import sys

# workaround for issues with ctrl+c on windows
if sys.platform=='win32':
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

from genx.model import Model
from genx.diffev import DiffEv
from genx import filehandling as io
from genx.plugins.add_ons.help_modules.reflectivity_utils import avail_models, SampleHandler, SampleBuilder
from genx.plugins.utils import PluginHandler

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
    return model, optimizer

def save(fname, model, optimizer):
    io.save_file(fname, model, optimizer, _config)

def fit_notebook(model, optimizer):
    """
    Function to fit a GenX model while giving feedback on a Jupyter notebook.
    """
    global _fit_output
    _fit_output=[]
    optimizer.text_output=text_output_api
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

def fit(model, optimizer):
    """
    Function to fit a GenX model while giving feedback with matplotlib graphs.
    """
    optimizer.text_output=print
    optimizer.start_fit(model)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plot_result=False
    else:
        plot_result=True
        from numpy import array
        from time import sleep

    if plot_result:
        fig=plt.figure(figsize=(14, 5))
        plt.ion()
        plt.suptitle("To stop fit, close the figure window")

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

        fig.canvas.mpl_connect('close_event', lambda event: optimizer.stop_fit())
        plt.draw()

    last=2
    print("To stop fit, press ctrl+C")
    while optimizer.running:
        try:
            if plot_result:
                if len(optimizer.fom_log)<=last:
                    plt.pause(0.1)
                    continue
                x, y=array(optimizer.fom_log).T
                last=len(x)
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
            else:
                sleep(0.1)
        except KeyboardInterrupt:
            optimizer.stop_fit()
    if plot_result:
        plt.close()
    print("If you want to update the model with the fit results, call api.fit_update(model, optimizer)")

def fit_update(model, optimizer):
    pnames=model.parameters.get_fit_pars()[1]
    for di in model.parameters.data:
        if di[0] in pnames:
            di[1]=optimizer.best_vec[pnames.index(di[0])]
    model.simulate()

class Reflectivity(SampleBuilder):
    """
    Interface to build a model script for reflectivity simulations. Surves the same purpose
    as the Reflectivity plugin in the GUI.
    """

    def __init__(self, model, optimizer, analyze_model=True):
        SampleBuilder.__init__(self, model)
        self.optimizer=optimizer
        if analyze_model:
            self.ReadModel()

    @classmethod
    def create_new(cls, modelname='spec_nx'):
        """
        Create a completely new GenX model for reflectivity.

        returns model, optimizer, Reflectivity
        """
        model=Model()
        optimizer=DiffEv()
        optimizer.model=model
        out=cls(model, optimizer, analyze_model=False)
        out.new_model(modelname)
        return model,optimizer, out

    def AppendSim(self, sim_func, inst, args):
        self.sim_funcs.append(sim_func)
        self.sim_insts.append(inst)
        self.sim_args.append(args)

    def new_model(self, modelname='spec_nx'):
        '''Init the script in the model to yield the
        correct script for initilization
        '''
        modelname='models.'+modelname
        model_data=self.GetModel().get_data()
        nb_data_sets=len(model_data)

        script=self.GetNewModelScript(modelname, nb_data_sets)
        self.BuildNewModel(script)

        self.instruments={'inst': self.model.Instrument()}

        self.data_names=[data_set.name for data_set in model_data]
        self.uservars_lines=[]
        self.sim_exp=[[] for item in self.data_names]
        self.sim_funcs=['Specular']*nb_data_sets
        self.sim_insts=['inst']*nb_data_sets
        self.sim_args=[['d.x'] for i in range(nb_data_sets)]
        self.WriteModel()

    def WriteModel(self):
        parameter_list=self.uservars_lines
        sim_funcs, sim_insts, sim_args = self.sim_funcs, self.sim_insts, self.sim_args
        expression_list=self.sim_exp
        instruments=self.instruments

        self.write_model_script(sim_funcs, sim_insts, sim_args,
                                expression_list, parameter_list, instruments)

    def ReadModel(self):
        '''ReadModel(self)  --> None

        Reads in the current model and locates layers and stacks
        and sample defined inside BEGIN Sample section.
        '''
        self.CompileScript()

        instrument_names=self.find_instrument_names()

        if len(instrument_names)==0:
            raise ValueError('Could not find any Instruments in the'
                                 ' model script. Check the script.')

        if not 'inst' in instrument_names:
            raise ValueError('Could not find the default Instrument, inst, in the'
                                 ' model script. Check the script.')

        sample_text=self.find_sample_section()

        if sample_text is None:
            raise ValueError('Could not find the sample section in the model script.\n '
                             'Can not load the sample in the editor.')

        all_names, layers, stacks=self.find_layers_stacks(sample_text)

        if len(layers)==0:
            raise ValueError('Could not find any Layers in the'
                             ' model script. Check the script.')

        # Now its time to set all the parameters so that we have the strings
        # instead of the evaluated value - looks better
        for lay in layers:
            for par in lay[1].split(','):
                vars=par.split('=')
                exec('%s.%s = "%s"'%(lay[0], vars[0].strip(), vars[1].strip()), self.GetModel().script_module.__dict__)

        data_names, insts, sim_args, sim_exp, sim_funcs=self.find_sim_function_parameters()

        uservars_lines=self.find_user_parameters()

        self.model=self.GetModel().script_module.model
        sample=self.GetModel().script_module.sample

        self.sampleh=SampleHandler(sample, all_names)
        self.sampleh.model=self.model
        instruments={}
        for name in instrument_names:
            instruments[name]=getattr(self.GetModel().script_module, name)
        self.instruments=instruments

        self.data_names=data_names
        self.sim_exp=sim_exp
        self.uservars_lines=uservars_lines

        self.sim_funcs=sim_funcs
        self.sim_insts=insts
        self.sim_args=sim_args

        self.GetModel().compiled=False

    _uvar_string='cp.new_var('

    def _repr_html_(self):
        output=''
        output+='<h3>Instruments:</h3>\n<div>'
        for key, value in self.instruments.items():
            output+=self.sampleh.htmlize('%s = %s'%(key, value))
            output+='<br />'
        output+='</div>'

        output+='<h3>Sample Structure:</h3>\n<div>'
        output+='<br />\n'.join(self.sampleh.getStringList(html_encoding=True))
        output+='</div>'

        output+='<h3>User Parameters:</h3>\n<div><ul>'
        for line in self.uservars_lines:
            var,val=line[len(self._uvar_string):].split(')')[0].split(',')
            output+='<li><b>%s</b> = %s</li>'%(eval(var), val)
        output+='</ul></div>'

        output+='<h3>Simulations:</h3>\n<div>'
        for name, func, inst, exps in zip(self.data_names, self.sim_funcs, self.sim_insts, self.sim_exp):
            output+='<b>%s:</b> %s(%s, %s)'%(name, func, 'd.x', inst)
            output+='<ul>'
            for ei in exps:
                output+='<li>%s</li>'%ei
            output+='</ul><br />'
        output+='</div>'

        return output

    def add_stack(self, name, **kwargs):
        if self.sampleh.checkName(name) or not name.isidentifier():
            raise KeyError("Name already exists or not proper identifier")
        self.sampleh.insertItem(len(self.sampleh)-1, 'Stack', name)
        stack=self.sampleh[name]
        for key, value in kwargs.items():
            setattr(stack, key, value)
        self.WriteModel()
        return self[name]

    def add_layer(self, name, layer=None, **kwargs):
        if self.sampleh.checkName(name) or not name.isidentifier():
            raise KeyError("Name already exists or not proper identifier")
        self.sampleh.insertItem(len(self.sampleh)-1, 'Layer', name)
        lay=self.sampleh[name]
        if layer is not None:
            self[name]=layer
        else:
            for key, value in kwargs.items():
                setattr(lay, key, value)
            self.WriteModel()
        return self[name]

    def move_up(self, item):
        self.sampleh.moveUp(item)
        self.WriteModel()

    def move_down(self, item):
        self.sampleh.move_down(item)
        self.WriteModel()

    @property
    def Layer(self):
        return self.model.Layer

    @property
    def Stack(self):
        return self.model.Stack

    @property
    def Instrument(self):
        return self.model.Instrument

    def __getitem__(self, item):
        if item in self.sampleh.names:
            return self.sampleh[item]
        elif item in self.instruments:
            return self.instruments[item]
        else:
            raise KeyError("Has to be name of a Stack, Layer or Instrument")

    def __setitem__(self, key, value):
        if key in self.sampleh.names:
            prev=self.sampleh[key]
            if type(prev)!=type(value):
                raise ValueError("Has to be of type %s"%type(prev))
            for k in prev._parameters.keys():
                setattr(prev, k, getattr(value, k))
        elif key in self.instruments:
            prev=self.instruments[key]
            if type(prev)!=type(value):
                raise ValueError("Has to be of type %s"%type(prev))
            self.instruments[key]=value
        else:
            raise KeyError("Has to be name of a Stack, Layer or Instrument")
        self.WriteModel()

class DataLoaderInterface():

    def __init__(self):
        head=os.path.dirname(os.path.abspath(__file__))
        self._handler=PluginHandler(None, os.path.join(head, 'plugins', ''), 'data_loaders')
        for dl in self._handler.get_possible_plugins():
            try:
                self._handler.load_plugin(dl)
                setattr(self, dl, self._handler.loaded_plugins[dl])
            except Exception as error:
                print(error)

    def __repr__(self):
        output="Available data loaders:"
        for dl in sorted(self._handler.loaded_plugins.keys()):
            output+='\n  data_loader.%s.LoadData(dataset, filename)'%dl
        return output

data_loader=DataLoaderInterface()
