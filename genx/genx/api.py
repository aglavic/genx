"""
Scripting interface for GenX for use in python scripting or Jupyter Notebooks.
"""

__all__=[]

import os
import sys

# workaround for issues with ctrl+c on windows
if sys.platform=='win32':
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER']='1'

from .model import Model
from .diffev import DiffEv
from .model_control import ModelController
from .plugins.add_ons.help_modules.reflectivity_utils import avail_models, SampleHandler, SampleBuilder
from .plugins.utils import PluginHandler

_fit_output=[]

controller=ModelController(DiffEv())

def text_output_api(text):
    _fit_output.append(text)

def load(fname, compile=True):
    controller.load_file(fname)
    if compile:
        controller.model.compile_script()
    controller.optimizer.model=controller.model
    return controller.model, controller.optimizer

def save(fname, model=None, optimizer=None):
    if model is not None:
        controller.model=model
    if optimizer is not None:
        controller.optimizer=optimizer
    controller.save_file(fname)

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

    display(fig)

    last=2
    while optimizer.running:
        try:
            sleep(0.1)
            if len(optimizer.fom_log)<=last:
                continue
            x, y=array(optimizer.fom_log).T
            last=len(x)
            # t1.set_text('FOM: %.4e'%optimizer.best_fom)
            t1.set_text(_fit_output[-1])
            # vec=optimizer.best_vec
            # list(map(lambda func, value: func(value), model.get_fit_pars()[0], vec))
            # model.evaluate_sim_func()
            ax2.clear()
            t2 = plt.title('Data display')
            refls = []
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

            line.set_xdata(x)
            line.set_ydata(y)

            ax1.set_xlim(0, x[-1])
            ax1.set_ylim(y.min()*0.9, y.max()*1.1)
            try:
                plt.draw()
            except Exception as e:
                print(e)
            else:
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
    _uvar_string='cp.new_var('

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
        return model, optimizer, out

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
        sim_funcs, sim_insts, sim_args=self.sim_funcs, self.sim_insts, self.sim_args
        expression_list=self.sim_exp
        instruments=self.instruments

        # make sure the number of datasets is reflected correctly
        model_data=self.GetModel().get_data()
        nb_data_sets=len(model_data)
        diff=nb_data_sets-len(sim_funcs)
        if diff>0:
            expression_list+=[[] for item in range(diff)]
            for i in range(diff):
                self.insert_new_data_segment(len(sim_funcs))
        elif diff<0:
            for i in range(-diff):
                expression_list.pop(-1)
                self.remove_data_segment(nb_data_sets)

        self.write_model_script(sim_funcs, sim_insts, sim_args,
                                expression_list, parameter_list, instruments)

    def update_script(self):
        self.WriteModel()

    def add_data(self, name=''):
        self.GetModel().data.add_new(name=name)
        self.update_script()

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

    def add_stack(self, name, stack=None, **kwargs):
        if self.sampleh.checkName(name) or not name.isidentifier():
            raise KeyError("Name already exists or not proper identifier")
        self.sampleh.insertItem(len(self.sampleh)-1, 'Stack', name)
        stack=self.sampleh[name]
        if stack is not None:
            self[name]=stack
        else:
            for key, value in kwargs.items():
                setattr(stack, key, value)
            self.WriteModel()
        return self[name]

    def add_layer(self, name, layer=None, **kwargs):
        if self.sampleh.checkName(name) or name in self.instruments or not name.isidentifier():
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

    def add_instrument(self, name, instrument=None, **kwargs):
        if self.sampleh.checkName(name) or name in self.instruments or not name.isidentifier():
            raise KeyError("Name already exists or not proper identifier")
        if instrument is None:
            self.instruments[name]=self.Instrument(**kwargs)
            self.WriteModel()
        else:
            self.instruments[name]=instrument
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

    def __repr__(self):
        output='Reflectivity Model:\n'
        output+='    Instruments:'
        for key, value in self.instruments.items():
            output+='\n        %s = %s'%(key, value)

        output+='\n    Sample Structure:\n        '
        output+='\n        '.join(self.sampleh.getStringList(html_encoding=False))

        output+='\n    User Parameters:'
        for line in self.uservars_lines:
            var, val=line[len(self._uvar_string):].split(')')[0].split(',')
            output+='\n        %s = %s'%(eval(var), val)

        output+='\n    Simulations:'
        for name, func, inst, exps in zip(self.data_names, self.sim_funcs, self.sim_insts, self.sim_exp):
            output+='\n        %s: %s(%s, %s)'%(name, func, 'd.x', inst)
            for ei in exps:
                output+='\n            %s'%ei

        return output

    def _repr_html_(self):
        output='<h3>Reflectivity Model</h3>'
        output+='<h4>Instruments:</h4>\n<div>'
        for key, value in self.instruments.items():
            output+=self.sampleh.htmlize('%s = %s'%(key, value))
            output+='<br />'
        output+='</div>'

        output+='<h4>Sample Structure:</h4>\n<div>'
        output+='<br />\n'.join(self.sampleh.getStringList(html_encoding=True))
        output+='</div>'

        output+='<h4>User Parameters:</h4>\n<div><ul>'
        for line in self.uservars_lines:
            var, val=line[len(self._uvar_string):].split(')')[0].split(',')
            output+='<li><b>%s</b> = %s</li>'%(eval(var), val)
        output+='</ul></div>'

        output+='<h4>Simulations:</h4>\n<div>'
        for name, func, inst, exps in zip(self.data_names, self.sim_funcs, self.sim_insts, self.sim_exp):
            output+='<b>%s:</b> %s(%s, %s)'%(name, func, 'd.x', inst)
            output+='<ul>'
            for ei in exps:
                output+='<li>%s</li>'%ei
            output+='</ul><br />'
        output+='</div>'

        return output

    @property
    def widget(self):
        # alternative widget to the model view with reflectivity controlls
        import ipywidgets as ipw
        model=self.GetModel()
        try:
            model.simulate()
        except Exception as error:
            print(error)

        graphw=ipw.Output()
        with graphw:
            from matplotlib import pyplot as plt
            fig=plt.figure(figsize=(10, 8))
            model.data.plot()
            plt.xlabel('q/tth')
            plt.ylabel('Intensity')
            from IPython.display import display
            display(fig)
            plt.close()

        dataw=model.data._repr_ipyw_()
        top=ipw.HBox([dataw, graphw])
        parameters=model.parameters._repr_ipyw_()

        replot=ipw.Button(description='simulate')
        replot.on_click(model._ipyw_replot)
        replot._plot_output=graphw

        tabs=ipw.Tab(children=[parameters, self.sample_widget, self.instrument_widget])
        tabs.set_title(0, 'Parameters')
        tabs.set_title(1, 'Sample')
        tabs.set_title(2, 'Instrument')

        return ipw.VBox([replot, top, tabs])

    @property
    def instrument_widget(self):
        import ipywidgets as ipw
        entries=[ipw.HTML('<b>Instruments (use string to define formulas):</b>')]
        for key, value in self.instruments.items():
            ient=[]
            if key=='inst':
                name=ipw.HTML(key, layout=ipw.Layout(width='14ex'))
            else:
                name=ipw.Text(value=key, layout=ipw.Layout(width='14ex'))
                name.observe(self._ipyw_change_iname, names='value')
            ient.append(name)
            ient.append(ipw.HTML('=Instrument(', layout=ipw.Layout(width='12ex')))

            items=str(value).split('(', 1)[1].rsplit(')', 1)[0].split(',')
            for item in items:
                try:
                    name, val=item.split('=', 1)
                except ValueError:
                    break
                name=name.strip()
                label_item=ipw.HTML('%s='%name, layout=ipw.Layout(width='%iex'%(len(name)+1)))
                ient.append(label_item)
                entry=ipw.Text(value=val, layout=ipw.Layout(width='18ex'))
                ient.append(entry)
                entry.change_item=name
                entry.change_name=key
                entry.label_item=label_item
                entry.observe(self._ipyw_change_eval, names='value')

            ient.append(ipw.HTML(')', layout=ipw.Layout(width='1ex')))
            entries.append(ipw.HBox(ient))
        btn=ipw.Button(description='Add Instrument')
        btn.on_click(self._ipyw_new_instrument)
        entries.append(btn)
        vbox=ipw.VBox(entries)
        btn.vbox=vbox
        return vbox

    @property
    def sample_widget(self):
        import ipywidgets as ipw
        entries=[ipw.HTML('<b>Sample (use string to define formulas):</b>')]

        maxlen=200
        vbox=ipw.VBox([])

        for i, line in enumerate(self.sampleh.getStringList(html_encoding=False)):
            key, data=map(str.strip, line.split('=', 1))
            ient=[]
            if key in ['Amb', 'Sub']:
                name_entr=ipw.HTML(key, layout=ipw.Layout(width='14ex'))
            else:
                name_entr=ipw.Text(value=key, layout=ipw.Layout(width='14ex'))
                name_entr.observe(self._ipyw_change_sname, names='value')
            ient.append(name_entr)

            if data.startswith('model.Stack'):
                ient.append(ipw.HTML('=Stack(', layout=ipw.Layout(width='12ex')))
                values=data[12:].strip()
            else:
                if key not in ['Amb', 'Sub']:
                    ient.insert(0, ipw.HTML("    ", layout=ipw.Layout(width='4ex')))
                cls, values=map(str.strip, data.split('(', 1))
                values=values.rsplit(')', 1)[0]
                ient.append(ipw.HTML('=%s('%cls[6:], layout=ipw.Layout(width='12ex')))

            thislen=14+12+4
            items=values.split(',')
            for item in items:
                try:
                    name, val=item.split('=', 1)
                except ValueError:
                    break
                name=name.strip()
                label_item=ipw.HTML('%s='%name, layout=ipw.Layout(width='%iex'%(len(name)+1)))
                ient.append(label_item)
                entry=ipw.Text(value=val, layout=ipw.Layout(width='18ex'))
                ient.append(entry)
                entry.change_item=name
                entry.change_name=key
                entry.label_item=label_item
                entry.observe(self._ipyw_change, names='value')
                thislen+=20+(len(name)+1)

            maxlen=max(maxlen, thislen+1)
            ient.append(ipw.HTML(')', layout=ipw.Layout(width='1ex')))
            if data.startswith('model.Stack'):
                btn=ipw.Button(description='Add Layer')
                btn.on_click(self._ipyw_new_layer)
                btn.change_stack=key
                btn.vbox=vbox
                ient.append(btn)
            else:
                name_entr.input_widgets=tuple(wi for wi in ient[2:] if type(wi) is ipw.Text)
            entries.append(ipw.HBox(ient))

        btn=ipw.Button(description='Add Stack')
        btn.on_click(self._ipyw_new_stack)
        entries.append(btn)
        btn.vbox=vbox
        vbox.children=tuple(entries)
        vbox.layout.width='%iex'%maxlen
        return vbox

    def _ipyw_change(self, change):
        item=self[change.owner.change_name]
        try:
            eval(change.new, self.GetModel().script_module.__dict__)
        except Exception as err:
            change.owner.label_item.value='<div style="color: red;">%s=</div>'%change.owner.change_item
        else:
            change.owner.label_item.value='%s='%change.owner.change_item
            setattr(item, change.owner.change_item, change.new)
            self.WriteModel()

    def _ipyw_change_eval(self, change):
        item=self[change.owner.change_name]
        try:
            setattr(item, change.owner.change_item, eval(change.new, self.GetModel().script_module.__dict__))
        except Exception as err:
            change.owner.label_item.value='<div style="color: red;">%s=</div>'%change.owner.change_item
        else:
            change.owner.label_item.value='%s='%change.owner.change_item
            self.WriteModel()

    def _ipyw_change_iname(self, change):
        self.instruments[change.new]=self.instruments[change.old]
        del (self.instruments[change.old])
        for i, inst in enumerate(self.sim_insts):
            if inst==change.old:
                self.sim_insts[i]=str(change.new)
        self.WriteModel()

    def _ipyw_new_instrument(self, btn):
        key='inst_%i'%len(self.instruments)
        self.add_instrument(key)
        value=self[key]

        import ipywidgets as ipw
        ient=[]
        name=ipw.Text(value=key, layout=ipw.Layout(width='14ex'))
        name.observe(self._ipyw_change_iname, names='value')
        ient.append(name)
        ient.append(ipw.HTML('=Instrument(', layout=ipw.Layout(width='12ex')))

        items=str(value).split('(', 1)[1].rsplit(')', 1)[0].split(',')
        for item in items:
            try:
                name, val=item.split('=', 1)
            except ValueError:
                break
            name=name.strip()
            label_item=ipw.HTML('%s='%name, layout=ipw.Layout(width='%iex'%(len(name)+1)))
            ient.append(label_item)
            entry=ipw.Text(value=val, layout=ipw.Layout(width='18ex'))
            ient.append(entry)
            entry.change_item=name
            entry.change_name=key
            entry.label_item=label_item
            entry.observe(self._ipyw_change, names='value')

        ient.append(ipw.HTML(')', layout=ipw.Layout(width='1ex')))
        btn.vbox.children=btn.vbox.children[:-1]+(ipw.HBox(ient), btn.vbox.children[-1])
        self.WriteModel()

    def _ipyw_new_layer(self, btn):
        idx=[bi.children[-1] for bi in btn.vbox.children[1:-1]].index(btn)+1
        key='new_layer_%i'%len(self.sampleh.names)
        self.sampleh.insertItem(btn.change_stack, 'Layer', key)
        lay=self[key]

        import ipywidgets as ipw
        ient=[]
        name_entr=ipw.Text(value=key, layout=ipw.Layout(width='14ex'))
        name_entr.observe(self._ipyw_change_sname, names='value')
        ient.append(name_entr)
        data=repr(lay)

        ient.insert(0, ipw.HTML("    ", layout=ipw.Layout(width='4ex')))
        cls, values=map(str.strip, data.split('(', 1))
        values=values.rsplit(')', 1)[0]
        ient.append(ipw.HTML('=%s('%cls, layout=ipw.Layout(width='12ex')))

        items=values.split(',')
        for item in items:
            try:
                name, val=item.split('=', 1)
            except ValueError:
                break
            name=name.strip()
            label_item=ipw.HTML('%s='%name, layout=ipw.Layout(width='%iex'%(len(name)+1)))
            ient.append(label_item)
            entry=ipw.Text(value=val, layout=ipw.Layout(width='18ex'))
            ient.append(entry)
            entry.change_item=name
            entry.change_name=key
            entry.label_item=label_item
            entry.observe(self._ipyw_change, names='value')
        ient.append(ipw.HTML(')', layout=ipw.Layout(width='1ex')))
        name_entr.input_widgets=tuple(wi for wi in ient[2:] if type(wi) is ipw.Text)

        btn.vbox.children=btn.vbox.children[:idx+1]+(ipw.HBox(ient),)+btn.vbox.children[idx+1:]
        self.WriteModel()

    def _ipyw_change_sname(self, change):
        idx=self.sampleh.names.index(change.old)
        self.sampleh.names[idx]=change.new
        # update the name for the layer entries
        for entr in change.owner.input_widgets:
            entr.change_name=change.new
        self.WriteModel()

    def _ipyw_new_stack(self, btn):
        pass

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
