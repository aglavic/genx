"""
Help class for the Reflectivity plugin to analyze and change the script.
"""

import re

from genx.models.lib.refl_base import ReflBase

from .reflectivity_misc import ReflectivityModule

avail_models = ["spec_nx", "spec_inhom", "spec_adaptive", "interdiff", "mag_refl", "soft_nx"]


def default_html_decorator(name, str):
    return str


class SampleHandler:
    model: ReflectivityModule

    def __init__(self, sample, names):
        self.sample = sample
        self.names = names
        self.getStringList()

    def set_model(self, model: ReflectivityModule):
        self.model = model

    def getStringListNew(self, html_encoding=False, html_decorator=default_html_decorator):
        """
        Function to generate a list of strings that gives
        a visual representation of the sample.

        This is based on the new ModelParamBase derived reflectivity classes that don't require
        analysis of the model script.
        """
        slist = [self.sample.Substrate._repr_call()]
        poslist = [(None, None)]
        i = 0
        for stack in self.sample.Stacks:
            j = 0
            for layer in stack.Layers:
                slist.append(layer._repr_call())
                poslist.append((i, j))
                j += 1
            slist.append("Stack: Repetitions = %s" % stack._ca.get("Repetitions", stack.Repetitions))
            for key in list(stack._parameters.keys()):
                if not key in ["Repetitions", "Layers"]:
                    slist[-1] += ", %s = %s" % (key, stack._ca.get(key, repr(getattr(stack, key))))
            poslist.append((i, None))
            i += 1
        slist.append(self.sample.Ambient._repr_call())
        for item in range(len(slist)):
            name = self.names[-item - 1]
            par_str = slist[item]
            if slist[item][0] == "L" and item != 0 and item != len(slist) - 1:
                if html_encoding:
                    slist[item] = (
                        "<code>&nbsp;&nbsp;&nbsp;<b>" + name + "</b> = " + html_decorator(name, par_str) + "</code>"
                    )
                else:
                    slist[item] = self.names[-item - 1] + " = model." + slist[item]
            else:
                if item == 0 or item == len(slist) - 1:
                    # This is then the ambient or substrates
                    if html_encoding:
                        slist[item] = "<code><b>" + name + "</b> = " + html_decorator(name, par_str) + "</code>"
                    else:
                        slist[item] = self.names[-item - 1] + " = model." + slist[item]
                else:
                    # This is a stack!
                    if html_encoding:
                        slist[item] = (
                            '<font color = "BLUE"><code><b>'
                            + name
                            + "</b> = "
                            + html_decorator(name, par_str)
                            + "</code></font>"
                        )
                    else:
                        slist[item] = self.names[-item - 1] + " = model." + slist[item]
        poslist.append((None, None))
        slist.reverse()
        poslist.reverse()
        self.poslist = poslist
        return slist

    def getStringList(self, html_encoding=False, html_decorator=default_html_decorator):
        """
        Function to generate a list of strings that gives
        a visual representation of the sample.
        """
        if isinstance(self.sample, ReflBase):
            return self.getStringListNew(html_encoding, html_decorator)

        slist = [self.sample.Substrate.__repr__()]
        poslist = [(None, None)]
        i = 0
        j = 0
        for stack in self.sample.Stacks:
            j = 0
            for layer in stack.Layers:
                slist.append(layer.__repr__())
                poslist.append((i, j))
                j += 1
            slist.append("Stack: Repetitions = %s" % str(stack.Repetitions))
            for key in list(stack._parameters.keys()):
                if not key in ["Repetitions", "Layers"]:
                    slist[-1] += ", %s = %s" % (key, str(getattr(stack, key)))
            poslist.append((i, None))
            i += 1
        slist.append(self.sample.Ambient.__repr__())
        for item in range(len(slist)):
            name = self.names[-item - 1]
            par_str = slist[item]
            if slist[item][0] == "L" and item != 0 and item != len(slist) - 1:
                if html_encoding:
                    slist[item] = (
                        "<code>&nbsp;&nbsp;&nbsp;<b>" + name + "</b> = " + html_decorator(name, par_str) + "</code>"
                    )
                else:
                    slist[item] = self.names[-item - 1] + " = model." + slist[item]
            else:
                if item == 0 or item == len(slist) - 1:
                    # This is then the ambient or substrates
                    if html_encoding:
                        slist[item] = "<code><b>" + name + "</b> = " + html_decorator(name, par_str) + "</code>"
                    else:
                        slist[item] = self.names[-item - 1] + " = model." + slist[item]
                else:
                    # This is a stack!
                    if html_encoding:
                        slist[item] = (
                            '<font color = "BLUE"><code><b>'
                            + name
                            + "</b> = "
                            + html_decorator(name, par_str)
                            + "</code></font>"
                        )
                    else:
                        slist[item] = self.names[-item - 1] + " = model." + slist[item]
        poslist.append((None, None))
        slist.reverse()
        poslist.reverse()
        self.poslist = poslist
        return slist

    def htmlize(self, code):
        """htmlize(self, code) --> code

        htmlize the code for display
        """
        p = code.index("=")
        name = "<code><b>%s</b></code>" % code[:p]
        items = code[p:].split(",")
        return name + "".join(["<code>%s,</code>" % item for item in items])

    def getCodeNew(self):
        """
        Generate the python code for the current sample structure.
        """
        slist = self.getStringListNew()
        layer_code = ""

        # Create code for the layers:
        for item in slist:
            if item.find("Layer") > -1:
                itemp = item.lstrip()
                layer_code = layer_code + itemp + "\n"
        # Create code for the Stacks:
        i = 0
        stack_code = ""
        item = slist[i]
        maxi = len(slist) - 1
        while i < maxi:
            if item.find("Stack") > -1:
                stack_strings = item.split(":")
                stack_code = stack_code + stack_strings[0] + "(Layers=["
                i += 1
                item = slist[i]
                stack_layers = []
                while item.find("Stack") < 0 and i < maxi:
                    itemp = item.split("=")[0]
                    itemp = itemp.lstrip()
                    stack_layers.append(itemp)
                    # stack_code = stack_code + itemp+','
                    i += 1
                    item = slist[i]
                stack_layers.reverse()
                stack_code += ", ".join(stack_layers)
                i -= 1
                if stack_code[-1] != "[":
                    stack_code = stack_code[:-1] + "]," + stack_strings[1] + ")\n"
                else:
                    stack_code = stack_code[:] + "]," + stack_strings[1] + ")\n"
            i += 1
            item = slist[i]
        # Create the code for the sample
        sample_code = "sample = model.Sample(Stacks = ["
        stack_strings = stack_code.split("\n")
        rest_sample_rep = "], "
        sample_string_pars = ", ".join(
            [
                f"{name} = {self.sample._ca.get(name)}"
                for name in self.sample._parameters.keys()
                if name in self.sample._ca and name not in ["Stacks", "Ambient", "Substrate"]
            ]
        )
        if len(sample_string_pars) != 0:
            rest_sample_rep += sample_string_pars + ", "
        rest_sample_rep += "Ambient = Amb, Substrate = Sub)\n"
        if stack_strings != [""]:
            stack_strings.reverse()
            for item in stack_strings[1:]:
                itemp = item.split("=")[0]
                sample_code = sample_code + itemp + ","
            sample_code = sample_code[:-2] + rest_sample_rep
        else:
            sample_code += rest_sample_rep

        return layer_code, stack_code, sample_code

    def getCode(self):
        """
        Generate the python code for the current sample structure.
        """
        if isinstance(self.sample, ReflBase):
            return self.getCodeNew()
        slist = self.getStringList()
        layer_code = ""

        # Create code for the layers:
        for item in slist:
            if item.find("Layer") > -1:
                itemp = item.lstrip()
                layer_code = layer_code + itemp + "\n"
        # Create code for the Stacks:
        i = 0
        stack_code = ""
        item = slist[i]
        maxi = len(slist) - 1
        while i < maxi:
            if item.find("Stack") > -1:
                stack_strings = item.split(":")
                stack_code = stack_code + stack_strings[0] + "(Layers=["
                i += 1
                item = slist[i]
                stack_layers = []
                while item.find("Stack") < 0 and i < maxi:
                    itemp = item.split("=")[0]
                    itemp = itemp.lstrip()
                    stack_layers.append(itemp)
                    # stack_code = stack_code + itemp+','
                    i += 1
                    item = slist[i]
                stack_layers.reverse()
                stack_code += ", ".join(stack_layers)
                i -= 1
                if stack_code[-1] != "[":
                    stack_code = stack_code[:-1] + "]," + stack_strings[1] + ")\n"
                else:
                    stack_code = stack_code[:] + "]," + stack_strings[1] + ")\n"
            i += 1
            item = slist[i]
        # Create the code for the sample
        sample_code = "sample = model.Sample(Stacks = ["
        stack_strings = stack_code.split("\n")
        rest_sample_rep = "], "
        sample_string_pars = self.sample.__repr__().split(":")[1].split("\n")[0].lstrip()
        if len(sample_string_pars) != 0:
            sample_string_pars += ", "
        rest_sample_rep += sample_string_pars + "Ambient = Amb, Substrate = Sub)\n"
        if stack_strings != [""]:
            # Added 20080831 MB bugfix
            stack_strings.reverse()
            for item in stack_strings[1:]:
                itemp = item.split("=")[0]
                sample_code = sample_code + itemp + ","
            sample_code = sample_code[:-2] + rest_sample_rep
        else:
            sample_code += rest_sample_rep

        return layer_code, stack_code, sample_code

    def getItem(self, pos):
        """
        Returns the item (Stack or Layer) at position pos
        """
        if pos == 0:
            return self.sample.Ambient
        if pos == len(self.poslist) - 1:
            return self.sample.Substrate
        stack = self.sample.Stacks[self.poslist[pos][0]]
        if self.poslist[pos][1] is None:
            return stack
        return stack.Layers[self.poslist[pos][1]]

    def __getitem__(self, item):
        if item in self.names:
            return self.getItem(self.names.index(item))
        elif type(item) is int:
            return self.getItem(item)
        else:
            raise IndexError("Item has to be in .names list or an integer")

    def __delitem__(self, key):
        return self.deleteItem(key)

    def __len__(self):
        return len(self.names)

    def deleteItem(self, pos):
        """
        Delete item pos in the lsit if the item is a stack all the Layers
        are deleted as well.
        """
        if pos in self.names:
            pos = self.names.index(pos)

        if pos == 0:
            return None
        if pos == len(self.poslist) - 1:
            return None
        stack = self.sample.Stacks[self.poslist[pos][0]]
        if self.poslist[pos][1] is None:
            self.sample.Stacks.pop(self.poslist[pos][0])
            p = self.poslist[pos][0]
            pt = pos
            while self.poslist[pt][0] == p:
                pt += 1
            pt -= 1
            while self.poslist[pt][0] == p:
                self.names.pop(pt)
                pt -= 1

        else:
            stack.Layers.pop(self.poslist[pos][1])
            self.names.pop(pos)
        return self.getStringList()

    def insertItem(self, pos, type, name="test"):
        """
        Insert an item into the sample at position pos in the list
        and of type. type is a string of either Stack or Layer
        """
        if pos in self.names:
            pos = self.names.index(pos)

        spos = self.poslist[pos]
        added = False
        last = False
        if pos == 0:
            spos = (self.poslist[1][0], self.poslist[1][1])  # +1
            # spos=(None,None)
        if pos == len(self.poslist) - 1:
            spos = self.poslist[-2]
            last = True
        stackp = False
        if spos[1] is None:
            spos = (spos[0], 0)
            stackp = True
        if spos[0] is None:
            spos = (0, spos[1])

        # If it not the first item i.e. can't insert anything before the
        # ambient layer
        if pos != 0:
            if type == "Stack":
                stack = self.model.Stack(Layers=[])
                if last:
                    self.names.insert(pos, name)
                else:
                    if stackp:
                        self.names.insert(pos + len(self.sample.Stacks[spos[0]].Layers) + 1, name)
                    else:
                        self.names.insert(pos + spos[1] + 1, name)
                self.sample.Stacks.insert(spos[0], stack)
                added = True

            if type == "Layer" and len(self.poslist) > 2:
                layer = self.model.Layer()
                if last:
                    self.names.insert(pos, name)
                else:
                    if spos[1] >= 0:
                        self.names.insert(pos + 1, name)
                    else:
                        self.names.insert(pos + len(self.sample.Stacks[spos[0]].Layers) + 1, name)
                if last:
                    self.sample.Stacks[spos[0]].Layers.insert(0, layer)
                else:
                    if self.poslist[pos][1] is None:
                        self.sample.Stacks[spos[0]].Layers.append(layer)
                    else:
                        self.sample.Stacks[spos[0]].Layers.insert(spos[1], layer)
                added = True

        else:
            if type == "Stack":
                stack = self.model.Stack(Layers=[])
                self.sample.Stacks.append(stack)
                added = True
                self.names.insert(pos + 1, name)
            if type == "Layer" and len(self.poslist) > 2:
                layer = self.model.Layer()
                self.sample.Stacks[spos[0]].Layers.append(layer)
                added = True
                self.names.insert(pos + 2, name)
        if added:
            return self.getStringList()
        else:
            return None

    def canInsertLayer(self):
        return self.poslist > 2

    def checkName(self, name):
        return self.names.__contains__(name)

    def getName(self, pos):
        """Returns the name for the object at pos
        :param pos: list position for the name
        :return: the name (string)
        """
        return self.names[pos]

    def changeName(self, pos, name):
        if pos in self.names:
            pos = self.names.index(pos)
        if name in self.names and name != self.names[pos]:
            return False
        elif pos == len(self.names) - 1 or pos == 0:
            return False
        else:
            self.names[pos] = name
            return True

    def moveUp(self, pos):
        """
        Move the item up - with stacks move the entire stack up one step.
        Moves layer only if it is possible.
        """
        if pos in self.names:
            pos = self.names.index(pos)
        if pos > 1 and pos != len(self.poslist) - 1:
            if self.poslist[pos][1] is None:
                temp = self.sample.Stacks.pop(self.poslist[pos][0])
                temps = []
                for index in range(len(temp.Layers) + 1):
                    temps.append(self.names.pop(pos))
                for index in range(len(temp.Layers) + 1):
                    self.names.insert(pos - len(self.sample.Stacks[self.poslist[pos][0]].Layers) - 1, temps[-index - 1])
                self.sample.Stacks.insert(self.poslist[pos][0] + 1, temp)
                return self.getStringList()
            else:  # i.e. it is a layer we move
                if pos > 2:
                    temp = self.sample.Stacks[self.poslist[pos][0]].Layers.pop(self.poslist[pos][1])
                    temps = self.names.pop(pos)
                    if self.poslist[pos - 1][1] is None:  # Next item a Stack i.e. jump up
                        self.sample.Stacks[self.poslist[pos - 2][0]].Layers.insert(0, temp)
                        self.names.insert(pos - 1, temps)
                    else:  # Moving inside a stack
                        self.sample.Stacks[self.poslist[pos][0]].Layers.insert(self.poslist[pos][1] + 1, temp)
                        self.names.insert(pos - 1, temps)
                    return self.getStringList()
                else:
                    return None
        else:
            return None

    def moveDown(self, pos):
        """
        Move the item down - with stacks move the entire stack up one step.
        Moves layer only if it is possible.
        """
        if pos in self.names:
            pos = self.names.index(pos)

        if pos != 0 and pos < len(self.poslist) - 2:
            if self.poslist[pos][1] is None:  # Moving a stack
                if self.poslist[pos][0] != 0:
                    temp = self.sample.Stacks.pop(self.poslist[pos][0])
                    temps = []
                    for index in range(len(temp.Layers) + 1):
                        temps.append(self.names.pop(pos))
                    for index in range(len(temp.Layers) + 1):
                        self.names.insert(
                            pos + len(self.sample.Stacks[self.poslist[pos][0] - 1].Layers) + 1, temps[-index - 1]
                        )
                    self.sample.Stacks.insert(self.poslist[pos][0] - 1, temp)
                    return self.getStringList()
                else:
                    return None

            else:  # i.e. it is a layer we move
                if pos < len(self.poslist) - 2:
                    temp = self.sample.Stacks[self.poslist[pos][0]].Layers.pop(self.poslist[pos][1])
                    temps = self.names.pop(pos)
                    if self.poslist[pos + 1][1] is None:  # Next item a Stack i.e. jump down
                        self.sample.Stacks[self.poslist[pos + 1][0]].Layers.insert(
                            len(self.sample.Stacks[self.poslist[pos + 1][0]].Layers), temp
                        )
                        self.names.insert(pos + 1, temps)
                    else:  # Moving inside a stack
                        self.sample.Stacks[self.poslist[pos][0]].Layers.insert(self.poslist[pos][1] - 1, temp)  # -2
                        self.names.insert(pos + 1, temps)
                    return self.getStringList()
        else:
            return None


class SampleBuilder:
    defs = ["Instrument", "Sample"]
    sim_returns_sld = False

    def __init__(self, model):
        self._model_object = model

    def GetModel(self):
        return self._model_object

    def SetModelScript(self, script):
        """SetModelScript(self, script) --> None

        Sets the script of the current model. This overwrite the current
        script.
        """
        self._model_object.set_script(script)

    def CompileScript(self):
        """CompileScript(self) --> None

        Compiles the model script
        """
        self._model_object.compile_script()

    def GetNewModelScript(self, modelname="models.spec_nx", nb_data_sets=1):
        script = "from numpy import *\n"
        script += "import %s as model\n" % modelname
        script += "from models.utils import UserVars, fp, fw, bc, bw\n\n"
        for item in self.defs:
            script += "# BEGIN %s DO NOT CHANGE\n" % item
            script += "# END %s\n\n" % item
        script += "# BEGIN Parameters DO NOT CHANGE\n"
        script += "cp = UserVars()\n"
        script += "# END Parameters\n\n"
        script += "SLD = []\n"
        script += "def Sim(data):\n"
        script += "    I = []\n"
        script += "    SLD[:] = []\n"
        for i in range(nb_data_sets):
            script += "    # BEGIN Dataset %i DO NOT CHANGE\n" % i
            script += "    d = data[%i]\n" % i
            script += "    I.append(sample.SimSpecular(d.x, inst))\n"
            script += "    if _sim: SLD.append(sample.SimSLD(None, None, inst))\n"
            script += "    # END Dataset %i\n" % i
        script += "    return I\n"
        return script

    def BuildNewModel(self, script):
        self.sim_returns_sld = True
        self.SetModelScript(script)
        self.CompileScript()
        self.model = self.GetModel().script_module.model
        names = ["Amb", "Sub"]
        Amb = self.model.Layer()
        Sub = self.model.Layer()
        sample = self.model.Sample(Stacks=[], Ambient=Amb, Substrate=Sub)
        # self.sample_widget.SetSample(sample, names)
        self.sampleh = SampleHandler(sample, names)
        self.sampleh.model = self.model

    def insert_code_segment(self, code, descriptor, insert_code):
        """insert_code_segment(self, code, descriptor, insert_code) --> None

        Inserts code segment into the file. See find_code segment.
        """
        found = 0
        script_lines = code.splitlines(True)
        start_index = -1
        stop_index = -1
        for line in range(len(script_lines)):
            if script_lines[line].find("# BEGIN %s" % descriptor) != -1:
                start_index = line + 1
            if script_lines[line].find("# END %s" % descriptor) != -1:
                stop_index = line - 1
                break

        # Check so everything have preceeded well
        if stop_index < 0 and start_index < 0:
            raise LookupError("Code segement: %s could not be found" % descriptor)

        # Find the tablevel
        # tablevel = len([' ' for char in script_lines[stop_index+1]\
        #    if char == ' '])
        tablevel = len(script_lines[stop_index + 1]) - len(script_lines[stop_index + 1].lstrip())

        # Make the new code tabbed
        tabbed_code = [" " * tablevel + line for line in insert_code.splitlines(True)]
        # Replace the new code segment with the new
        new_code = "".join(script_lines[:start_index] + tabbed_code + script_lines[stop_index + 1 :])

        return new_code

    def find_code_segment(self, code, descriptor):
        """find_code_segment(self, code, descriptor) --> string

        Finds a segment of code between BEGIN descriptor and END descriptor
        returns a LookupError if the segement can not be found
        """

        return find_code_segment(code, descriptor)

    def insert_new_data_segment(self, number):
        """insert_new_data_segment(self, number) --> None

        Inserts a new data segment into the script
        """
        code = self.GetModel().get_script()
        script_lines = code.splitlines(True)
        line_index = 0
        found = 0
        for line in script_lines[line_index:]:
            line_index += 1
            if line.find("    return I") != -1:
                found = 1
                break

        if found < 1:
            raise LookupError('Could not find "return I" in the script')

        self.AppendSim("Specular", "inst", ["d.x"])

        script = "".join(script_lines[: line_index - 1])
        script += "    # BEGIN Dataset %i DO NOT CHANGE\n" % number
        script += "    d = data[%i]\n" % number
        script += "    I.append(sample.SimSpecular(d.x, inst))\n"
        script += "    if _sim: SLD.append(sample.SimSLD(None, None, inst))\n"
        script += "    # END Dataset %i\n" % number
        script += "".join(script_lines[line_index - 1 :])
        self.SetModelScript(script)

    def remove_data_segment(self, number):
        """remove_data_segment(self, number) --> None

        Removes data segment number
        """
        code = self.GetModel().get_script()
        found = 0
        script_lines = code.splitlines(True)
        start_index = -1
        stop_index = -1
        for line in range(len(script_lines)):
            if script_lines[line].find("# BEGIN Dataset %i" % number) != -1:
                start_index = line + 1
            if script_lines[line].find("# END Dataset %i" % number) != -1:
                stop_index = line - 1
                break

        # Check so everything have preceeded well
        if stop_index < 0 and start_index < 0:
            raise LookupError("Code segement: %s could not be found" % number)

        script = "".join(script_lines[: start_index - 1])
        script += "".join(script_lines[stop_index + 2 :])
        self.SetModelScript(script)

    def write_model_script(self, sim_funcs, sim_insts, sim_args, expression_list, parameter_list, instruments):
        script = self.GetModel().get_script()
        # Instrument script creation
        code = "from models.utils import create_fp, create_fw\n"
        for inst_name in instruments:
            inst_i = instruments[inst_name]
            if isinstance(inst_i, ReflBase):
                inst_repr = inst_i._repr_call()
            else:
                inst_repr = inst_i.__repr__()
            code += f"{inst_name} = model.{inst_repr}\n"
            code += "%s_fp = create_fp(%s.wavelength);" % (inst_name, inst_name)
            code += " %s_fw = create_fw(%s.wavelength)\n\n" % (inst_name, inst_name)
        code += "fp.set_wavelength(inst.wavelength); " + "fw.set_wavelength(inst.wavelength)\n"
        script = self.insert_code_segment(script, "Instrument", code)
        # Sample script creation
        layer_code, stack_code, sample_code = self.sampleh.getCode()
        code = layer_code + "\n" + stack_code + "\n" + sample_code
        script = self.insert_code_segment(script, "Sample", code)
        # User Vars (Parameters) script creation
        code = "cp = UserVars()\n"
        code += "".join([line + "\n" for line in parameter_list])
        script = self.insert_code_segment(script, "Parameters", code)
        # Expressions evaluted during simulations (parameter couplings) script creation
        for i, exps in enumerate(expression_list):
            exp = [ex + "\n" for ex in exps]
            exp.append("d = data[%i]\n" % i)
            str_arg = ", ".join(sim_args[i])
            exp.append("I.append(sample." "Sim%s(%s, %s))\n" % (sim_funcs[i], str_arg, sim_insts[i]))
            if self.sim_returns_sld:
                exp.append("if _sim: SLD.append(sample." "SimSLD(None, None, %s))\n" % sim_insts[i])
            code = "".join(exp)
            script = self.insert_code_segment(script, "Dataset %i" % i, code)
        self.SetModelScript(script)

    def find_user_parameters(self):
        script = self.GetModel().script
        # Load the custom parameters:
        code = self.find_code_segment(script, "Parameters")
        uservars_lines = code.splitlines()[1:]
        return uservars_lines

    def find_sim_function_parameters(self):
        # Load the simulation parameters
        script = self.GetModel().script
        sim_exp = []
        data = self.GetModel().get_data()
        data_names = [di.name for di in data]
        # Lists holding the simulation function arguments
        sim_funcs = []
        sim_args = []
        insts = []
        for i in range(len(data)):
            code = self.find_code_segment(script, "Dataset %i" % i)
            sim_exp.append([])
            # for line in code.splitlines()[:-1]:
            #    sim_exp[-1].append(line.strip())
            for line in code.splitlines():
                if line.find("I.append") == -1 and line.find("SLD.append") == -1 and line.find("d = data") == -1:
                    # The current line is a command for a parameter
                    sim_exp[-1].append(line.strip())
                elif line.find("I.append") > -1:
                    # The current line is a simulations
                    (tmp, sim_func, args) = line.split("(", 2)
                    sim_funcs.append(sim_func[10:])
                    sim_args.append([arg.strip() for arg in args.split(",")[:-1]])
                    insts.append(args.split(",")[-1][:-2].strip())
        return data_names, insts, sim_args, sim_exp, sim_funcs

    def find_layers_stacks(self, sample_text):
        re_layer = re.compile("([A-Za-z]\w*)\s*=\s*model\.Layer\s*\((.*)\)\n")
        re_stack = re.compile("([A-Za-z]\w*)\s*=\s*model\.Stack\s*\(\s*Layers=\[(.*)\].*\n")
        layers = re_layer.findall(sample_text)
        layer_names = [t[0] for t in layers]
        stacks = re_stack.findall(sample_text)

        all_names = [layer_names.pop(0)]
        for stack in stacks:
            all_names.append(stack[0])
            first_name = stack[1].split(",")[0].strip()
            # check so stack is non-empty
            if first_name != "":
                # Find all items above the first name in the stack
                while layer_names[0] != first_name:
                    all_names.append(layer_names.pop(0))
                all_names.append(layer_names.pop(0))
        all_names += layer_names

        return all_names, layers, stacks

    def find_sample_section(self):
        # Get the current script and split the lines into list items
        script_lines = self.GetModel().get_script().splitlines(True)
        # Try to find out if the script works with multiple SLDs
        for line in script_lines:
            if line.find("SLD[:]") != -1:
                self.sim_returns_sld = True
                break
            else:
                self.sim_returns_sld = False
        script = ""
        # Locate the Sample definition
        line_index = 0
        # Start by finding the right section
        found = 0
        for line in script_lines[line_index:]:
            line_index += 1
            if line.find("# BEGIN Sample") != -1:
                found += 1
                break
        sample_text = ""
        for line in script_lines[line_index:]:
            line_index += 1
            sample_text += line
            if line.find("# END Sample") != -1:
                found += 1
                break
        if found != 2:
            return None
        else:
            return sample_text

    def find_instrument_names(self):
        script = self.GetModel().script
        code = self.find_code_segment(script, "Instrument")
        re_layer = re.compile("([A-Za-z]\w*)\s*=\s*model\.Instrument\s*\((.*)\)\n")
        instrument_strings = re_layer.findall(code)
        instrument_names = [t[0] for t in instrument_strings]
        return instrument_names


def find_code_segment(code, descriptor):
    """find_code_segment(code, descriptor) --> string

    Finds a segment of code between BEGIN descriptor and END descriptor
    returns a LookupError if the segement can not be found
    """
    found = 0
    script_lines = code.splitlines(True)
    line_index = 0
    for line in script_lines[line_index:]:
        line_index += 1
        if line.find("# BEGIN %s" % descriptor) != -1:
            found += 1
            break

    text = ""
    for line in script_lines[line_index:]:
        line_index += 1
        if line.find("# END %s" % descriptor) != -1:
            found += 1
            break
        text += line

    if found != 2:
        raise LookupError("Code segement: %s could not be found" % descriptor)

    return text
