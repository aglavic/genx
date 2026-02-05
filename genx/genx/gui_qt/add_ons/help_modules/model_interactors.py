__author__ = "Matts Bjorck"

import inspect
import math

from dataclasses import dataclass

from genx.core.custom_logging import iprint
from genx.models import utils


def replace_list_items(list_a, list_b):
    list_a[:] = []
    [list_a.append(b) for b in list_b]


@dataclass
class SimMethodInfo:
    name: str
    args: list
    def_args: list


class ModelScriptInteractor:
    _begin_string = "# BEGIN %s"
    _end_string = "# END %s"
    _tab_string = 4 * " "
    _sim_def_string = "def Sim(data):\n" + _tab_string + "I = []\n"
    _sim_end_string = _tab_string + "return I"
    _data_set_name = "DataSet"
    _custom_parameter_name = "Parameters"

    def __init__(self, preamble=""):
        self.preable = preamble
        self.code_section_interactors = {}
        self.code_sections = []
        self.interactor_lists = {}
        self.data_sections_interactors = []
        self.custom_parameters_interactors = []

    def add_section(self, name, object_interactor, **kwargs):
        if hasattr(self, name):
            raise ValueError("ModelScriptInteractor: The name %s already exist")

        def constructor():
            return object_interactor(**kwargs)

        self.code_section_interactors[name] = constructor
        self.code_sections.append(name)
        setattr(self, name.lower(), [])

    def append_dataset(self):
        iprint("Append data set")
        self.data_sections_interactors.append(DataSetSimulationInteractor())
        default_name = self.get_sim_object_names()[-1]
        sim_method = self.get_sim_methods_info(default_name)[0]
        self.data_sections_interactors[-1].set_from_sim_method(default_name, sim_method)
        self.data_sections_interactors[-1].instrument = self.instruments[0].name

    def append_custom_parameter(self, name, value):
        self.custom_parameters_interactors.append(CustomParameterInteractor(name, value))

    def delete_custom_parameter(self, index):
        self.custom_parameters_interactors.pop(index)

    def get_sim_object_names(self):
        names = []
        for section in self.code_sections:
            for interactor in getattr(self, section.lower()):
                if interactor.get_sim_methods_info():
                    names.append(interactor.name)
        return names

    def get_sim_methods_info(self, name):
        interactor = self.get_interactor_by_name(name)
        return interactor.get_sim_methods_info()

    def get_interactor_by_name(self, name):
        for section in self.code_sections:
            for interactor in getattr(self, section.lower()):
                if interactor.name.lower() == name.lower():
                    return interactor
        raise ValueError("%s is not a name of an interactor (get_interactor_name)")

    def find_section_index(self, code, name):
        begin_string = self._begin_string % name
        try:
            begin_index = code.index(begin_string) + len(begin_string) + 1
        except ValueError:
            raise ValueError('ModelScriptInteractor Could not locate the beginning of the code section "%s"' % name)
        try:
            end_index = code.index(self._end_string % name)
        except ValueError:
            raise ValueError('ModelScriptInteractor Could not locate the end of the code section "%s"' % name)
        return begin_index, end_index

    def find_new_data_set_insertion_index(self, code):
        try:
            index = code.index(self._sim_end_string)
        except ValueError:
            raise ValueError("ModelScriptInteractor could not locate the end of the simulation function.")
        return index

    def find_section(self, name, code):
        begin_index, end_index = self.find_section_index(code, name)
        code = code[begin_index:end_index].strip()
        return code

    def split_section(self, code, multiline):
        if multiline:
            chunks = [""]
            for line in code.splitlines():
                line = line.strip()
                if line == "":
                    chunks.append("")
                else:
                    chunks[-1] += line + "\n"
        else:
            chunks = code.splitlines()
        return chunks

    def parse_code(self, code):
        for name in self.code_section_interactors:
            interactor_constructor = self.code_section_interactors[name]
            code_section = self.find_section(name, code)
            tmp = interactor_constructor()
            chunks = self.split_section(code_section, tmp.get_multiline())
            interactors = []
            for chunk in chunks:
                interactors.append(interactor_constructor())
                interactors[-1].parse_code(chunk)
            replace_list_items(getattr(self, name.lower()), interactors)

        self.data_sections_interactors[:] = []
        missing_data_section = False
        i = 0
        while not missing_data_section:
            try:
                code_section = self.find_section(self.get_data_set_name(i), code)
            except ValueError:
                missing_data_section = True
            else:
                self.append_dataset()
                self.data_sections_interactors[-1].parse_code(code_section)
                i += 1

        code_section = self.find_section(self._custom_parameter_name, code)
        chunks = self.split_section(code_section, False)
        interactors = []
        for chunk in chunks[1:]:
            interactors.append(CustomParameterInteractor())
            interactors[-1].parse_code(chunk)
        replace_list_items(self.custom_parameters_interactors, interactors)

    def get_section_code(self, name, tab_lvl=0, outer_comments=True, member_name=None):
        if not member_name:
            interactors = getattr(self, name.lower())
        else:
            interactors = getattr(self, member_name)
        code = ""
        if outer_comments:
            code += tab_lvl * self._tab_string + self._begin_string % name + "\n"
        for interactor in interactors:
            for line in interactor.get_code().splitlines():
                code += tab_lvl * self._tab_string + line + "\n"
            if interactor.get_multiline():
                code += "\n"
        if interactors and interactors[-1].get_multiline():
            code = code[:-1]
        if outer_comments:
            code += tab_lvl * self._tab_string + self._end_string % name + "\n"
        return code

    def get_data_set_name(self, index):
        return "%s %d" % (self._data_set_name, index)

    def get_data_set_code(self, index, tab_lvl=0, outer_comments=True):
        code = ""
        name = self.get_data_set_name(index)
        if outer_comments:
            code += tab_lvl * self._tab_string + self._begin_string % name + "\n"
        interactor = self.data_sections_interactors[index]
        interactor.set_pos(index)
        for line in interactor.get_code().splitlines():
            code += tab_lvl * self._tab_string + line + "\n"
        if outer_comments:
            code += tab_lvl * self._tab_string + self._end_string % name + "\n"
        return code

    def insert_section_code(self, code, name, new_section_code, tab_lvl):
        begin_index, end_index = self.find_section_index(code, name)
        new_code = code[:begin_index] + new_section_code + code[(end_index - tab_lvl * len(self._tab_string)) :]
        return new_code

    def update_section(self, code, name, tab_lvl=0, member_name=None):
        new_section_code = self.get_section_code(name, tab_lvl, False, member_name=member_name)
        new_code = self.insert_section_code(code, name, new_section_code, tab_lvl)
        return new_code

    def update_custom_parameter_section(self, code):
        new_section_code = self.get_custom_parameter_code(outer_comments=False)
        new_code = self.insert_section_code(code, self._custom_parameter_name, new_section_code, 0)
        return new_code

    def update_data_set(self, code, index, tab_lvl=0):
        new_code = self.insert_section_code(
            code, self.get_data_set_name(index), self.get_data_set_code(index, tab_lvl, False), tab_lvl
        )
        return new_code

    def remove_section(self, code, name):
        begin_string = self._begin_string % name
        end_string = self._end_string % name
        new_code = ""
        keeplines = True
        for line in code.splitlines(True):
            if begin_string in line:
                keeplines = False
            elif end_string in line:
                keeplines = True
            elif keeplines:
                new_code += line
        return new_code

    def get_number_data_sets_in_code(self, code):
        missing_data_section = False
        i = 0
        while not missing_data_section:
            try:
                self.find_section(self.get_data_set_name(i), code)
            except ValueError:
                missing_data_section = True
            else:
                i += 1
        return i

    def get_preamble(self):
        return self.preable + "\n\n"

    def get_sim_code(self):
        code = self._sim_def_string
        for i, _data_set in enumerate(self.data_sections_interactors):
            code += self.get_data_set_code(i, tab_lvl=1)
        code += self._sim_end_string
        return code

    def get_custom_parameter_code(self, outer_comments=True):
        tab_lvl = 0
        code = ""
        if outer_comments:
            code += tab_lvl * self._tab_string + self._begin_string % self._custom_parameter_name + "\n"
        code += "cp = UserVars()\n"
        for cust_par in self.custom_parameters_interactors:
            code += cust_par.get_code()
            code += "\n"
        if outer_comments:
            code += tab_lvl * self._tab_string + self._end_string % self._custom_parameter_name + "\n"
        return code

    def get_code(self):
        code = self.get_preamble()
        for name in self.code_sections:
            code += self.get_section_code(name)
            code += "\n"
        code += self.get_custom_parameter_code()
        code += "\n"
        code += self.get_sim_code()
        return code

    def update_code(self, code):
        new_code = code[:]
        for name in self.code_sections:
            new_code = self.update_section(new_code, name)
        new_code = self.update_custom_parameter_section(new_code)
        data_sets_in_code = self.get_number_data_sets_in_code(code)
        data_sets_to_update = min(data_sets_in_code, len(self.data_sections_interactors))
        for i in range(0, data_sets_to_update):
            new_code = self.update_data_set(new_code, i, tab_lvl=1)
        if data_sets_in_code > len(self.data_sections_interactors):
            for i in range(len(self.data_sections_interactors), data_sets_in_code):
                new_code = self.remove_section(new_code, self.get_data_set_name(i))
        elif data_sets_in_code < len(self.data_sections_interactors):
            for i in range(data_sets_in_code, len(self.data_sections_interactors)):
                insertion_index = self.find_new_data_set_insertion_index(new_code)
                new_code = (
                    new_code[:insertion_index]
                    + self.get_data_set_code(i, tab_lvl=1)
                    + new_code[insertion_index:]
                )
        return new_code


class ScriptInteractor:
    def create_new(self):
        return self.__class__()

    def copy(self):
        cpy = self.create_new()
        cpy.parse_code(self.get_code())
        return cpy

    def get_multiline(self):
        return False

    def get_code(self):
        raise NotImplementedError()

    def parse_code(self, code):
        raise NotImplementedError()

    def parse_list(self, code):
        code = code.strip()
        if not (code[0] == "[" and code[-1] == "]"):
            raise ValueError("Could not parse list: " + code)
        return [item.strip() for item in code[1:-1].split(",")]


class ObjectScriptInteractor(ScriptInteractor):
    def __init__(self, class_name=None, class_impl=None):
        self.class_name = class_name
        self.class_impl = class_impl
        self.parameters = {}
        self.name = ""

    def get_sim_methods_info(self):
        if not self.class_impl:
            return []
        sim_methods = []
        for name, method in inspect.getmembers(self.class_impl, predicate=inspect.isfunction):
            if name.startswith("Sim"):
                sig = inspect.signature(method)
                args = [p.name for p in sig.parameters.values()][1:]
                sim_methods.append(SimMethodInfo(name=name[3:], args=args, def_args=args[:]))
        return sim_methods

    def parse_parameter_string(self, code):
        values = {}
        for item in utils.parse_parameters(code):
            values[item[0]] = item[1]
        return values

    def get_parameter_code(self):
        code = ""
        for name, val in self.parameters.items():
            code += "%s=%s, " % (name, val)
        return code[:-2] if code.endswith(", ") else code

    def parse_code(self, code):
        if "=" not in code:
            return
        name, code = code.split("=", 1)
        self.name = name.strip()
        code = code[code.index("(") + 1 : code.rindex(")")]
        self.parameters = self.parse_parameter_string(code)

    def get_code(self):
        return "%s = %s(%s)" % (self.name, self.class_name, self.get_parameter_code())


class ParameterExpressionInteractor(ScriptInteractor):
    def __init__(self, obj_name="", obj_method="", expression=""):
        self.obj_name = obj_name
        self.obj_method = obj_method
        self.expression = expression

    def parse_code(self, code):
        self.obj_name = code[: code.index(".")].strip()
        self.obj_method = code[code.index(".") + 1 : code.index("(")].strip()
        self.expression = code[code.index("(") + 1 : code.rindex(")")]

    def get_code(self):
        return "%s.%s(%s)" % (self.obj_name, self.obj_method, self.expression)


class DataSetSimulationInteractor(ScriptInteractor):
    def __init__(self):
        self.obj_name = ""
        self.obj_method = ""
        self.instrument = "inst"
        self.arguments = []
        self.expression_list = []
        self.position = 0

    def set_pos(self, pos):
        self.position = pos

    def set_from_sim_method(self, name, sim_method_info: SimMethodInfo):
        self.obj_name = name
        self.obj_method = sim_method_info.name
        self.arguments = sim_method_info.def_args[:]

    def parse_code(self, code):
        self.expression_list = []
        for line in code.splitlines():
            if line.strip().startswith("#"):
                continue
            if "I.append" in line:
                sim_call = line[line.index("I.append") + len("I.append(") : line.rindex(")")]
                obj_call, self.instrument = sim_call.split(",")
                self.instrument = self.instrument.strip()
                self.obj_name = obj_call[: obj_call.index(".")].strip()
                self.obj_method = obj_call[obj_call.index(".") + 1 : obj_call.index("(")].strip()
                args = obj_call[obj_call.index("(") + 1 : obj_call.rindex(")")]
                self.arguments = [item.strip() for item in args.split(",") if item.strip()]
            elif line.strip().startswith("d ="):
                continue
            elif line.strip():
                self.expression_list.append(line.strip())

    def get_code(self):
        code = ""
        for exp in self.expression_list:
            code += exp + "\n"
        code += "d = data[%d]\n" % self.position
        args = ", ".join(self.arguments)
        code += "I.append(%s.%s(%s, %s))\n" % (self.obj_name, self.obj_method, args, self.instrument)
        return code


class CustomParameterInteractor(ScriptInteractor):
    def __init__(self, name="", value=""):
        self.name = name
        self.value = value

    def parse_code(self, code):
        if ".new_var" in code:
            name, rest = code.split("(", 1)
            args = rest.strip().strip(")").split(",", 1)
            self.name = args[0].strip().strip("'").strip('"')
            self.value = args[1].strip()
        elif ".new_sys_err" in code:
            name, rest = code.split("(", 1)
            args = rest.strip().strip(")").split(",", 2)
            self.name = args[0].strip().strip("'").strip('"')
            self.value = args[1].strip()

    def get_code(self):
        return "cp.new_var('%s', %s)" % (self.name, self.value)
