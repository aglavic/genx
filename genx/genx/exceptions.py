"""
Module to define all user exceptions raised by GenX.
"""


class GenxError(Exception):
    """Basic GenX error class to allow filtering for all specific GenX errors."""

    pass


class GenxIOError(IOError, GenxError):
    """Error class for input output, mostly concerning files"""

    def __init__(self, error_message, file=""):
        self.error_message = error_message
        self.file = file
        text = "Input/Output error for file:\n" + self.file + "\n\n Python error:\n " + self.error_message
        IOError.__init__(self, text)


class GenxOptionError(IndexError, GenxError):
    """Error class for not finding an option section pair in the configuration"""

    def __init__(self, section, option):
        self.section = section
        self.option = option
        text = (
            "Error in trying to loacate values in GenX configuration."
            + "\nCould not locate the section: "
            + self.section
            + " or option: "
            + self.option
            + "."
        )
        IndexError.__init__(self, text)


class ErrorBarError(RuntimeError, GenxError):
    def __init__(self, error_message=None):
        text = error_message
        if error_message is None:
            text = "Could not evaluate the error bars. A fit has to be run before they can be calculated"
        RuntimeError.__init__(self, text)


class ParameterError(ValueError, GenxError):
    """Class for yielding Parameter errors"""

    def __init__(self, parameter: str, parameter_number: int, py_error: Exception = None, what: int = -1):
        """
        parameter: the name of the parameter
        parameter_number: the position of the parameter in the list
        error_mesage: pythons error from the original exception
        int: where the error occured
            -1 : undefined
             0 : an not find the parameter
             1 : can not evaluate i.e. set the parameter
             2 : value are larger than max
             3 : value are smaller than min
             4 : No parameters to fit
        """
        self.parameter = parameter
        self.parameter_number = parameter_number
        self.what = what

        text = ""
        text += "Parameter number %i, %s, " % (self.parameter_number, self.parameter)

        # Take care of the different cases
        if self.what == 0:
            text += "could not be found. Check the spelling.\n"
        elif self.what == 1:
            text += "could not be evaluated. Check the code of the function.\n"
        elif self.what == 2:
            text += "is larger than the value in the max column.\n"
        elif self.what == 3:
            text += "is smaller than the value in the min column\n"
        elif self.what == 4:
            text = (
                "There are no parameter selcted to be fitted.\n"
                + "Select the parameters you want to fit by checking the "
                + "boxes in the fit column, folder grid"
            )
        else:
            text += "yielded an undefined error. Check the Python output\n"

        if py_error is not None:
            text += "\nPython error output:\n" + str(py_error)
        ValueError.__init__(self, text)


class ModelError(RuntimeError, GenxError):
    """Class for yielding compile or evaluation errors in the model text"""

    def __init__(self, error_message, where):
        """
        error_mesage: pythons error message from the original exception
        where: integer describing where the error was raised.
                -1: undef
                 0: compile error
                 1: evaulation error
        """
        self.where = where
        text = ""
        if self.where == 0:
            text += "It was not possible to compile the model script.\n"
        elif self.where == 1:
            text += "It was not possible to evaluate the model script.\n" + "Check the Sim function.\n"
        elif self.where == -1:
            text += "Undefined error from the Model. See below.\n"
        text += "\n" + error_message
        RuntimeError.__init__(self, text)


class FomError(RuntimeError, GenxError):
    """Error class for the fom evaluation"""

    def __init__(self, error_message):
        text = "Could not evaluate the FOM function. See python output.\n" + "\n" + error_message
        RuntimeError(self, text)


class OptimizerInterrupted(RuntimeError, GenxError):
    """Error raised to stop a running refinement function from within a thread"""
