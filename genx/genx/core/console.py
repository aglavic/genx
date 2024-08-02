"""
Classes for handling of messages with console.

If available (UNIX) uses the curses library for a pseudo-interactive interface.
"""

import atexit
import logging

from numpy import arange, array

from ..model_control import ModelController
from ..solver_basis import GenxOptimizer, GenxOptimizerCallback
from . import config as io
from .custom_logging import iprint


class CursesHandler(logging.Handler):
    message_history = []
    max_messages = 8

    def __init__(self, optimizer: GenxOptimizer, *args, **kwargs):
        logging.Handler.__init__(self, *args, **kwargs)
        self.opt = optimizer
        import curses

        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.clear()

        _, width = self.stdscr.getmaxyx()
        self.stdscr.addstr(9, width // 2 - 22, "Relative value and spread of fit parameters:")
        self.stdscr.addstr(9, width - 11, "best/width")
        self.stdscr.nodelay(True)
        self.stdscr.refresh()

        rl = logging.getLogger()
        rl.handlers[0].setLevel(logging.ERROR + 1)

        self.setLevel(logging.INFO)
        rl.addHandler(self)

        atexit.register(self.reset_console)

    def emit(self, record):
        try:
            msg = self.format(record)
            self.message_history += msg.strip().splitlines()
            lkey = self.stdscr.getch()
            if lkey == ord("q"):
                self.message_history.append('"q" pressed, trying to stop fit.')
                self.opt.stop = True
            self.message_history = self.message_history[-self.max_messages :]
            for i, msg in enumerate(self.message_history):
                _, width = self.stdscr.getmaxyx()
                self.stdscr.addstr(i, 4, msg + " " * (width - 4 - len(msg)))
            self.stdscr.refresh()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def reset_console(self):
        import curses

        self.setLevel(logging.ERROR + 1)
        curses.nocbreak()
        curses.echo()
        curses.endwin()
        rl = logging.getLogger()
        rl.handlers[0].setLevel(logging.INFO)
        print("\n".join(self.message_history))


def calc_errorbars(mod, opt: GenxOptimizer):
    error_values = []
    n_elements = len(opt.start_guess)
    for index in range(n_elements):
        # calculate the error for given optimizer, accuracy depends on settings
        (error_low, error_high) = opt.calc_error_bar(index)
        error_values.append("(%.3e, %.3e,)" % (error_low, error_high))
    mod.parameters.set_error_pars(error_values)


class ConsoleCallback(GenxOptimizerCallback):

    def __init__(self, stdscr, ctrl: ModelController, error=False, outfile=None):
        self.stdscr = stdscr
        self.ctrl = ctrl
        self.error = error
        self.outfile = outfile

    def autosave(self):
        # print 'Updating the parameters'
        model = self.ctrl.get_model()
        model.parameters.set_value_pars(self.ctrl.optimizer.best_vec)
        if self.error:
            iprint("Calculating error bars")
            calc_errorbars(self.ctrl.model, self.ctrl.optimizer)
        if self.outfile:
            iprint("Saving to %s" % self.outfile)
            self.ctrl.save_file(self.outfile)

    def parameter_output(self, param_info):
        if self.stdscr:
            height, width = self.stdscr.getmaxyx()
            full_width = width - 16 - 12
            pwidth = param_info.max_val - param_info.min_val
            pres = (param_info.values - param_info.min_val) / pwidth
            population = array(param_info.population)
            pmin = (population.min(axis=0) - param_info.min_val) / pwidth
            pmax = (population.max(axis=0) - param_info.min_val) / pwidth
            pwidth = pmax - pmin
            maxpar = height - 10
            if maxpar < param_info.values.shape[0]:
                order = pwidth.argsort()[::-1]
            else:
                order = arange(param_info.values.shape[0])

            for i in range(min(param_info.values.shape[0], maxpar)):
                param_id = order[i]
                self.stdscr.hline(10 + i, 16, " ", full_width)
                self.stdscr.addstr(10 + i, 1, "Parameter %02i: [" % param_id)
                self.stdscr.addstr(10 + i, width - 12, "] %.2f/%.2f" % (pres[param_id], pwidth[param_id]))

                ppos = max(0.0, min(1.0, pres[param_id]))
                wstart = max(0.0, min(1.0, pmin[param_id]))
                wwidth = max(0.0, min(1.0 - wstart, pwidth[param_id]))
                self.stdscr.hline(10 + i, 16 + int(full_width * wstart), "=", int(full_width * wwidth))
                self.stdscr.addstr(10 + i, 16 + int(full_width * ppos), "#")

            self.stdscr.refresh()

    def text_output(self, text):
        iprint(text, flush=True)

    def plot_output(self, update_data):
        pass

    def fitting_ended(self, result_data):
        pass


def setup_console(ctrl: ModelController, error=False, outfile=None, use_curses=True):
    if use_curses:
        try:
            import curses
        except ImportError:
            stdscr = None
        else:
            chandler = CursesHandler(ctrl.optimizer)
            stdscr = chandler.stdscr
    else:
        stdscr = None

    cb = ConsoleCallback(stdscr, ctrl, error, outfile)
    ctrl.set_callbacks(cb)
