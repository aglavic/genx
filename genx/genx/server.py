"""
Executable module to start a server that can be used for
remote refinement on a server.
"""
import os
import asyncio
import argparse
import appdirs
import logging

from . import version
from .core import custom_logging


def set_numba_single():
    config_path = os.path.abspath(appdirs.user_data_dir('GenX3', 'ArturGlavic'))
    cache_dir = os.path.join(config_path, 'single_cpu_numba_cache')

    logging.debug('Setting numba JIT compilation to single CPU')
    import numba
    numba.config.CACHE_DIR = cache_dir
    old_jit = numba.jit

    def jit(*args, **opts):
        opts['parallel'] = False
        return old_jit(*args, **opts)

    numba.jit = jit
    numba.GENX_OVERWRITE_SINGLE = True
    try:
        numba.set_num_threads(1)
    except AttributeError:
        pass


def main():
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    try:
        from mpi4py import MPI
    except ImportError:
        rank = 0
        __mpi__ = False
    else:
        __mpi__ = bool(MPI.COMM_WORLD.Get_size()>1)
        rank = MPI.COMM_WORLD.Get_rank()

    parser = argparse.ArgumentParser(description="GenX %s, fits data to a model. Server script for remote fitting."%version.__version__,
                                     epilog="For support, manuals and bug reporting see http://genx.sf.net"
                                     )
    parser.add_argument('address', nargs='?', default='localhost', help='Network address to listen on')
    parser.add_argument('port', nargs='?', default=3000, help='Network port to listen on')
    parser.add_argument('-l', '--logfile', dest='logfile', default=None, type=str,
                        help='Output debug information to logfile.')
    parser.add_argument('-p', '--password', dest='password', default='empty', type=str,
                        help='Password used when establishing connection from client.')
    parser.add_argument('--debug', dest='debug', default=False, action="store_true",
                        help='Show additional debug information on console/logfile')
    parser.add_argument('--disable-nb', dest='disable_numba', default=False, action="store_true",
                        help='Disable the use of numba JIT compiler')
    parser.add_argument('--no-nb1', dest='numba_single', default=True, action="store_false",
                        help='Compile numba JIT functions with parallel computing support (use more then one core). '
                             'This is the default for no-server based fits.')

    args = parser.parse_args()

    if not args.debug:
        custom_logging.CONSOLE_LEVEL = logging.INFO
    if rank>0:
        custom_logging.CONSOLE_LEVEL = logging.WARNING
    custom_logging.setup_system()

    if args.logfile:
        if __mpi__:
            lfbase, lfend = args.logfile.rsplit('.', 1)
            custom_logging.activate_logging(f'{lfbase}_{rank:02}.{lfend}')
        else:
            custom_logging.activate_logging(args.logfile)
    if rank==0:
        logging.debug("Arguments from parser: %s"%args)

    if args.disable_numba:
        from genx.models import lib as modellib
        modellib.USE_NUMBA = False
    else:
        if args.numba_single:
            try:
                set_numba_single()
            except ImportError:
                pass
        if rank==0:
            logging.info('Importing numba based modules to pre-compile JIT functions, this can take some time')
        from genx.models.lib import paratt_numba, neutron_numba, instrument_numba, offspec, surface_scattering
        if rank==0:
            logging.info('Modules imported successfully')

    if rank==0:
        logging.info('Starting RemoteController')
    from .remote import controller
    ctrl = controller.RemoteController()
    ctrl.key = args.password.encode('utf-8')
    asyncio.run(ctrl.serve(args.address, args.port))


if __name__=='__main__':
    main()
