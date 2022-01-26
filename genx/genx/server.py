"""
Executable module to start a server that can be used for
remote refinement on a server.
"""

import asyncio
import argparse
from logging import info, debug, INFO

from . import version
from .core import custom_logging

def set_numba_single():
    debug('Setting numba JIT compilation to single CPU')
    import numba
    old_jit = numba.jit
    def jit(*args, **opts):
        opts['parallel']=False
        opts['cache']=False
        return old_jit(*args, **opts)
    numba.jit = jit
    try:
        numba.set_num_threads(1)
    except AttributeError:
        pass

def main():
    try:
        from mpi4py import MPI
    except ImportError:
        pass
    else:
        __mpi__ = True
        rank = MPI.COMM_WORLD.Get_rank()

    parser=argparse.ArgumentParser(description="GenX %s, fits data to a model."%version.__version__,
                                   epilog="For support, manuals and bug reporting see http://genx.sf.net"
                                   )
    parser.add_argument('address', nargs='?', default='localhost', help='Network address to listen on')
    parser.add_argument('port', nargs='?', default=3000, help='Network port to listen on')
    parser.add_argument('-l', '--logfile', dest='logfile', default=None, type=str,
                            help='Output debug information to logfile.')
    parser.add_argument('--debug', dest='debug', default=False, action="store_true",
                            help='Show additional debug information on console/logfile')
    parser.add_argument('--disable-nb', dest='disable_numba', default=False, action="store_true",
                            help='Disable the use of numba JIT compiler')
    parser.add_argument('--nb1', dest='numba_single', default=False, action="store_true",
                            help='Compile numba JIT functions without parallel computing support (use one core only). '
                                 'This does disable caching to prevent parallel versions from being loaded.')

    args=parser.parse_args()

    if not args.debug:
        custom_logging.CONSOLE_LEVEL=INFO
    custom_logging.setup_system()

    if args.logfile:
        custom_logging.activate_logging(args.logfile)
    debug("Arguments from parser: %s"%args)

    if parser.disable_numba:
        from genx.models import lib as modellib
        modellib.USE_NUMBA=False
    else:
        if parser.numba_single:
            try:
                set_numba_single()
            except ImportError:
                pass
        info('Importing numba based modules to pre-compile JIT functions, this can take some time')
        from genx.models.lib import paratt_numba, neutron_numba, instrument_numba, offspec, surface_scattering
        info('Modules imported successfully')

    if rank==0:
        info('Starting RemoteController')
        from .remote import messaging, controller
        ctrl=controller.RemoteController()
        asyncio.run(ctrl.serve(args.address, args.port))

if __name__ == '__main__':
    main()
