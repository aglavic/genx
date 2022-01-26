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

    args=parser.parse_args()

    if not args.debug:
        custom_logging.CONSOLE_LEVEL=INFO
    custom_logging.setup_system()

    if args.logfile:
        custom_logging.activate_logging(args.logfile)
    debug("Arguments from parser: %s"%args)

    try:
        set_numba_single()
    except ImportError:
        pass

    if rank==0:
        info('Starting RemoteController')
        from .remote import messaging, controller
        ctrl=controller.RemoteController()
        asyncio.run(ctrl.serve(args.address, args.port))

if __name__ == '__main__':
    main()
