"""
The model controller and socket server for remote optimization.
"""

import asyncio
import struct
from logging import debug, info, warning
from hashlib import blake2b

from . import messaging
from ..core import AUTH_SIZE, HANDSHAKE1, HANDSHAKE2
from ..model_control import ModelController
from ..diffev import DiffEv
from ..solver_basis import GenxOptimizerCallback, SolverParameterInfo, SolverResultInfo, SolverUpdateInfo


try:
    from mpi4py import MPI
except ImportError:
    rank = 0
    __mpi__ = False
else:
    __mpi__ = bool(MPI.COMM_WORLD.Get_size()>1)
    rank = MPI.COMM_WORLD.Get_rank()
    comm = MPI.COMM_WORLD


class RemotCallback(GenxOptimizerCallback):
    loop: asyncio.AbstractEventLoop = None

    def __init__(self, parent):
        GenxOptimizerCallback.__init__(self)
        self.parent = parent

    async def send_message(self, message: messaging.GenXMessage):
        self.parent.writer.write(message.message())
        await self.parent.writer.drain()

    def _send_message(self, message: messaging.GenXMessage):
        if self.loop is None:
            raise RuntimeError('Could not send message, no asyncio event loop defined')
        else:
            debug(f'sending message {message}')
            asyncio.run_coroutine_threadsafe(self.send_message(message), self.loop)

    def _end_fit(self):
        if self.loop is None:
            raise RuntimeError('Could not send message, no asyncio event loop defined')
        else:
            debug(f'cleanup fit locally')
            asyncio.run_coroutine_threadsafe(self.parent.cleanup(), self.loop)

    def text_output(self, text):
        msg = messaging.StingMessage(text)
        self._send_message(msg)

    def plot_output(self, update_data: SolverUpdateInfo):
        msg = messaging.OptimizerUpdate(update_data)
        self._send_message(msg)

    def parameter_output(self, param_info: SolverParameterInfo):
        msg = messaging.OptimizerUpdate(param_info)
        self._send_message(msg)

    def fitting_ended(self, result_data: SolverResultInfo):
        msg = messaging.OptimizerUpdate(result_data)
        self._send_message(msg)
        self._end_fit()

    def autosave(self):
        pass


class RemoteController(ModelController):
    """
    A model controller to execute on a server. Use RemoteController.serve with asyncio.
    """
    reader = None
    writer = None
    cleanup_phase = False

    key = b'empty'

    def __init__(self):
        ModelController.__init__(self, DiffEv())
        self.lock = None
        if rank==0:
            self.callbacks = RemotCallback(self)
            self.set_callbacks(self.callbacks)

    async def serve(self, address, port):
        self.lock = asyncio.locks.Lock()

        if rank==0:
            server = await asyncio.start_server(
                self.handle_connection, address, port)

            async with server:
                info(f"Starting listening on {address} with port={port}")
                await server.serve_forever()
        else:
            debug(f'Starting MPI client process for server on {address} with port={port}')
            await self.recive_mpi()

    async def handle_connection(self, reader, writer):
        debug('New connection')
        async with self.lock:
            debug('Connection setup and handshake')
            self.reader = reader
            self.writer = writer
            await self.handshake()
            self.cleanup_phase = False
            while self.reader is not None and not self.cleanup_phase:
                await self.recv_messages()

    async def handshake(self):
        key = self.key
        ref1 = blake2b(HANDSHAKE1, key=key, digest_size=AUTH_SIZE).hexdigest().encode('ascii')
        ref2 = blake2b(HANDSHAKE2, key=key, digest_size=AUTH_SIZE).hexdigest().encode('ascii')
        res = await self.reader.read(len(ref1))
        if res==ref1:
            debug('Incoming message correct, sending response.')
            self.writer.write(ref2)
            await self.writer.drain()
        else:
            debug("Handshake failed")
            self.writer.close()
            await self.writer.wait_closed()
            debug("Connection closed")
            self.reader = None
            self.writer = None

    async def recv_messages(self):
        try:
            res = await messaging.GenXMessage.receive(self.reader)
        except (ConnectionResetError, struct.error):
            if self.cleanup_phase:
                return
            else:
                warning("Connection was reset")
                await self.cleanup()
                return
        if isinstance(res, messaging.StingMessage):
            info(f"Received text message: {res.text}")
        elif isinstance(res, messaging.ActionMessage):
            if res.action_type is messaging.ActionType.START_FIT:
                info('Start fit was triggered')
                self.callbacks.loop = asyncio.get_running_loop()
                if self.optimizer.opt.use_mpi:
                    if __mpi__:
                        # server is just run on rank=0 so send model to clients
                        self.optimizer.opt.use_parallel_processing = False
                        self.optimizer.WriteConfig()
                        comm.Barrier()
                        comm.bcast(self.model, root=0)
                        comm.bcast(self.optimizer.opt, root=0)
                    else:
                        warning(
                            'Fit was started with use_mpi option selected, but server is started without MPI support')
                        self.optimizer.opt.use_mpi = False
                        self.optimizer.WriteConfig()
                self.StartFit()
            elif res.action_type is messaging.ActionType.STOP_FIT:
                info('Stop fit was triggered')
                await self.cleanup()
                self.callbacks.loop = None
            else:
                warning(f'Action not implemented {res!r}')
                return
        elif isinstance(res, messaging.ModelTransfer):
            info('Setting a new model')
            self.model = res.model
            self.optimizer.opt = res.fitparams
            self.optimizer.WriteConfig()
        elif isinstance(res, messaging.EchoMessage):
            info(f'Echoing message "{res.text}"')
            msg = messaging.StingMessage(res.text)
            await self.send_message(msg)

    async def send_message(self, message: messaging.GenXMessage):
        debug(f'Sending message {message}')
        self.writer.write(message.message())
        await self.writer.drain()

    async def recive_mpi(self):
        while True:
            comm.Barrier()
            self.model = comm.bcast(self.model, root=0)
            self.optimizer.opt = comm.bcast(self.optimizer.opt, root=0)
            self.optimizer.WriteConfig()
            self.StartFit()
            while self.optimizer.is_running():
                await asyncio.sleep(0.1)

    async def cleanup(self):
        if self.cleanup_phase:
            return
        debug("Starting cleanup sequence")
        self.cleanup_phase = True
        if self.optimizer.is_running():
            self.StopFit()
        while self.optimizer.is_running():
            await asyncio.sleep(0.1)
        debug("Closing connection")
        self.writer.close()
        await self.writer.wait_closed()
        debug("Connection closed")
        self.reader = None
        self.writer = None
