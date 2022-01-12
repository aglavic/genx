"""
The model controller and socket server for remote optimization.
"""

import asyncio
from logging import debug, info, warning
from hashlib import blake2b

from . import messaging
from ..core import AUTH_SIZE, HANDSHAKE1, HANDSHAKE2
from ..model_control import ModelController
from ..diffev import DiffEv
from ..solver_basis import GenxOptimizerCallback, SolverParameterInfo, SolverResultInfo, SolverUpdateInfo


class RemotCallback(GenxOptimizerCallback):

    def __init__(self, parent):
        GenxOptimizerCallback.__init__(self)
        self.parent = parent

    async def send_message(self, message: messaging.GenXMessage):
        self.parent.writer.write(message.message())
        await self.parent.writer.drain()

    def text_output(self, text):
        msg = messaging.StingMessage(text)
        asyncio.create_task(self.send_message(msg))

    def plot_output(self, update_data: SolverUpdateInfo):
        msg = messaging.OptimizerUpdate(update_data)
        asyncio.create_task(self.send_message(msg))

    def parameter_output(self, param_info: SolverParameterInfo):
        msg = messaging.OptimizerUpdate(param_info)
        asyncio.create_task(self.send_message(msg))

    def fitting_ended(self, result_data: SolverResultInfo):
        msg = messaging.OptimizerUpdate(result_data)
        asyncio.create_task(self.send_message(msg))

    def autosave(self):
        pass


class RemoteController(ModelController):
    """
    A model controller to execute on a server. Use RemoteController.serve with asyncio.
    """
    reader = None
    writer = None

    def __init__(self):
        ModelController.__init__(self, DiffEv())
        self.lock = asyncio.locks.Lock()
        self.callbacks = RemotCallback(self)
        self.set_callbacks(self.callbacks)

    async def serve(self, address, port):
        server = await asyncio.start_server(
            self.handle_connection, address, port)

        async with server:
            info(f"Starting listening on {address} with {port=}")
            await server.serve_forever()

    async def handle_connection(self, reader, writer):
        debug('New connection')
        async with self.lock:
            debug('Connection setup and handshake')
            self.reader = reader
            self.writer = writer
            await self.handshake()
            while self.reader is not None:
                await self.recv_messages()

    async def handshake(self):
        key = b'empty'
        ref1 = blake2b(HANDSHAKE1, key=key, digest_size=AUTH_SIZE).hexdigest().encode('ascii')
        ref2 = blake2b(HANDSHAKE2, key=key, digest_size=AUTH_SIZE).hexdigest().encode('ascii')
        res = await self.reader.read(len(ref1))
        if res==ref1:
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
        except ConnectionResetError:
            warning("Connection was reset")
            await self.cleanup()
            return
        if isinstance(res, messaging.StingMessage):
            info(f"Received text message: {res.text}")
        elif isinstance(res, messaging.ActionMessage):
            if res.action_type is messaging.ActionType.START_FIT:
                info('Start fit was triggered')
                self.StartFit()
            elif res.action_type is messaging.ActionType.STOP_FIT:
                info('Stop fit was triggered')
                await self.cleanup()
            else:
                warning(f'Action not implemented {res!r}')
                return
        elif isinstance(res, messaging.ModelTransfer):
            info('Setting a new model')
            self.model = res.model
            self.optimizer.opt = res.fitparams
        elif isinstance(res, messaging.EchoMessage):
            info(f'Echoing message "{res.text}"')
            msg = messaging.StingMessage(res.text)
            await self.send_message(msg)

    async def send_message(self, message: messaging.GenXMessage):
        debug(f'Sending message {message}')
        self.writer.write(message.message())
        await self.writer.drain()

    async def cleanup(self):
        debug("Starting cleanup sequence")
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
