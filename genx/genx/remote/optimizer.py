"""
A remote client that performs model refinement on a mashine running genx.server.

Currently, only differential evolution is supported.
"""

import asyncio
from dataclasses import dataclass
from hashlib import blake2b
from logging import debug

from numpy import array

from . import messaging
from ..core import AUTH_SIZE, HANDSHAKE1, HANDSHAKE2
from ..core.config import BaseConfig
from ..model import Model
from ..solver_basis import GenxOptimizer, GenxOptimizerCallback, SolverResultInfo
from ..diffev import DiffEvConfig, DiffEvDefaultCallbacks


@dataclass
class RemoteDiffEvConfig(BaseConfig):
    section = 'solver'

    address: str = 'localhost'
    port: int = 3000
    key: str = 'empty'

    km: float = BaseConfig.GParam(0.7, pmin=0.0, pmax=1.0)
    kr: float = BaseConfig.GParam(0.7, pmin=0.0, pmax=1.0)
    allowed_fom_discrepancy: float = 1e-10

    use_pop_mult: bool = True
    pop_size: int = BaseConfig.GParam(50, pmin=5, pmax=10000, label='Fixed size')
    pop_mult: int = BaseConfig.GParam(3, pmin=1, pmax=100, label='Relative size')
    create_trial: str = BaseConfig.GChoice('best_1_bin', ['best_1_bin', 'rand_1_bin', 'best_either_or',
                                                          'rand_either_or', 'jade_best', 'simplex_best_1_bin'],
                                           label='Method')

    use_max_generations: bool = False
    max_generations: int = BaseConfig.GParam(500, pmin=10, pmax=10000, label='Fixed size')
    max_generation_mult: int = BaseConfig.GParam(6, pmin=1, pmax=100, label='Relative size')
    min_parameter_spread: float = BaseConfig.GParam(0.0, pmin=0.0, pmax=100.0, label='parameter spread to stop (%)')

    use_start_guess: bool = True
    use_boundaries: bool = True

    max_log_elements: int = BaseConfig.GParam(100000, pmin=1000, pmax=1000000, label=', # elements')
    use_mpi: bool = False
    use_parallel_processing: bool = False
    parallel_processes: int = BaseConfig.GParam(16, pmin=2, pmax=1000, label='# processes')
    parallel_chunksize: int = BaseConfig.GParam(10, pmin=1, pmax=1000, label='items/chunk')

    use_autosave: bool = False
    autosave_interval: int = BaseConfig.GParam(10, pmin=1, pmax=1000, label=', interval')

    save_all_evals: bool = False
    errorbar_level: float = BaseConfig.GParam(1.05, pmin=1.001, pmax=2.0)

    groups = {  # for building config dialogs
        'Server': ['address', 'port', 'key'],
        'Fitting': [['use_start_guess', 'use_boundaries'], ['use_autosave', 'autosave_interval'],
                    ['save_all_evals', 'max_log_elements']],
        'Differential Evolution':
            ['km', 'kr', 'create_trial',
             ['Population size:', 'use_pop_mult', 'pop_mult', 'pop_size'],
             ['Max. Generations:', 'use_max_generations', 'max_generations', 'max_generation_mult'],
             'min_parameter_spread',
             ],
        'Parallel processing': ['use_mpi', 'use_parallel_processing', 'parallel_processes', 'parallel_chunksize']
        }


class RemoteOptimizer(GenxOptimizer):
    opt: RemoteDiffEvConfig
    n_fom_evals = 0

    _callbacks: GenxOptimizerCallback = DiffEvDefaultCallbacks()

    def __init__(self):
        GenxOptimizer.__init__(self)

        # Control flags:
        self.running = False  # true if optimization is running
        self.stop = False  # true if the optimization should stop
        self.setup_ok = False  # True if the optimization have been setup
        self.error = None  # None/string if an error ahs occurred

        # Logging variables
        self.fom_log = array([[0, 0]])[0:0]

        self.start_guess = array([])
        self.updated_kr = []
        self.updated_km = []

        self.reader = None
        self.writer = None

    def pickle_string(self, clear_evals: bool = False):
        pass

    def pickle_load(self, pickled_string: bytes):
        pass

    def is_running(self) -> bool:
        return self.running

    def get_start_guess(self):
        pass

    def get_model(self) -> Model:
        pass

    def get_fom_log(self):
        pass

    def calc_error_bar(self, index: int) -> (float, float):
        pass

    def project_evals(self, index: int):
        pass

    def start_fit(self, model: Model):
        '''
        Starts fitting on a remote server.
        '''
        if not self.running:
            self.stop = False
            self.text_output('Trying to connect to server...')
            asyncio.run(self.connect())
            asyncio.run(self.send_message(self.model_message(model)))
            self.text_output('Starting the fit...')
            asyncio.run(self.send_message(messaging.ActionMessage(messaging.ActionType.START_FIT, 'Start fit', '')))
            self.running = True
            self._recv_task = asyncio.create_task(self.receive())
            return True
        else:
            self.text_output('Fit is already running, stop and then start')
            return False

    def model_message(self, model: Model) -> messaging.ModelTransfer:
        opt = DiffEvConfig()
        for key in opt.asdict():
            # transfer all options to send config
            setattr(opt, key, getattr(self.opt, key))
        mm = messaging.ModelTransfer(model, opt)
        return mm

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.opt.address, self.opt.port)
        key = b'empty'
        ref1 = blake2b(HANDSHAKE1, key=key, digest_size=AUTH_SIZE).hexdigest().encode('ascii')
        ref2 = blake2b(HANDSHAKE2, key=key, digest_size=AUTH_SIZE).hexdigest().encode('ascii')
        self.writer.write(ref1)
        await self.writer.drain()
        fb = await self.reader.read(len(ref2))
        if ref2 != fb:
            self.writer.close()
            await self.writer.wait_closed()
            self.reader = None
            self.writer = None

    async def send_message(self, message: messaging.GenXMessage):
        debug(f'send_message {message}')
        self.writer.write(message.message())
        await self.writer.drain()

    async def receive(self):
        while self.running:
            obj = await messaging.GenXMessage.receive(self.reader)
            print(repr(obj))

    def stop_fit(self):
        self.text_output('trying to stop the fit')
        asyncio.run(self.send_message(messaging.ActionMessage(messaging.ActionType.STOP_FIT, 'Stop fit', '')))

    def resume_fit(self, model: Model):
        pass

    def is_fitted(self):
        pass

    def is_configured(self):
        pass

    def set_callbacks(self, callbacks: GenxOptimizerCallback):
        self._callbacks = callbacks

    def get_callbacks(self) -> GenxOptimizerCallback:
        return self._callbacks

    def get_result_info(self) -> SolverResultInfo:
        pass

    def text_output(self, text: str):
        self._callbacks.text_output(text)
