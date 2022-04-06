"""
Messaging protocol for remote model refinement with GenX server.
"""

import zlib
import struct
import asyncio
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from logging import debug
from pickle import dumps, loads
from typing import Union

from ..diffev import DiffEvConfig
from ..model import Model
from ..solver_basis import GenxOptimizer, SolverParameterInfo, SolverResultInfo, SolverUpdateInfo


header_types = {}


@dataclass
class GenXMessage(ABC):
    """
    Base class for all messages. Does handle reading the
    header strings and unpacking the objects from the message body.

    The message classes are data classes with arbitray information
    as fields that all get send. Any other methods may be provided
    to work with the data.

    The Constructed message has the following form:
    ## Header Length ## (4 bytes, single unsigned integer)
    ## Header Type ## (4 bytes, single unsigned integer) Defines the message class for the header
    ## Header Data ## (pickled header with length {Header Length}-24)
    ## Data Length ## (8 bytes, single unsigned long long)
    ## Data ## (bz2 compressed pickled class with data)
    """
    HEADER_TYPE = 0

    @classmethod
    def use_header(cls, header_data):
        # has to be overwritten if HEADER_TYPE !+ 0
        pass

    def make_header(self) -> bytes:
        """
        Messages do not have to provide a header but it may be
        usefull if information about the data should be used
        while it is transmitted.
        """
        return b''

    def make_data(self):
        return dumps(self)

    def message(self):
        header = self.make_header()
        data = self.make_data()
        data = zlib.compress(data, 1)
        message = struct.pack("I", len(header)+4)
        message += struct.pack("I", self.HEADER_TYPE)
        message += header
        message += struct.pack("I", len(data))
        message += data
        return message

    @staticmethod
    async def receive(io: asyncio.StreamReader) -> "GenXMessage":
        debug('GenXMessage receiving')
        header_length = struct.unpack("I", await io.read(4))[0]
        header_type = struct.unpack("I", await io.read(4))[0]
        debug(f'GenXMessage header_length={header_length} ; header_type={header_type}')
        if header_type!=0:
            header_string = await io.read(header_length)
            header_data = loads(header_string)
            header_types[header_type].use_header(header_data)
        data_length = struct.unpack("I", await io.read(4))[0]
        debug(f'GenXMessage data_length={data_length}')
        data_string = await io.read(data_length)
        debug(f'Length of data_string received: {len(data_string)}')
        while len(data_string)<data_length:
            data_string += await io.read(data_length-len(data_string))
        data_string = zlib.decompress(data_string)
        res = loads(data_string)
        debug(f'GenXMessage of type {type(res)} sucessfully unpacked')
        return res


@dataclass
class StingMessage(GenXMessage):
    text: str


@dataclass
class EchoMessage(GenXMessage):
    # A string message that is send back by the server.
    text: str


class ActionType(int, Enum):
    START_FIT = 1
    STOP_FIT = 2


@dataclass
class ActionMessage(GenXMessage):
    action_type: ActionType
    short_info: str
    description: str


@dataclass
class ModelTransfer(GenXMessage):
    model: Model
    fitparams: DiffEvConfig


@dataclass
class OptimizerUpdate(GenXMessage):
    payload: Union[SolverUpdateInfo, SolverParameterInfo, SolverResultInfo]

    def __repr__(self):
        return f'OptimizerUpdate(payload={self.payload.__class__.__name__})'
