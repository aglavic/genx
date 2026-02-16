"""
Single-instance socket helpers for the GenX GUI.
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Callable, List

import wx


class SingleInstanceServer(threading.Thread):
    """
    Background server thread listening for single-instance client messages.
    """

    HANDSHAKE_TOKEN = "GENX_SINGLE_INSTANCE_HELLO"
    _RESPONSE_OK = "OK"
    _RESPONSE_ACK = "ACK"
    _RESPONSE_ERR = "ERR"

    def __init__(
        self,
        on_message: Callable[[List[str]], None] | None = None,
        host: str = "127.0.0.1",
        port: int = 37983,
        timeout: float = 0.5,
    ) -> None:
        super().__init__(name="GenXSingleInstanceServer", daemon=True)
        self.on_message = on_message
        self.host = host
        self.port = port
        self.timeout = timeout
        self._stop_event = threading.Event()
        self._socket: socket.socket | None = None

    def stop(self) -> None:
        self._stop_event.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        self.join(1.0)

    def run(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket = server
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(self.timeout)

        while not self._stop_event.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                self._handle_client(conn)

        try:
            server.close()
        except OSError:
            pass

    def _handle_client(self, conn: socket.socket) -> None:
        conn.settimeout(5.0)
        handshake = self._recv_line(conn)
        if handshake != self.HANDSHAKE_TOKEN:
            self._send_line(conn, self._RESPONSE_ERR)
            return
        self._send_line(conn, self._RESPONSE_OK)

        payload = self._recv_line(conn)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            self._send_line(conn, self._RESPONSE_ERR)
            return

        if not isinstance(data, dict):
            self._send_line(conn, self._RESPONSE_ERR)
            return

        messages = data.get("messages")
        if not isinstance(messages, list) or not all(isinstance(item, str) for item in messages):
            self._send_line(conn, self._RESPONSE_ERR)
            return

        if self.on_message is not None:
            wx.CallAfter(self.on_message, messages)

        self._send_line(conn, self._RESPONSE_ACK)

    @staticmethod
    def _recv_line(conn: socket.socket) -> str:
        data = bytearray()
        while True:
            chunk = conn.recv(1)
            if not chunk:
                break
            if chunk == b"\n":
                break
            data.extend(chunk)
        return data.decode("utf-8")

    @staticmethod
    def _send_line(conn: socket.socket, message: str) -> None:
        conn.sendall(message.encode("utf-8") + b"\n")


class SingleInstanceClient:
    """
    Client for checking a running single-instance server and sending messages.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 37983, timeout: float = 1.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def is_server_running(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=0.05):
                return True
        except OSError:
            return False

    def send_message(self, messages: List[str]) -> bool:
        if not isinstance(messages, list) or not all(isinstance(item, str) for item in messages):
            raise TypeError("messages must be a list[str]")

        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as conn:
                conn.settimeout(self.timeout)

                self._send_line(conn, SingleInstanceServer.HANDSHAKE_TOKEN)
                response = self._recv_line(conn)
                if response != SingleInstanceServer._RESPONSE_OK:
                    return False

                payload = json.dumps({"messages": messages}, ensure_ascii=True)
                self._send_line(conn, payload)
                ack = self._recv_line(conn)
                return ack == SingleInstanceServer._RESPONSE_ACK
        except OSError:
            return False


    @staticmethod
    def _recv_line(conn: socket.socket) -> str:
        data = bytearray()
        while True:
            chunk = conn.recv(1)
            if not chunk:
                break
            if chunk == b"\n":
                break
            data.extend(chunk)
        return data.decode("utf-8")

    @staticmethod
    def _send_line(conn: socket.socket, message: str) -> None:
        conn.sendall(message.encode("utf-8") + b"\n")
