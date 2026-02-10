"""TCP transports for workers and the master router."""

from __future__ import annotations

import base64
import os
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .protocol import MsgType, dumps, loads, pack_message, unpack_header


HELLO_TAG = 100


class TransportError(RuntimeError):
    pass


@dataclass
class Message:
    src: int
    dest: int
    tag: int
    payload: object


class WorkerTransport:
    def __init__(self, sock: socket.socket, rank: int, cancel_event: threading.Event):
        self.sock = sock
        self.rank = rank
        self.cancel_event = cancel_event
        self.inbox: queue.Queue[Message] = queue.Queue()
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def _recv_loop(self):
        try:
            while True:
                header = self._recv_exact(17)
                if not header:
                    return
                length, msg_type, src, dest, tag = unpack_header(header)
                payload = self._recv_exact(length) if length else b""
                if msg_type == MsgType.DATA:
                    self.inbox.put(Message(src=src, dest=dest, tag=tag, payload=loads(payload)))
                elif msg_type == MsgType.CONTROL and tag == CANCEL_TAG:
                    self.cancel_event.set()
        except OSError:
            return

    def _recv_exact(self, length: int) -> bytes:
        buf = b""
        while len(buf) < length:
            chunk = self.sock.recv(length - len(buf))
            if not chunk:
                return b""
            buf += chunk
        return buf

    def send(self, dest: int, tag: int, obj):
        payload = dumps(obj)
        msg = pack_message(MsgType.DATA, src=self.rank, dest=dest, tag=tag, payload=payload)
        self.sock.sendall(msg)

    def recv(self, tag: Optional[int] = None, timeout: Optional[float] = None) -> Message:
        start = time.time()
        while True:
            try:
                msg = self.inbox.get(timeout=0.1)
            except queue.Empty:
                if timeout is not None and time.time() - start > timeout:
                    raise TimeoutError("recv timed out")
                continue
            if tag is None or msg.tag == tag:
                return msg
            self.inbox.put(msg)


class MasterRouter:
    def __init__(self, host: str, port: int, expected_workers: int):
        self.host = host
        self.port = port
        self.expected_workers = expected_workers
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen()
        self.actual_port = self.server.getsockname()[1]

        self.inbox: queue.Queue[Message] = queue.Queue()
        self._connections: Dict[int, socket.socket] = {}
        self._threads: list[threading.Thread] = []

    def accept_all(self, timeout_s: float):
        end = time.time() + timeout_s
        while len(self._connections) < self.expected_workers:
            remaining = max(0.1, end - time.time())
            self.server.settimeout(remaining)
            try:
                client, _addr = self.server.accept()
            except socket.timeout:
                raise TransportError("Timed out waiting for workers to connect")
            rank = self._handshake(client)
            if rank in self._connections:
                client.close()
                raise TransportError(f"Duplicate rank connected: {rank}")
            self._connections[rank] = client
            t = threading.Thread(target=self._route_loop, args=(rank, client), daemon=True)
            t.start()
            self._threads.append(t)

    def _handshake(self, client: socket.socket) -> int:
        header = self._recv_exact(client, 17)
        length, msg_type, src, _dest, tag = unpack_header(header)
        if msg_type != MsgType.CONTROL or tag != HELLO_TAG:
            raise TransportError("Invalid handshake from worker")
        payload = self._recv_exact(client, length)
        data = loads(payload)
        return int(data["rank"])

    def _route_loop(self, rank: int, sock: socket.socket):
        try:
            while True:
                header = self._recv_exact(sock, 17)
                if not header:
                    return
                length, msg_type, src, dest, tag = unpack_header(header)
                payload = self._recv_exact(sock, length) if length else b""
                if msg_type == MsgType.DATA:
                    if dest == 0:
                        self.inbox.put(Message(src=src, dest=dest, tag=tag, payload=loads(payload)))
                    else:
                        target = self._connections.get(dest)
                        if target is None:
                            raise TransportError(f"Unknown destination rank {dest}")
                        target.sendall(pack_message(MsgType.DATA, src=src, dest=dest, tag=tag, payload=payload))
        except OSError:
            return

    def _recv_exact(self, sock: socket.socket, length: int) -> bytes:
        buf = b""
        while len(buf) < length:
            chunk = sock.recv(length - len(buf))
            if not chunk:
                return b""
            buf += chunk
        return buf

    def send(self, dest: int, tag: int, obj):
        payload = dumps(obj)
        msg = pack_message(MsgType.DATA, src=0, dest=dest, tag=tag, payload=payload)
        target = self._connections.get(dest)
        if target is None:
            raise TransportError(f"Unknown destination rank {dest}")
        target.sendall(msg)

    def send_control(self, dest: int, tag: int, obj=None):
        payload = dumps(obj) if obj is not None else b""
        msg = pack_message(MsgType.CONTROL, src=0, dest=dest, tag=tag, payload=payload)
        target = self._connections.get(dest)
        if target is None:
            raise TransportError(f"Unknown destination rank {dest}")
        target.sendall(msg)

    def recv(self, tag: Optional[int] = None, timeout: Optional[float] = None) -> Message:
        start = time.time()
        while True:
            try:
                msg = self.inbox.get(timeout=0.1)
            except queue.Empty:
                if timeout is not None and time.time() - start > timeout:
                    raise TimeoutError("recv timed out")
                continue
            if tag is None or msg.tag == tag:
                return msg
            self.inbox.put(msg)


CANCEL_TAG = 200


def connect_to_master(host: str, port: int, rank: int, cancel_event: threading.Event) -> WorkerTransport:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    hello = dumps({"rank": rank})
    sock.sendall(pack_message(MsgType.CONTROL, src=rank, dest=0, tag=HELLO_TAG, payload=hello))
    return WorkerTransport(sock, rank=rank, cancel_event=cancel_event)


def encode_args(args, kwargs) -> str:
    payload = dumps({"args": args, "kwargs": kwargs})
    return base64.b64encode(payload).decode("ascii")


def decode_args(data: str):
    payload = base64.b64decode(data.encode("ascii"))
    data = loads(payload)
    return data.get("args", []), data.get("kwargs", {})
