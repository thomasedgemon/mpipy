"""Wire protocol helpers for framing and serializing messages."""

import enum
import pickle
import struct
from typing import Tuple


class MsgType(enum.IntEnum):
    DATA = 1
    CONTROL = 2


HEADER_STRUCT = struct.Struct("!I B I I I")
# length (uint32), type (uint8), src (uint32), dest (uint32), tag (uint32)


def pack_message(msg_type: MsgType, src: int, dest: int, tag: int, payload: bytes) -> bytes:
    header = HEADER_STRUCT.pack(len(payload), int(msg_type), src, dest, tag)
    return header + payload


def unpack_header(data: bytes) -> Tuple[int, MsgType, int, int, int]:
    length, msg_type, src, dest, tag = HEADER_STRUCT.unpack(data)
    return length, MsgType(msg_type), src, dest, tag


def dumps(obj) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def loads(payload: bytes):
    return pickle.loads(payload)
