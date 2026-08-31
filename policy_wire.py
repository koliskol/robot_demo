"""Wire protocol shared by policy_server.py (lerobot conda env) and policy_client.py (isaac_sim
conda env, imported from collect_pickplace_demo.py) - see CLAUDE.md's "LeRobot pick-and-place
pipeline" for why closed-loop rollout needs two processes rather than one script importing both
`isaacsim` and `lerobot`/`torch`.

Deliberately stdlib + numpy only (no torch, no lerobot) so this same file can be imported
unmodified from both conda envs without pulling either stack into the other.

Framing: one message per direction, 4-byte big-endian length prefix + UTF-8 JSON body. Numpy
arrays travel as base64-encoded raw bytes with explicit shape/dtype in the JSON - not pickle, to
avoid an arbitrary-code-exec deserializer even though this is meant for 127.0.0.1-only, single
trusted local client traffic.
"""

import base64
import json
import socket
import struct

import numpy as np

_LENGTH_PREFIX = struct.Struct(">I")


def encode_array(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    return {"dtype": str(arr.dtype), "shape": list(arr.shape), "data": base64.b64encode(arr.tobytes()).decode("ascii")}


def decode_array(obj: dict) -> np.ndarray:
    raw = base64.b64decode(obj["data"])
    return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])


def send_message(sock: socket.socket, message: dict) -> None:
    body = json.dumps(message).encode("utf-8")
    sock.sendall(_LENGTH_PREFIX.pack(len(body)) + body)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while reading a message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> dict:
    (length,) = _LENGTH_PREFIX.unpack(_recv_exact(sock, _LENGTH_PREFIX.size))
    return json.loads(_recv_exact(sock, length).decode("utf-8"))
