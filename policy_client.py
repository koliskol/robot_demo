"""Client for policy_server.py's rollout inference protocol - used by collect_pickplace_demo.py's
--rollout mode. See CLAUDE.md's "LeRobot pick-and-place pipeline" and policy_wire.py's docstring
for why this is a socket client rather than an in-process import: torch/lerobot must not be
installed into the isaac_sim conda env, so this file (and policy_wire.py underneath it) is
deliberately stdlib + numpy only, safe to import from isaac_sim's Python without pulling in either
stack.
"""

import socket

import numpy as np

from policy_wire import decode_array, encode_array, recv_message, send_message


class PolicyClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None

    def connect(self, timeout_s: float = 10.0) -> None:
        """Fails loudly (raises) if the server isn't reachable - don't silently fall back to
        teleop, a --rollout run with no policy behind it is a misconfiguration, not a valid mode."""
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout_s)
        self.sock.settimeout(timeout_s)

    def reset(self) -> None:
        send_message(self.sock, {"cmd": "reset"})
        reply = recv_message(self.sock)
        if not reply.get("ok"):
            raise RuntimeError(f"policy_server reset failed: {reply}")

    def predict(self, image: np.ndarray, state: np.ndarray, task: str) -> np.ndarray:
        send_message(
            self.sock,
            {"cmd": "predict", "image": encode_array(image), "state": encode_array(state.astype(np.float32)), "task": task},
        )
        reply = recv_message(self.sock)
        if "error" in reply:
            raise RuntimeError(f"policy_server predict failed: {reply['error']}")
        return decode_array(reply["action"])

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None
