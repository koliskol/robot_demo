"""Closed-loop rollout inference server for a fine-tuned GR00T N1.6 checkpoint - a drop-in
alternative to policy_server.py for collect_pickplace_demo.py's --rollout mode.

Speaks the exact same wire protocol as policy_server.py (see policy_wire.py's docstring for the
framing and why it's not pickle), so collect_pickplace_demo.py needs zero changes: point
--policy-host/--policy-port (or --policy2-host/--policy2-port) at this server instead of an ACT
policy_server.py instance and it just works - PolicyClient.predict()/reset() don't know or care
which model is behind the socket.

Runs in its own `gr00t_infer` conda env (python 3.10, torch 2.7.1+cu128, transformers 4.51.3,
gr00t installed editable from a local NVIDIA/Isaac-GR00T-main checkout pinned to the
`n1.6-release` tag - the checkpoint's config.json declares architecture "Gr00tN1d6", which only
exists at that tag; the current `main` branch has moved on to N1.7-only and has no registered
pipeline for it). Kept separate from both `isaac_sim` and `lerobot` envs for the same
conflict-avoidance reason CLAUDE.md gives for the ACT split - this stack is even heavier
(flash-attn, a ~2B-param Eagle VLM backbone).

State/action layout matches collect_pickplace_demo.py's 22-dim schema exactly (confirmed against
gr00t_config/modality.json and the checkpoint's own processor_config.json, which stores this same
per-key breakdown from training):
    left_arm[0:7], right_arm[7:14], torso[14:19], left_gripper[19:20], right_gripper[20:21],
    chassis_forward[21:22]

Gr00tPolicy.get_action() returns a 16-step action chunk per call (this checkpoint's trained
action_horizon), not a single step - unlike ACT's policy.select_action(), which dequeues its own
internal chunk one step at a time. This server always returns just the chunk's first step
(index 0) and lets collect_pickplace_demo.py's existing per-record_fps-tick predict() call keep
re-querying, the same cadence it already uses for the ACT server - it does not attempt to consume
the remaining 15 predicted steps itself. Whether re-predicting every tick vs. executing more of
each chunk matters for closed-loop tracking quality is unverified; this is the simplest thing that
matches the existing client contract (one predict() call in, one action vector out).

Live-verified once on this machine (RTX 4080 Laptop, 12GB): checkpoint loads and runs get_action()
using ~6.9GB VRAM (bf16 weights + one forward pass), comfortably inside budget - well under the
9.81GB raw weight size, and nowhere near the tight-fit worried about when this env didn't exist yet.

Run (in the gr00t_infer env, from this repo's directory so gr00t_datasets/ paths etc. resolve if
you use them elsewhere):

    conda run -n gr00t_infer python gr00t_policy_server.py \
        --checkpoint-dir ./gr00t_model_trained/pickup_policy_ckpt \
        --task pickup_policy
"""

import argparse
import socket
import sys
import traceback

import numpy as np

from policy_wire import decode_array, encode_array, recv_message, send_message

STATE_ACTION_GROUPS = [
    ("left_arm", 0, 7),
    ("right_arm", 7, 14),
    ("torso", 14, 19),
    ("left_gripper", 19, 20),
    ("right_gripper", 20, 21),
    ("chassis_forward", 21, 22),
]


def build_observation(image: np.ndarray, state: np.ndarray, task: str, language_key: str) -> dict:
    state_dict = {name: state[None, None, start:end] for name, start, end in STATE_ACTION_GROUPS}
    return {
        "video": {"head_camera": image[None, None]},
        "state": state_dict,
        "language": {language_key: [[task]]},
    }


def action_to_vector(action: dict) -> np.ndarray:
    """Concatenate the chunk's first predicted step back into our flat 22-dim order."""
    parts = [np.asarray(action[name])[0, 0] for name, _, _ in STATE_ACTION_GROUPS]
    return np.concatenate(parts).astype(np.float32)


def serve_connection(conn: socket.socket, policy, language_key: str, default_task: str) -> None:
    while True:
        try:
            request = recv_message(conn)
        except ConnectionError:
            print("[gr00t_policy_server] client disconnected")
            return

        cmd = request.get("cmd")
        if cmd == "reset":
            policy.reset()
            send_message(conn, {"ok": True})
        elif cmd == "predict":
            image = decode_array(request["image"])
            state = decode_array(request["state"]).astype(np.float32)
            task = request.get("task") or default_task
            obs = build_observation(image, state, task, language_key)
            action, _info = policy.get_action(obs)
            send_message(conn, {"action": encode_array(action_to_vector(action))})
        else:
            send_message(conn, {"error": f"unknown cmd {cmd!r}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Assembled checkpoint dir - config.json/model.safetensors.index.json/processor_config.json/statistics.json/embodiment_id.json plus the safetensors shards, all in one directory (see gr00t_model_trained/pickup_policy_ckpt for the layout).")
    parser.add_argument("--task", type=str, required=True, help="Task string sent as the language annotation on every predict() call unless the client overrides it, e.g. 'pickup_policy'.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address - keep this 127.0.0.1, this protocol has no auth.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gr00t-repo", type=str, default="/home/kholis/Isaac-GR00T-main", help="Path to the NVIDIA/Isaac-GR00T checkout (pinned to n1.6-release) providing the `gr00t` package, in case it isn't pip-installed in this env.")
    args = parser.parse_args()

    if args.host != "127.0.0.1" and args.host != "localhost":
        print(f"[gr00t_policy_server] WARNING: binding to {args.host!r}, not 127.0.0.1 - this protocol has no authentication.")

    sys.path.insert(0, args.gr00t_repo)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=args.checkpoint_dir,
        device=args.device,
        strict=True,
    )
    language_key = policy.language_key

    print(f"[gr00t_policy_server] loaded {args.checkpoint_dir}, language_key={language_key!r}, listening on {args.host}:{args.port}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        while True:
            conn, addr = listener.accept()
            print(f"[gr00t_policy_server] client connected from {addr}")
            with conn:
                try:
                    serve_connection(conn, policy, language_key, args.task)
                except Exception:
                    traceback.print_exc()


if __name__ == "__main__":
    main()
