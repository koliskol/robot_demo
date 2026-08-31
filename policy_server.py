"""Closed-loop rollout inference server for a trained LeRobot policy - the `lerobot`-env half of
the two-process split collect_pickplace_demo.py's --rollout mode needs (see CLAUDE.md's "LeRobot
pick-and-place pipeline" for why: torch/lerobot must not be installed into the isaac_sim conda env,
so a live Isaac Sim scene and policy inference can't share one process/env).

Loads a checkpoint via lerobot_policy_utils.load_policy() (the same loading path
evaluate_act_checkpoint.py uses) and serves it over a 127.0.0.1-only TCP socket using the framed
JSON protocol in policy_wire.py - see that module's docstring for the wire format and why it's not
pickle. Single persistent connection, one client (collect_pickplace_demo.py) at a time, synchronous
request/response - there's no need for concurrency here, a rollout is inherently sequential.

Two request types:
    {"cmd": "reset"} -> {"ok": true}
        Clears the policy's internal action-chunk queue (ACTPolicy.reset()) - call this once at
        the start of every fresh rollout attempt, the same way training treats a new episode.
    {"cmd": "predict", "image": <encoded HWC uint8 array>, "state": <encoded float32 (21,) array>,
     "task": <str>}
        -> {"action": <encoded float32 (21,) array>}
        Runs one step of the policy's real inference path (predict_action(), the same helper
        evaluate_act_checkpoint.py uses - internally this is policy.select_action(), which handles
        ACT's chunk-and-dequeue behavior itself).

Run in the `lerobot` conda env (same one used for convert_to_lerobot.py/training):

    conda run -n lerobot python policy_server.py \
        --checkpoint-dir ./act_training/table_to_table2/checkpoints/last/pretrained_model \
        --dataset-root ./lerobot_dataset --dataset-repo-id local/pick_box_table_to_table2
"""

import argparse
import socket
import traceback

import torch
from lerobot.utils.control_utils import predict_action

from lerobot_policy_utils import load_policy
from policy_wire import decode_array, encode_array, recv_message, send_message


def serve_connection(conn: socket.socket, policy, preprocessor, postprocessor, device: torch.device, camera_key: str) -> None:
    while True:
        try:
            request = recv_message(conn)
        except ConnectionError:
            print("[policy_server] client disconnected")
            return

        cmd = request.get("cmd")
        if cmd == "reset":
            policy.reset()
            send_message(conn, {"ok": True})
        elif cmd == "predict":
            image = decode_array(request["image"])
            state = decode_array(request["state"])
            observation = {camera_key: image, "observation.state": state}
            action = predict_action(
                observation, policy, device, preprocessor, postprocessor, use_amp=False, task=request.get("task", "")
            )
            # predict_action's docstring claims it strips the batch dim; the installed lerobot
            # (0.4.4) doesn't - it returns shape (1, N) (kept for vectorized-env compatibility
            # upstream). Squeeze explicitly rather than relying on broadcasting to paper over it.
            send_message(conn, {"action": encode_array(action.squeeze(0).cpu().numpy())})
        else:
            send_message(conn, {"error": f"unknown cmd {cmd!r}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to a saved pretrained_model/ directory.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root of the LeRobotDataset this checkpoint was trained on.")
    parser.add_argument("--dataset-repo-id", type=str, required=True, help="repo_id of that same dataset.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address - keep this 127.0.0.1, this protocol has no auth.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.host != "127.0.0.1" and args.host != "localhost":
        print(f"[policy_server] WARNING: binding to {args.host!r}, not 127.0.0.1 - this protocol has no authentication.")

    device = torch.device(args.device)
    policy, preprocessor, postprocessor, camera_key = load_policy(
        args.checkpoint_dir, args.dataset_root, args.dataset_repo_id, device=args.device
    )
    print(f"[policy_server] loaded checkpoint, camera_key={camera_key!r}, listening on {args.host}:{args.port}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        while True:
            conn, addr = listener.accept()
            print(f"[policy_server] client connected from {addr}")
            with conn:
                try:
                    serve_connection(conn, policy, preprocessor, postprocessor, device, camera_key)
                except Exception:
                    traceback.print_exc()


if __name__ == "__main__":
    main()
