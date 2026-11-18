"""Wire protocol between a hackathon policy server (nwaves_hackathon/serve.py, runs the team's policy.py) and
its clients: the simulation scorer (nwaves-eval) and the real-arm runner.

Every message is a 4-byte big-endian length followed by that many bytes of JSON. An observation
message is followed by the JPEG bytes of its images, in the order of its "images" list with the
sizes given in "sizes". Only numpy, opencv and the standard library are needed, so the same file
works in the Isaac Lab python, the session's policy venv and on the robot.

    client -> server   {"type": "hello"}                              -> {"ok": true, "name": ..., "model_type": ...}
                       {"type": "reset", "task": ..., "instruction": ...}  -> {"ok": true}
                       {"type": "observation", "joint_state": [6], "task": ..., "instruction": ...,
                        "time_left": s, "images": ["scene", "wrist"], "sizes": {...}} + JPEGs
                                                                       -> {"actions": [[6], ...], "ms": ...}
                       {"type": "bye"}
"""
import json
import socket
import time

import numpy as np

JPEG_QUALITY = 92
IMAGE_NAMES = ("scene", "wrist")


def send_msg(sock, obj, blobs=()):
    data = json.dumps(obj).encode()
    sock.sendall(len(data).to_bytes(4, "big") + data + b"".join(blobs))


def recv_exact(sock, n):
    parts, got = [], 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            raise ConnectionError("connection closed")
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def recv_msg(sock):
    n = int.from_bytes(recv_exact(sock, 4), "big")
    return json.loads(recv_exact(sock, n))


def encode_jpeg(rgb):
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("jpeg encoding failed")
    return buf.tobytes()


def decode_jpeg(blob):
    import cv2
    bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class PolicyClient:
    """Talks to one policy session on a nwaves_hackathon/serve.py server."""

    def __init__(self, host="127.0.0.1", port=7600, connect_timeout=180.0, io_timeout=60.0, hello_timeout=900.0):
        deadline = time.monotonic() + connect_timeout
        last = None
        while time.monotonic() < deadline:
            try:
                self.sock = socket.create_connection((host, port), timeout=5.0)
                break
            except OSError as exc:
                last = exc
                time.sleep(0.5)
        else:
            raise ConnectionError(f"no policy server at {host}:{port} ({last})")
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(hello_timeout)       # the server loads the model now: a large one can take minutes
        send_msg(self.sock, {"type": "hello"})
        self.info = recv_msg(self.sock)
        if not self.info.get("ok"):
            raise RuntimeError(f"policy server refused: {self.info}")
        self.sock.settimeout(io_timeout)
        self.latencies_ms = []

    def reset(self, task, instruction):
        send_msg(self.sock, {"type": "reset", "task": task, "instruction": instruction})
        reply = recv_msg(self.sock)
        if not reply.get("ok"):
            raise RuntimeError(f"policy reset failed: {reply.get('error')}")

    def act(self, joint_state, images, task, instruction, time_left):
        blobs, sizes = [], {}
        for name in IMAGE_NAMES:
            blobs.append(encode_jpeg(images[name]))
            sizes[name] = len(blobs[-1])
        t0 = time.perf_counter()
        send_msg(self.sock, {"type": "observation", "joint_state": [float(v) for v in joint_state], "task": task,
                             "instruction": instruction, "time_left": float(time_left), "images": list(IMAGE_NAMES),
                             "sizes": sizes}, blobs)
        reply = recv_msg(self.sock)
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if "error" in reply:
            raise RuntimeError(f"policy error: {reply['error']}")
        actions = np.asarray(reply["actions"], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None]
        if actions.ndim != 2 or actions.shape[1] != 6 or len(actions) == 0 or not np.all(np.isfinite(actions)):
            raise ValueError(f"policy returned actions of shape {actions.shape} (need (6,) or (k, 6), finite)")
        return actions

    def close(self):
        try:
            send_msg(self.sock, {"type": "bye"})
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
