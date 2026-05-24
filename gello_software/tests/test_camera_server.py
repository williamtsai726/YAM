"""Tests for ``gello.cameras.camera_server.CameraServer``.

Covers:
* ``_snapshot`` payload shape, timestamps, and the missing-timestamp default.
* ``_maybe_heartbeat`` state-machine: skip-within-window, fire-and-reset.
* End-to-end REQ/REP wire protocol against a real ZMQ server bound to a
  loopback TCP port (ping, obs, unknown-cmd error).
"""
from __future__ import annotations

import pickle
import socket
import threading
import time

import numpy as np
import pytest
import zmq

from gello.cameras.camera_client import CameraClient, CameraClientError
from gello.cameras.camera_server import CameraServer


# ---------------------------------------------------------------------------
# Fake camera
# ---------------------------------------------------------------------------


class FakeCamera:
    """Stands in for ``RealSenseCamera`` for the duration of one test.

    Returns a deterministic frame (filled with ``read_count`` so successive
    reads differ) and surfaces ``_latest_frame_timestamp`` the way the real
    camera does.
    """

    def __init__(self, shape=(48, 64, 3), timestamp=None):
        self.shape = shape
        self._latest_frame_timestamp = timestamp
        self._stop_event = threading.Event()
        self.read_count = 0

    def read(self):
        self.read_count += 1
        frame = np.full(self.shape, fill_value=self.read_count % 256, dtype=np.uint8)
        return frame, None


# ---------------------------------------------------------------------------
# _snapshot
# ---------------------------------------------------------------------------


def test_snapshot_payload_shape():
    cams = {
        "left": FakeCamera(timestamp=1234.5),
        "right": FakeCamera(timestamp=2345.6),
    }
    server = CameraServer(cameras=cams)
    snap = server._snapshot()

    assert snap["ok"] is True
    assert set(snap["frames"]) == {"left", "right"}
    assert snap["frames"]["left"].shape == (48, 64, 3)
    assert snap["frames"]["left"].dtype == np.uint8
    assert set(snap["timestamps"]) == {"left", "right"}


def test_snapshot_uses_camera_timestamp():
    cams = {"front": FakeCamera(timestamp=42.0)}
    server = CameraServer(cameras=cams)
    snap = server._snapshot()
    assert snap["timestamps"]["front"] == 42.0


def test_snapshot_missing_timestamp_defaults_to_zero():
    cam = FakeCamera()
    cam._latest_frame_timestamp = None
    server = CameraServer(cameras={"cam": cam})
    snap = server._snapshot()
    assert snap["timestamps"]["cam"] == 0.0


def test_snapshot_calls_read_once_per_camera():
    cams = {"a": FakeCamera(), "b": FakeCamera()}
    server = CameraServer(cameras=cams)
    server._snapshot()
    assert cams["a"].read_count == 1
    assert cams["b"].read_count == 1


# ---------------------------------------------------------------------------
# _maybe_heartbeat
# ---------------------------------------------------------------------------


def test_maybe_heartbeat_does_not_fire_within_window():
    server = CameraServer(cameras={}, heartbeat_sec=60.0)
    server._req_window = 7
    server._last_heartbeat = time.time()  # just now
    server._maybe_heartbeat()
    assert server._req_window == 7  # untouched


def test_maybe_heartbeat_fires_and_resets():
    server = CameraServer(cameras={}, heartbeat_sec=0.01)
    server._req_window = 12
    server._last_heartbeat = time.time() - 1.0  # well past threshold
    before = server._last_heartbeat
    server._maybe_heartbeat()
    assert server._req_window == 0
    assert server._last_heartbeat > before


# ---------------------------------------------------------------------------
# End-to-end ZMQ
# ---------------------------------------------------------------------------


def _free_tcp_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def running_server():
    """Start a ``CameraServer.run()`` in a daemon thread bound to a free TCP port.

    Yields the endpoint string. Tears down via ``shutdown()`` + ``join``.
    """
    port = _free_tcp_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    server = CameraServer(
        cameras={"left": FakeCamera(timestamp=10.0), "right": FakeCamera(timestamp=20.0)},
        rep_endpoint=endpoint,
        pub_endpoint=None,
        heartbeat_sec=60.0,
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.1)
    try:
        yield endpoint
    finally:
        # Signal-only shutdown: let run()'s finally close the sockets so the
        # poll loop doesn't race with socket teardown.
        server._stop_event.set()
        thread.join(timeout=2.0)


def test_zmq_ping_round_trip(running_server):
    client = CameraClient(running_server, request_timeout_ms=2000, max_frame_age_sec=None)
    try:
        assert client.ping() is True
    finally:
        client.close()


def test_zmq_obs_round_trip(running_server):
    client = CameraClient(running_server, request_timeout_ms=2000, max_frame_age_sec=None)
    try:
        frames = client.get_obs()
    finally:
        client.close()
    assert set(frames) == {"left", "right"}
    assert frames["left"].shape == (48, 64, 3)
    assert frames["left"].dtype == np.uint8


def test_zmq_unknown_cmd_returns_error(running_server):
    """Send a raw bad ``cmd`` so we hit ``_handle_request``'s error branch."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.setsockopt(zmq.SNDTIMEO, 2000)
    sock.connect(running_server)
    try:
        sock.send(pickle.dumps({"cmd": "definitely_not_a_real_cmd"}))
        resp = pickle.loads(sock.recv())
    finally:
        sock.close(linger=0)

    assert resp["ok"] is False
    assert "unknown cmd" in resp["error"]


def test_zmq_stale_frame_detection():
    """Server publishes timestamps; client raises if ``max_frame_age_sec`` exceeded."""
    port = _free_tcp_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    cams = {"cam": FakeCamera(timestamp=time.time() - 5.0)}  # 5s old
    server = CameraServer(cameras=cams, rep_endpoint=endpoint, pub_endpoint=None)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.1)
    try:
        client = CameraClient(endpoint, request_timeout_ms=2000, max_frame_age_sec=0.5)
        with pytest.raises(CameraClientError, match="Stale frame"):
            client.get_obs()
        client.close()
    finally:
        server._stop_event.set()
        thread.join(timeout=2.0)
