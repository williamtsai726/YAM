"""Tests for ``experiments.launch_yaml_eval_molmoact``.

Covers the pure-ish helpers that don't need real hardware:
* ``dynamic_smoothing`` — single-step vs. interpolated path, return value.
* ``_park_robot`` — env-None no-op, idempotency, bimanual dispatch, exception swallowing.
* ``_convert_if_any`` — empty-list shortcut, converter invocation, exception swallowing.
* ``run_one_rollout`` — y-key short-circuit, timeout, inference chunking.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments import launch_yaml_eval_molmoact as M


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEnv:
    """Records ``step_command_only`` calls; ``get_obs`` returns a stable dict."""

    def __init__(self, curr_joints, final_obs=None):
        self._curr = np.array(curr_joints, dtype=np.float32)
        self.commands = []
        self._final_obs = final_obs or {"joint_positions": self._curr.copy()}

    def get_robot_state(self):
        return {"joint_positions": self._curr.copy()}

    def get_obs(self):
        return self._final_obs

    def step_command_only(self, j):
        self.commands.append(np.array(j, dtype=np.float32))


class FakeRolloutEnv:
    """Slightly richer env for ``run_one_rollout`` — tracks step count in obs."""

    def __init__(self, dof=7):
        self.curr = np.zeros(dof, dtype=np.float32)
        self.step_count = 0

    def get_obs(self):
        return {"joint_positions": self.curr.copy(), "marker": self.step_count}

    def get_robot_state(self):
        return {"joint_positions": self.curr.copy()}

    def step_command_only(self, j):
        self.curr = np.array(j, dtype=np.float32)
        self.step_count += 1


class FakePolicy:
    def __init__(self, horizon=3, dof=7):
        self.horizon = horizon
        self.dof = dof
        self.inference_calls = 0
        self.prepare_calls = 0

    def get_action_horizon(self):
        return self.horizon

    def prepare_input(self, obs, instruction):
        self.prepare_calls += 1
        return {"obs": obs, "instruction": instruction}

    def inference(self, _input_dict):
        self.inference_calls += 1
        return {"actions": [np.zeros(self.dof, dtype=np.float32) for _ in range(self.horizon)]}


class FakeSaver:
    def __init__(self):
        self.steps = []

    def add_step(self, obs_pre, obs_post):
        self.steps.append((obs_pre, obs_post))


class FakeLiveView:
    """Returns the key configured for a given step, else ``None``."""

    def __init__(self, key_at_step=None):
        self.key_at_step = key_at_step or {}
        self.calls = 0

    def update(self, **kwargs):
        self.calls += 1
        return self.key_at_step.get(kwargs["step"])


# ---------------------------------------------------------------------------
# dynamic_smoothing
# ---------------------------------------------------------------------------


def test_dynamic_smoothing_small_delta_single_step():
    curr = np.zeros(7, dtype=np.float32)
    target = curr.copy()
    target[0] = 0.005  # max_delta = 0.005 -> steps = 0 -> single-step path
    env = FakeEnv(curr)
    out = M.dynamic_smoothing(env, target)
    assert len(env.commands) == 1
    assert np.allclose(env.commands[0], target)
    assert out is env._final_obs


def test_dynamic_smoothing_large_delta_interpolates():
    curr = np.zeros(7, dtype=np.float32)
    target = curr.copy()
    target[0] = 0.5  # steps = min(50, 100) = 50
    env = FakeEnv(curr)
    M.dynamic_smoothing(env, target)
    assert len(env.commands) == 50
    assert np.allclose(env.commands[-1], target)
    assert np.allclose(env.commands[0], curr)


def test_dynamic_smoothing_clamps_to_100_steps():
    curr = np.zeros(7, dtype=np.float32)
    target = curr.copy()
    target[0] = 5.0  # would be 500 steps; clamp to 100
    env = FakeEnv(curr)
    M.dynamic_smoothing(env, target)
    assert len(env.commands) == 100


def test_dynamic_smoothing_returns_get_obs_result():
    env = FakeEnv(np.zeros(7))
    sentinel = {"joint_positions": np.zeros(7), "marker": "post"}
    env._final_obs = sentinel
    out = M.dynamic_smoothing(env, np.zeros(7))
    assert out is sentinel


# ---------------------------------------------------------------------------
# _park_robot
# ---------------------------------------------------------------------------


@pytest.fixture
def park_globals(monkeypatch):
    """Reset the module-level globals ``_park_robot`` reads from."""
    monkeypatch.setattr(M, "_cleanup_done", False)
    monkeypatch.setattr(M, "_env", None)
    monkeypatch.setattr(M, "_bimanual", False)
    monkeypatch.setattr(M, "_left_cfg", None)
    monkeypatch.setattr(M, "_right_cfg", None)
    return monkeypatch


def test_park_robot_noop_when_env_none(park_globals):
    calls = []
    park_globals.setattr(M, "move_to_start_position", lambda *a, **k: calls.append((a, k)))
    M._park_robot()
    assert calls == []


def test_park_robot_unimanual_invokes_move(park_globals):
    calls = []
    park_globals.setattr(M, "_env", object())
    park_globals.setattr(M, "_left_cfg", {"L": 1})
    park_globals.setattr(M, "move_to_start_position", lambda *a, **k: calls.append((a, k)))
    M._park_robot()
    assert len(calls) == 1
    args, _ = calls[0]
    assert args[1] is False
    assert args[2] == {"L": 1}


def test_park_robot_bimanual_invokes_move_with_both_cfgs(park_globals):
    calls = []
    park_globals.setattr(M, "_env", object())
    park_globals.setattr(M, "_bimanual", True)
    park_globals.setattr(M, "_left_cfg", {"L": 1})
    park_globals.setattr(M, "_right_cfg", {"R": 2})
    park_globals.setattr(M, "move_to_start_position", lambda *a, **k: calls.append((a, k)))
    M._park_robot()
    assert len(calls) == 1
    args, _ = calls[0]
    assert args[1] is True
    assert args[2] == {"L": 1}
    assert args[3] == {"R": 2}


def test_park_robot_only_runs_once(park_globals):
    calls = []
    park_globals.setattr(M, "_env", object())
    park_globals.setattr(M, "_left_cfg", {})
    park_globals.setattr(M, "move_to_start_position", lambda *a, **k: calls.append(1))
    M._park_robot()
    M._park_robot()
    assert len(calls) == 1


def test_park_robot_swallows_exceptions(park_globals):
    def boom(*_a, **_k):
        raise RuntimeError("park failure")
    park_globals.setattr(M, "_env", object())
    park_globals.setattr(M, "_left_cfg", {})
    park_globals.setattr(M, "move_to_start_position", boom)
    M._park_robot()  # must not raise


# ---------------------------------------------------------------------------
# _convert_if_any
# ---------------------------------------------------------------------------


def test_convert_if_any_empty_list_skips(monkeypatch):
    called = []
    monkeypatch.setattr(M, "convert_session_to_lerobot", lambda **kw: called.append(kw))
    M._convert_if_any([], Path("/tmp/x"), "20260523_120000", {})
    assert called == []


def test_convert_if_any_passes_through_lerobot_cfg(monkeypatch, tmp_path):
    captured = {}
    def fake_convert(**kw):
        captured.update(kw)
    monkeypatch.setattr(M, "convert_session_to_lerobot", fake_convert)
    left_cfg = {
        "hz": 30,
        "lerobot": {
            "fps": 25,
            "robot_type": "yam",
            "hf_repo_id": "x/y",
            "action_mode": "abs",
            "vcodec": "libx264",
        },
    }
    rollout = tmp_path / "rollout_1"
    M._convert_if_any([rollout], tmp_path, "20260523_120000", left_cfg)

    assert captured["fps"] == 25
    assert captured["robot_type"] == "yam"
    assert captured["repo_id"] == "x/y"
    assert captured["vcodec"] == "libx264"
    assert captured["session_rollout_dirs"] == [rollout]
    assert Path(captured["output_dir"]) == tmp_path / "eval_lerobot_v30" / "20260523_120000"


def test_convert_if_any_falls_back_to_hz_when_fps_missing(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(M, "convert_session_to_lerobot", lambda **kw: captured.update(kw))
    M._convert_if_any([tmp_path / "r"], tmp_path, "ts", {"hz": 50})
    assert captured["fps"] == 50


def test_convert_if_any_swallows_converter_failure(monkeypatch, tmp_path):
    def boom(**_kw):
        raise RuntimeError("conversion exploded")
    monkeypatch.setattr(M, "convert_session_to_lerobot", boom)
    M._convert_if_any([tmp_path / "r"], tmp_path, "ts", {})  # must not raise


# ---------------------------------------------------------------------------
# run_one_rollout
# ---------------------------------------------------------------------------


def test_run_one_rollout_y_key_returns_success():
    env = FakeRolloutEnv()
    policy = FakePolicy(horizon=3)
    saver = FakeSaver()
    view = FakeLiveView(key_at_step={5: "y"})
    out = M.run_one_rollout(
        env=env, policy=policy, saver=saver, instruction="task",
        rollout_idx=0, num_rollouts=1, max_steps=100, live_view=view,
    )
    assert out.end_reason == "success"
    assert out.last_step == 5
    assert len(saver.steps) == 5


def test_run_one_rollout_n_key_returns_failure():
    env = FakeRolloutEnv()
    policy = FakePolicy(horizon=3)
    saver = FakeSaver()
    view = FakeLiveView(key_at_step={2: "n"})
    out = M.run_one_rollout(
        env=env, policy=policy, saver=saver, instruction="task",
        rollout_idx=0, num_rollouts=1, max_steps=100, live_view=view,
    )
    assert out.end_reason == "failure"
    assert out.last_step == 2


def test_run_one_rollout_q_key_returns_quit():
    env = FakeRolloutEnv()
    policy = FakePolicy(horizon=3)
    saver = FakeSaver()
    view = FakeLiveView(key_at_step={1: "q"})
    out = M.run_one_rollout(
        env=env, policy=policy, saver=saver, instruction="task",
        rollout_idx=0, num_rollouts=1, max_steps=100, live_view=view,
    )
    assert out.end_reason == "quit"
    assert out.last_step == 1


def test_run_one_rollout_timeout_returns_timeout():
    env = FakeRolloutEnv()
    policy = FakePolicy(horizon=3)
    saver = FakeSaver()
    view = FakeLiveView()
    out = M.run_one_rollout(
        env=env, policy=policy, saver=saver, instruction="task",
        rollout_idx=0, num_rollouts=1, max_steps=4, live_view=view,
    )
    assert out.end_reason == "timeout"
    assert out.last_step == 4
    assert len(saver.steps) == 4


def test_run_one_rollout_inference_chunking():
    """Policy is queried only once per chunk_size outer steps."""
    env = FakeRolloutEnv()
    policy = FakePolicy(horizon=3)
    saver = FakeSaver()
    view = FakeLiveView()
    M.run_one_rollout(
        env=env, policy=policy, saver=saver, instruction="task",
        rollout_idx=0, num_rollouts=1, max_steps=7, live_view=view,
    )
    # 7 steps, horizon=3 -> chunk refresh at steps 0, 3, 6 = 3 inferences.
    assert policy.inference_calls == 3
    assert policy.prepare_calls == 3
    assert view.calls == 7


