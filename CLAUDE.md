# CLAUDE.md

Guidance for future Claude sessions working in this repo.

## What this repo is

**Bimanual YAM** — a workspace stitching together four upstream-ish projects to run teleop, data collection, and policy evaluation on a bimanual YAM robot arm setup.

Top-level layout:

| Path | Purpose |
|---|---|
| `i2rt/` | Low-level motor / CAN driver code. Contains `motor_config_tool/` (timeout/zero/ping) and `scripts/reset_all_can.sh`. |
| `gello_software/` | Main runtime: configs, teleop launchers, data collection, eval. Most day-to-day work happens here under `experiments/` and `configs/`. |
| `lerobot/` | Local checkout of LeRobot used for dataset conversion / training-side compat. |
| `oculus_reader/` | Optional Oculus VR teleop input. |
| `robots_realtime/` | Newer realtime-control sandbox (untracked in git on this machine). |
| `molmoact_to_lerobot_v30.py` | Top-level converter: raw collected JSON → LeRobot v3.0 dataset, with optional HF upload + tag. |
| `docs/` | Lab-facing tutorials; `docs/grasp_lab_eval.md` is the GRASP rig walkthrough. |
| `videos/` | Gitignored output dir. |

## This machine's environment

- **Conda env:** `ai2_yam` (Python 3.11). Always `conda activate ai2_yam` before running anything.
- **CAN interfaces (machine-specific!):** `can_leader_l` (left arm) and `can_follower_r` (right arm). Verify with `ip link show | grep can` before assuming names.
- The configs already point at the right channels:
  - `gello_software/configs/yam_left.yaml` → `channel: can_leader_l`
  - `gello_software/configs/yam_right.yaml` → `channel: can_follower_r`
- Other config files in `gello_software/configs/` (e.g. `yam_passive.yaml`, `yam_active.yaml`) still use legacy `can_left`/`can_right` names — they are not the active configs and may be stale.

## Standard startup sequence (every fresh boot / replug)

```bash
conda activate ai2_yam
sh i2rt/scripts/reset_all_can.sh
python i2rt/i2rt/motor_config_tool/set_timeout.py --channel can_leader_l
python i2rt/i2rt/motor_config_tool/set_timeout.py --channel can_follower_r
# Only if using the linear gripper at full grip and the gripper drifted on power-cycle:
python i2rt/i2rt/motor_config_tool/set_zero.py --channel=can_leader_l --motor_id=7
python i2rt/i2rt/motor_config_tool/set_zero.py --channel=can_follower_r --motor_id=7
```

`set_timeout.py` with no `--timeout` flag sets timeout to `0` (disabled) for motors 1–7. That is the desired state for long teleop / data collection runs — the default motor timeout otherwise causes arms to collapse.

## Key entry points (`gello_software/experiments/`)

- `launch_yaml.py` — bimanual teleop.
- `launch_yaml_collect_data.py` — teleop + data collection + auto-convert/upload pipeline.
- `launch_yaml_eval.py` — eval for `dp` (DiffusionPolicy) or `pi05` (PI05Policy); selected via `configs/yam_left.yaml: policy.type`.
- `launch_yaml_eval_molmoact.py` — eval against the MolmoAct-v2 policy. `eval.mode` in `yam_left.yaml` picks `local` (in-process `MolmoActLocal` loading the HF snapshot via transformers; needs ~10–14 GB VRAM at bf16) or `server` (HTTP POST to a remote FastAPI server at `eval.molmoact_server`; accepts a full URL or bare `host:port`). Session-based: `-n N` runs N rollouts, arm interpolates to `agent.start_joints` between rollouts before each instruction prompt (Enter reuses last). A live 3-pane cv2 view shows LEFT / FRONT / RIGHT; press `y`/`n`/`q` in that window to end + label, or let it time out for a stdin prompt. When the camera server is on (default) the viewer is driven by a daemon thread subscribed to the PUB stream, so it keeps repainting through `policy.inference()`. Saves PNG + `episode.h5` per rollout under `{base_dir}/data/{task_directory}/{eval,success,failure}/...` (DROID-style), then batch-converts labeled rollouts to a LeRobot v3.0 dataset under `eval_lerobot_v30/{session_ts}/` at end-of-session. Helpers live in `gello/utils/eval_utils.py`. End-user walkthrough: `docs/grasp_lab_eval.md`.
- `launch_yaml_replay.py`, `launch_yaml_open_loop.py`, `launch_yaml_molmoact_open_loop.py` — replay / open-loop testing from collected JSON episodes.

All launchers take `--left_config_path` and `--right_config_path`; most config knobs (cameras, storage, lerobot conversion, policy) live in `configs/yam_left.yaml` — `yam_right.yaml` mainly carries the right-arm robot/agent block.

## Camera server (eval-only)

`gello_software/gello/cameras/camera_server.py` runs the three RealSense pipelines in a long-lived process and serves the latest frames over ZMQ (REP on `:5555`, PUB on `:5556`). `camera_client.py` exposes `CameraClient` (REQ-side wrapper the policy uses for on-demand obs) and `CameraSubscriber` (SUB-side, drained by the `LiveCameraView` render thread). `RobotEnv` accepts a `camera_client=` kwarg and a `step_command_only(joints)` so sub-step interpolation no longer reads cameras.

Why: in the old path `dynamic_smoothing` re-read all 3 cameras on every interpolation tick (up to 100× per outer step). With the server architecture cameras stay warm across sessions and are sampled only when the policy actually needs an obs; the cv2 viewer subscribes to the PUB stream from a daemon thread so it keeps painting through `policy.inference()`.

Default-on for the MolmoAct eval launcher via `eval.camera_server.enabled: true` in `yam_left.yaml`. Two-terminal usage:

```bash
# Terminal A: leaves cameras hot across the workstation session
bash gello_software/scripts/start_camera_server.sh    # script hardcodes --config configs/yam_left.yaml
# Terminal B: run eval as usual
```

Set `eval.camera_server.enabled: false` to fall back to the in-process camera path (slower; the viewer freezes during inference). Data collection / replay / open-loop launchers still use the in-process path — the flag is per-launcher.

## Tests

`gello_software/tests/` has pytest coverage for the camera server (`_snapshot`, `_maybe_heartbeat`, end-to-end REQ/REP wire protocol, stale-frame detection) and the MolmoAct eval launcher (`dynamic_smoothing`, `_park_robot`, `_convert_if_any`, `run_one_rollout`). All tests use inline fakes — no RealSense / no CAN motors required.

```bash
cd gello_software && python -m pytest tests/ -q
```

`tests/conftest.py` puts `experiments/` on `sys.path` so `from molmoact import ...` resolves the same way the launcher resolves it at runtime.

## Conventions / gotchas

- **Don't blindly copy commands from README that reference `can_left`/`can_right`** for this machine — use `can_leader_l`/`can_follower_r`. If updating docs, mirror what's actually in `yam_left.yaml`/`yam_right.yaml`.
- The data-collection keypad (`s` start / `a` save+end / `b` discard+end) requires keyboard focus on the color pad window. `ctrl+c` only does cleanup — it skips the convert/upload pipeline.
- For `launch_yaml_eval_molmoact.py`, `ctrl+c` IS handled gracefully: the in-progress rollout is flushed to `eval/{timestamp}/` with an `err.md` marker, and the LeRobot conversion still runs over rollouts already labeled in this session.
- Conversion (`molmoact_to_lerobot_v30.py`) defaults to reading `gello_software/configs/yam_left.yaml` for `data_dir`, `output_dir`, and upload settings; CLI flags override.
- Setup order recommended by upstream: install `i2rt` first, then `gello_software`, then `lerobot`. Each subdir has its own README/requirements.

## When asked to add/change a runtime command

Cross-check three places before suggesting it works:
1. The actual CAN interface names (`ip link show | grep can`).
2. The `channel:` field in `configs/yam_left.yaml` and `configs/yam_right.yaml`.
3. That `ai2_yam` is activated.
