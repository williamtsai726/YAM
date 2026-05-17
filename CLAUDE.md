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
- `launch_yaml_eval_molmoact.py` — eval against a remote MolmoAct-v2 server (URL hardcoded near top of `molmoact.py`, ~line 13). Session-based: `-n N` runs N rollouts, prompts for an instruction per rollout (Enter reuses last), shows a live 3-pane cv2 view; press `y`/`n`/`q` in the cv2 window to end + label, or let it time out for a stdin prompt. Saves PNG + `episode.h5` per rollout under `{base_dir}/{task_directory}/{eval,success,failure}/...` (DROID-style), then batch-converts labeled rollouts to a LeRobot v3.0 dataset under `eval_lerobot_v30/{session_ts}/` at end-of-session. Helpers live in `gello/utils/eval_utils.py`.
- `launch_yaml_replay.py`, `launch_yaml_open_loop.py`, `launch_yaml_molmoact_open_loop.py` — replay / open-loop testing from collected JSON episodes.

All launchers take `--left_config_path` and `--right_config_path`; most config knobs (cameras, storage, lerobot conversion, policy) live in `configs/yam_left.yaml` — `yam_right.yaml` mainly carries the right-arm robot/agent block.

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
