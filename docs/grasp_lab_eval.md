# Running MolmoAct eval on the GRASP-lab YAM rig

How to bring up the bimanual YAM rig and run a MolmoAct evaluation session end-to-end on the lab workstation. If something looks off, cross-check against `CLAUDE.md` at the repo root — that file is the canonical record of machine-specific quirks (CAN names, conda env, etc.).

---

## 0. Hardware checklist

Before you touch a terminal:

- Both YAM arms are powered and the e-stop is released.
- Both Gello teleop arms are powered (only relevant if you also plan to collect data — eval doesn't use them, but the launcher still imports the Gello agent).
- The 3 RealSense cameras (`left_camera`, `front_camera`, `right_camera`) are plugged into USB 3 ports. The names refer to mount position, not device serial — those live in `gello_software/configs/yam_left.yaml` under `sensors.cameras`.
- The CAN-to-USB adapters for both arms are plugged in. Verify with:
  ```bash
  ip link show | grep can
  ```
  You should see `can_leader_l` (left arm) and `can_follower_r` (right arm). **If you see different names** (`can0`, `can_left`, etc.) something has been re-cabled — fix the cables or update both YAML configs before proceeding. The configs ship pointing at the lab's wiring.

---

## 1. One-time per-boot setup

Run these once after every workstation reboot or after re-plugging the CAN/USB devices.

```bash
cd ~/projects/YAM/
conda activate ai2_yam

# Bring CAN interfaces up at 1 Mbit (sudo password required the first time)
bash i2rt/scripts/reset_all_can.sh

# Disable the motor watchdog so arms don't collapse during long sessions
python i2rt/i2rt/motor_config_tool/set_timeout.py --channel can_leader_l
python i2rt/i2rt/motor_config_tool/set_timeout.py --channel can_follower_r
```

Optional — only if the linear gripper drifted at power-on and `set_timeout` reported the gripper motor outside its expected range:
```bash
python i2rt/i2rt/motor_config_tool/set_zero.py --channel=can_leader_l   --motor_id=7
python i2rt/i2rt/motor_config_tool/set_zero.py --channel=can_follower_r --motor_id=7
```

---

## 2. Each-session startup

Two terminals. The camera server stays up across many eval sessions — you only need to restart it if you change camera wiring or `sensors.cameras` in the YAML.

### Terminal A — camera server (long-lived)

```bash
cd ~/projects/YAM/
conda activate ai2_yam
bash gello_software/scripts/start_camera_server.sh
```

Look for these lines in the output:
```
REP bound on tcp://127.0.0.1:5555
PUB bound on tcp://127.0.0.1:5556
```

Leave this terminal running. The server holds all three RealSense pipelines warm so eval launches in seconds instead of waiting on the per-camera ~3 s startup. It also lets the cv2 viewer keep refreshing during `policy.inference()`, because the viewer subscribes to the PUB stream from a background thread instead of waiting for the policy's obs.

### Terminal B — eval session

```bash
cd ~/projects/YAM/
conda activate ai2_yam
python gello_software/experiments/launch_yaml_eval_molmoact.py \
    --left_config_path  gello_software/configs/yam_left.yaml \
    --right_config_path gello_software/configs/yam_right.yaml \
    -n 10
```

`-n 10` means "run 10 rollouts this session". You can interrupt at any time with Ctrl-C — the in-progress rollout is flushed with an `err.md` marker and any rollouts already labelled this session still get converted to a LeRobot v3.0 dataset.

---

## 3. What happens in a rollout

For each of the N rollouts:

1. **Arm resets** to `agent.start_joints` (from the YAML). You'll see `Moving robot to start position: …` — this is your cue to re-arrange the workspace for the next trial.
2. **Instruction prompt** on stdin:
   ```
   [rollout 1/10] Task instruction (Enter to reuse 'fold the paper'):
   ```
   Type a new instruction, or press Enter to reuse the previous one.
3. **The rollout runs.** A 3-pane cv2 window (`YAM Eval`) shows LEFT / FRONT / RIGHT cameras with a header line for rollout index, step counter, and the active instruction.
4. **End the rollout** by pressing a key **in the cv2 window** (not the terminal):
   - `y` → success
   - `n` → failure
   - `q` → quit, no label (saved under `eval/`)
   - Or do nothing and let it time out at `max_steps` (default 1000) → stdin prompt afterwards.

---

## 4. Where files land

Under `{storage.base_dir}/data/{storage.task_directory}/` (defaults: `/home/kostas-lab/projects/YAM/data/shirt/`):

```
data/shirt/
├── eval/<timestamp>/              # quit / unlabeled rollouts stay here
├── success/<YYYY-MM-DD>/<timestamp>/
├── failure/<YYYY-MM-DD>/<timestamp>/
└── eval_lerobot_v30/<session_ts>/ # LeRobot v3.0 dataset built at end of session
```

Each rollout directory contains `episode.h5` (joint trajectory + instruction) and one PNG per camera per frame under `left_rgb/`, `front_rgb/`, `right_rgb/`.

To change where data lands, edit `storage.base_dir` and `storage.task_directory` in `gello_software/configs/yam_left.yaml`.

---

## 5. Common knobs to tweak

All in `gello_software/configs/yam_left.yaml`:

| Key | What it does |
|---|---|
| `eval.mode` | `local` runs MolmoAct in-process (needs a ~10–14 GB GPU for bf16). `server` POSTs obs to a remote MolmoAct FastAPI server. |
| `eval.local.repo_id` | HF repo for the local checkpoint. Default is `allenai/MolmoAct2-BimanualYAM`. |
| `eval.local.dtype` | `bfloat16` (~10–14 GB), `float16`, or `float32` (~26 GB). |
| `eval.local.warmup` | `true` eats the ~30 s first-call latency at startup so rollout step 0 isn't slow. Set `false` only if you'll never use step 0. |
| `eval.molmoact_server` | Used when `mode: server`. Accepts full URL, bare `host:port`, or ngrok hostname; `/act` is appended automatically. |
| `eval.camera_server.enabled` | `true` (default) talks to the long-lived camera server. Set `false` to fall back to opening RealSense devices in-process — slower, and the cv2 viewer freezes during inference. |
| `eval.live_view_enabled` | `false` disables the cv2 window entirely (useful for headless runs). |
| `max_steps` | Rollout timeout in control steps (top-level, not under `eval`). |
| `hz` | Control rate. Don't change unless you know why. |
| `storage.task_directory` | Sub-folder under `data/` to write rollouts into. Bump this when switching tasks so success/failure don't intermix. |

---

## 6. Troubleshooting

**`Camera server at tcp://127.0.0.1:5555 did not respond to ping.`**
The camera server isn't running, or it died. Restart Terminal A: `bash gello_software/scripts/start_camera_server.sh`. If the server itself crashes on startup, the most common cause is a RealSense already held by another process — `lsof | grep -i realsense` to find the culprit.

**`Stale frame from front_camera: 1.3s old (>0.500s).`**
A camera stopped streaming (USB renegotiated, cable loose, RealSense firmware hiccup). The camera server is still up, but at least one pipeline is wedged. Restart Terminal A.

**Arms collapse a few seconds after launching.**
You skipped the `set_timeout.py` step. Park the arms manually, then run both `set_timeout.py` commands from §1.

**`from gello.cameras... ImportError`** when running the eval launcher.
You're not in the `ai2_yam` conda env (or it's missing deps). `conda activate ai2_yam` and retry.

**cv2 window doesn't show up over SSH.**
Set `eval.live_view_enabled: false` in `yam_left.yaml` for headless runs. You'll still get the stdin prompts on timeout.

**Ctrl-C didn't write a LeRobot dataset.**
The session-end conversion runs only over rollouts that were *labeled* (success/failure). A session where you only ran Ctrl-C-interrupted rollouts will print `No labeled rollouts this session — nothing to convert.` That's expected.

---

## 7. Useful pointers

- `CLAUDE.md` (repo root) — machine-specific quirks; read this before suggesting any setup changes.
- `gello_software/configs/yam_left.yaml` — all the eval knobs in one place.
- `gello_software/gello/cameras/camera_server.py` — the ZMQ camera server. CLI: `python -m gello.cameras.camera_server --help`.
- `gello_software/gello/cameras/camera_client.py` — standalone live viewer for the PUB stream:
  ```bash
  cd ~/projects/YAM/gello_software
  python -m gello.cameras.camera_client --mode sub
  ```
  Use this to sanity-check the camera server independently of the eval loop.
- `gello_software/experiments/launch_yaml_eval_molmoact.py` — the launcher.
- `gello_software/tests/` — `python -m pytest gello_software/tests/ -q` runs 28 unit tests against the camera server and launcher (no hardware needed).
