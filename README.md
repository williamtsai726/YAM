# Bimanual YAM
A robotics framework for Teleoperation, Data Collection, and model evaluation on Bimanual YAM.

## Repository Layout
| Path | What it is |
|---|---|
| `i2rt/` | Low-level motor + CAN driver code, motor config tools, CAN reset script. |
| `gello_software/` | Main runtime: configs, teleop / data collection / eval launchers. |
| `lerobot/` | Local LeRobot checkout used for dataset conversion. |
| `oculus_reader/` | Optional Oculus VR teleop input. |
| `molmoact_to_lerobot_v30.py` | Top-level converter: raw collected JSON → LeRobot v3.0, with optional HF upload. |

## Quick Start

### 1. Environment Setup

Create a conda environment (this repo is developed against Python 3.11):

```bash
conda create -n ai2_yam python=3.11 -y
conda activate ai2_yam
```

Then install each subproject's dependencies. Recommended order: `i2rt` first (sets up YAM motors), then `gello_software` (sets up the gello + main runtime), then `lerobot` (dataset conversion). Each subdirectory has its own README and requirements.

> Throughout the rest of this guide, the conda env is assumed to be active. The CAN channel names used below — `can_leader_l` (left arm) and `can_follower_r` (right arm) — match this workstation's setup. Confirm yours with `ip link show | grep can` and adjust if different; the channel name in `gello_software/configs/yam_left.yaml` / `yam_right.yaml` must agree with what `ip link show` reports.

### 2. Per-Boot Startup Sequence

Run these every time the YAM is power-cycled or replugged:

```bash
# Reset all CAN interfaces (brings them up at 1 Mbit/s)
bash i2rt/scripts/reset_all_can.sh

# Disable the motor timeout on both arms (default is too short for long sessions
# and causes abrupt collapse during teleop / data collection / eval)
python i2rt/i2rt/motor_config_tool/set_timeout.py --channel can_leader_l
python i2rt/i2rt/motor_config_tool/set_timeout.py --channel can_follower_r
```

### 3. Reset Gripper Zero (only for the Linear gripper at max grip)
The gripper does not auto-recalibrate on power-up — it treats its current position as zero (closed), which can lead to motor overheat. Close the gripper fully before powering on, then reset its zero position:

```bash
python i2rt/i2rt/motor_config_tool/set_zero.py --channel=can_leader_l --motor_id=7
python i2rt/i2rt/motor_config_tool/set_zero.py --channel=can_follower_r --motor_id=7
```

## Gello Configuration
All teleop / data collection / eval entry points live under `gello_software/`:

```bash
cd gello_software
```

- Left-arm config: `configs/yam_left.yaml`
- Right-arm config: `configs/yam_right.yaml`

The shipped files are working samples for this workstation. To generate your own from scratch, follow the instructions in `gello_software/README.md`. Note that the auto-generated configs only contain the `robot` block — you must copy the rest (`agent`, `sensors`, `collection`, `storage`, `lerobot`, `policy`, etc.) over manually.

## Cameras
The cameras we used are Intel Realsense cameras. If you are using the same cameras, simply change the ```device_id``` in the configuration file ```configs/yam_left.yaml``` to match the ones you are using.

## Teleoperation
Now, we have setup everything we need to run the robot!
To perform teleoperation, simply run
```
python experiments/launch_yaml.py --left_config_path=configs/yam_left.yaml --right_config_path=configs/yam_right.yaml
```

## Data Collection
To perform data collection for a task, update ```configs/yam_left.yaml```, mainly sections ```storage``` and ```lerobot```:
```bash
# Data storage configuration
storage:
  episodes: 30
  base_dir: "/home/sean/Desktop/YAM/gello_software/data"
  task_directory: "test"
  language_instruction: "test"
  teleop_device: "gello" # ["oculus", "keyboard", "gello", "none"]
  save_format: "json" # ["json", "npy"]
  old_format: false

# LeRobot conversion + upload pipeline
lerobot:
  auto_convert: true
  auto_upload: true
  hf_repo_id: "your_huggingface_user/your_dataset_name"
  delete_local_after_upload: true
  fps: 30
  robot_type: "Bimanual_YAM"
  skip_initial_frames: 0
  action_mode: "next_joint_fields" # ["next_joint_fields", "next_state", "copy_state"]
  sanitize_online_viz_meta: true
```
```storage.episodes``` is the maximum episode index to collect. The collection loop ends when this limit is reached.

```storage.base_dir``` is the location to store collected raw json episodes for all tasks.

```storage.task_directory``` is the subdirectory name for that task.

```storage.language_instruction``` is the instruction for the task written into collected data.

```lerobot.auto_convert``` controls whether post-collection conversion is run.

```lerobot.auto_upload``` controls whether converted data is uploaded to Hugging Face after conversion.
If upload is enabled, the script also tries to create dataset tag ```v3.0``` and skips tag creation if it already exists.

```lerobot.hf_repo_id``` is the destination Hugging Face dataset repo in the form ```username/dataset_name```.

```lerobot.delete_local_after_upload``` controls whether local raw json and local LeRobot output are deleted after successful upload to huggingface.

```lerobot.fps``` is the frame rate metadata written into the generated LeRobot dataset (set this to match your collection/control frequency).

```lerobot.robot_type``` sets the robot metadata field saved in the LeRobot dataset.

```lerobot.skip_initial_frames``` skips the first N frames of each episode during conversion (useful to remove startup transients).

```lerobot.action_mode``` controls how action is derived:
- ```next_joint_fields``` (recommended): use ```next_left_joint```/```next_right_joint``` from json.
- ```next_state```: use shifted joint state at t+1 as action.
- ```copy_state```: use current joint state at t as action.

```lerobot.sanitize_online_viz_meta``` removes quantile-only metadata columns after conversion to improve compatibility with some online visualizers.

Most likely you can keep most of the fields unchange and only need to update ```epsiode, base_dir, task_directory, language_instruction, hf_repo_id```. To perform data collection after configuration simply run:
```bash
python experiments/launch_yaml_collect_data.py --left_config_path=configs/yam_left.yaml --right_config_path=configs/yam_right.yaml
```

The program will launch a color pad to take keyboard input.

Press ```s``` to start collecting 1 episode of data.

Press ```a``` to end and save collected episode.

Press ```b``` to end and delete collected episode.

After all episodes are collected, the script runs post-collection pipeline based on config:
- if ```auto_convert: true``` and ```auto_upload: false```, it converts only.
- if ```auto_convert: true``` and ```auto_upload: true```, it converts, uploads, and tags.

If the LeRobot output directory already exists, it will ask:
```bash
Do you want to remove it and continue? (y/n)
```
Type ```y``` to remove and continue, or ```n``` to cancel the post-collection pipeline.

Important: pressing ```ctrl+c``` exits early and only performs robot/socket cleanup.
It does **not** run the convert/upload/tag pipeline.

Note: make sure you are on the color pad so it can take in the keyboard input (don't put it in the background).
To kill the program with ```ctrl+c```, you will need to be on your IDE or Terminal.

##
### Data Converstion
Manual conversion is still available if needed. Data is saved in json format (same as MolmoAct-v1) and can be converted with ```molmoact_to_lerobot_v30.py```.

By default, the script loads parameters from ```gello_software/configs/yam_left.yaml```:
- ```data_dir = storage.base_dir / storage.task_directory```
- ```output_dir = storage.base_dir / storage.task_directory + "_lerobot_v30"```
- upload behavior is decided by ```lerobot.auto_upload```

You can also define the dir name yourself. See the ```molmoact_to_lerobot_v30.py``` for more details.
##
Field definitions used by conversion/upload in ```gello_software/configs/yam_left.yaml```:

```storage``` fields:
- ```base_dir```: root directory where collected json episodes are stored.
- ```task_directory```: task subfolder under ```base_dir``` (also used to derive output directory name).
- ```language_instruction```: default task instruction written into converted LeRobot episodes.

```lerobot``` fields:
- ```auto_convert```: enables post-collection conversion in the launcher pipeline.
- ```auto_upload```: if true, converted data is uploaded to Hugging Face.
- ```hf_repo_id```: target Hugging Face dataset repo (format: ```username/dataset_name```).
- ```delete_local_after_upload```: if true, remove local json + local LeRobot folder after successful upload/tag.
- ```fps```: frame rate metadata saved into the LeRobot dataset.
- ```robot_type```: robot metadata string saved into the LeRobot dataset.
- ```skip_initial_frames```: number of initial frames to skip per episode during conversion.
- ```action_mode```: how action is derived (```next_joint_fields```, ```next_state```, ```copy_state```).
- ```sanitize_online_viz_meta```: removes quantile-only metadata columns for better online visualizer compatibility.

So if your config is already set, you can run:
```bash
python molmoact_to_lerobot_v30.py
```

You can still override parameters manually:
```bash
python molmoact_to_lerobot_v30.py \
        --data_dir /path/to/molmoact \
        --output_dir /path/to/molmoact_lerobot_v30 \
        --repo_id your_huggingface_user/molmoact_v30 \
        --fps 30 \
        --upload_to_hf 1
```

When upload is enabled, the script uploads to Hugging Face and then adds tag ```v3.0``` automatically.
If the tag already exists, it skips creating the duplicate tag.


## Model Evaluation
Current evaluation supports two policy types in ```experiments/launch_yaml_eval.py```:
- ```dp``` (DiffusionPolicy)
- ```pi05``` (PI05Policy)

Set these fields in ```configs/yam_left.yaml``` under ```policy```:
```bash
# Policy configuration
policy:
  type: "dp"   # ["dp", "pi05"]
  checkpoint_path: "your_model_repo_or_local_checkpoint_path"
```

```policy.type``` selects which evaluator path is used:
- ```dp```: runs ```run_control_loop_eval``` with diffusion policy.
- ```pi05```: runs ```run_control_loop_eval_pi``` with PI05 chunked actions.

```policy.checkpoint_path``` is the policy checkpoint source (HF model id or local checkpoint path).

For ```pi05```, task instruction is taken from ```storage.language_instruction```.

After configuring policy, run:
```bash
python experiments/launch_yaml_eval.py --left_config_path=configs/yam_left.yaml --right_config_path=configs/yam_right.yaml
```

Notes: ```preprocess_observation``` in ```experiments/launch_yaml_eval.py``` converts robot observations to model inputs for the ```dp``` path. Make sure image size and camera mapping matches your model's expected input format.
```bash
# Define the target image size
TARGET_HEIGHT = 256
TARGET_WIDTH = 342

# Map cameras "observation : model"
camera_mapping = {"left_camera_rgb": 'left', "right_camera_rgb": 'right', "front_camera_rgb": 'front'}
```

## Model Evaluation for MolmoAct2
Current evaluation supports remote inference only. The MolmoAct2 model should be hosted in a remote server.

Update the server url in ```experiments/molmoact.py``` (line 13). The default task instruction lives in ```storage.language_instruction``` in ```configs/yam_left.yaml``` and is offered as the first-rollout default; you can override it interactively per rollout.

Run N rollouts in a single session:
```bash
python experiments/launch_yaml_eval_molmoact.py \
    --left_config_path=configs/yam_left.yaml \
    --right_config_path=configs/yam_right.yaml \
    -n 10
```

### Per-rollout flow
1. A stdin prompt asks for the task instruction for this rollout. Press Enter to reuse the previous rollout's instruction; the first rollout defaults to ```storage.language_instruction```.
2. A live cv2 window opens showing the three policy-input frames (left | front | right) side-by-side with a header showing the current rollout/step/instruction.
3. The rollout ends on one of:
   - Keypress ```y``` in the cv2 window → label **success** (moved to ```{base_dir}/{task_directory}/success/{YYYY-MM-DD}/{timestamp}/```).
   - Keypress ```n``` → label **failure** (moved to ```failure/{YYYY-MM-DD}/{timestamp}/```).
   - Keypress ```q``` → quit rollout, no label (stays in ```eval/{timestamp}/```).
   - ```max_steps``` reached → terminal prompts ```Label rollout (y / n / Enter)``` for a DROID-style label; Enter keeps it in ```eval/```.

### Saved per rollout
Each rollout directory contains:
- ```left_rgb/```, ```front_rgb/```, ```right_rgb/``` — one PNG per control step.
- ```episode.h5``` — joint trajectory (```state```, ```next_state```) + language instruction attribute.
- ```err.md``` — written only if the rollout was incomplete (e.g. Ctrl-C).

Directory layout under ```{base_dir}/{task_directory}/```:
```
eval/{timestamp}/                  # unlabeled or incomplete rollouts
success/{YYYY-MM-DD}/{timestamp}/
failure/{YYYY-MM-DD}/{timestamp}/
eval_lerobot_v30/{session_ts}/     # LeRobot v3.0 dataset for this session
```

### End of session
After the N rollouts finish — or on ```ctrl+c```, in which case the partial rollout is saved to ```eval/``` with an ```err.md``` marker — labeled rollouts (success + failure) from this session are batch-converted into a single LeRobot v3.0 dataset under ```eval_lerobot_v30/{session_ts}/```. Conversion reuses the existing ```lerobot:``` config block (```fps```, ```robot_type```, ```action_mode```, ```vcodec```, ```hf_repo_id```, etc.). Raw rollouts are kept on disk regardless of conversion outcome, so you can re-run ```python molmoact_to_lerobot_v30.py``` later if needed.

### Optional eval config
```yaml
eval:
  live_view_enabled: true  # show the 3-pane cv2 window during rollouts (default true)
```
```max_steps``` (default 1000) is read from the top-level config field; the ```lerobot:``` block is reused as-is.

## Other tools for testing
Within the ```gello_software/experiments```directory, we have:
- ```launch_yaml_replay```: this takes in a episode (json format) and replay the robot control and images collected.
- ```launch_yaml_open_loop```: this takes in a episode (json format) and used the collected image combined with the real-time robot motion as input to the dp/pi05 model.
- ```launch_yaml_molmoact_open_loop```: this takes in a episode (json format) and used the collected image combined with the real-time robot motion as input to the MolmoAct-v2 model.

These scripts can be used to verified that the data collection pipeline is setup correctly, and if the model has been trained correctly.