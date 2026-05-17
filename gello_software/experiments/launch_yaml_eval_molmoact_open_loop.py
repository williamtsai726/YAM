import atexit
from math import inf
from multiprocessing import Process
import os
import signal
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

from PIL import Image
from PIL.Image import logger
from molmoact import MolmoAct
import torch
import tyro
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict, move_to_start_position
from gello.dynamixel.driver import DynamixelDriver
import numpy as np

from gello.data_utils.data_replay import DataReplayer
from gello.env import RobotEnv
from gello.utils.logging_utils import log_collect_demos
DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"

# Global variables for cleanup
cleanup_in_progress = False

_env = None
_bimanual = False
_left_cfg = None
_right_cfg = None


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("Cleaning up resources...")
    if _bimanual:
        move_to_start_position(_env, _bimanual, _left_cfg, _right_cfg)
    else:
        move_to_start_position(_env, _bimanual, _left_cfg)

    print("Cleanup completed.")


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    # use_save_interface: bool = False
    # """Enable saving data with keyboard interface."""


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    cleanup()
    import os

    os._exit(0)


def main():
    # Register cleanup handlers
    # If terminated without cleanup, can leave ZMQ sockets bound causing "address in use" errors or resource leaks

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    bimanual = args.right_config_path is not None

    # Load configs
    left_cfg = OmegaConf.to_container(
        OmegaConf.load(args.left_config_path), resolve=True
    )
    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )

    # Create robot(s)
    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )

    left_robot = instantiate_from_dict(left_robot_cfg)

    if bimanual:
        from gello.robots.robot import BimanualRobot

        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )

        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)

        # For bimanual, use the left config for general settings (hz, etc.)
        cfg = left_cfg
    else:
        robot = left_robot
        cfg = left_cfg

    env = RobotEnv(robot, control_rate_hz=cfg.get("hz", 30))

    # Store global variables for cleanup
    global _env, _bimanual, _left_cfg, _right_cfg
    _env = env
    _bimanual = bimanual
    _left_cfg = left_cfg
    _right_cfg = right_cfg if bimanual else None

    # Move robot to start_joints position if specified in config
    from gello.utils.launch_utils import move_to_start_position

    if bimanual:
        move_to_start_position(env, bimanual, left_cfg, right_cfg)
    else:
        move_to_start_position(env, bimanual, left_cfg)

    print(
        f"Launching robot: {robot.__class__.__name__}"
    )
    print(f"Control loop: {cfg.get('hz', 30)} Hz")

    logger.info("Start open loop evaluation...")
    task = input("Enter task to replay: ")
    episode_number = int(input("Enter episode number to replay: "))

    data_replayer = DataReplayer(save_format=left_cfg['storage']['save_format'], old_format=left_cfg['storage']['old_format'])
    data_replayer.load_episode(left_cfg['storage']['base_dir'] + '/' + task, episode_number)
    
    molmoact = MolmoAct(server=(left_cfg.get("eval") or {}).get("molmoact_server"))
    if bimanual:
        run_control_loop_eval_open_loop(env, policy=molmoact, data_replayer=data_replayer, instruction=left_cfg["storage"]["language_instruction"])
    else:
        run_control_loop_eval_open_loop(env, policy=molmoact, data_replayer=data_replayer, instruction=left_cfg["storage"]["language_instruction"])

def run_control_loop_eval_open_loop(
    env: RobotEnv,
    policy: MolmoAct = None,
    data_replayer: DataReplayer = None,
    instruction: str = None,
) -> None:
    """Run the main control loop.
    """
    logger.info("Starting policy inference...")
    # Init environment and warm up agent
    obs = env.get_obs()

    # Main control loop
    demo_length = data_replayer.get_demo_length()
    obs_index = 0
    while obs_index < demo_length:
        obs = data_replayer.get_observation(obs_index)
        input_dict = {
            "left_camera_rgb": obs["image_left_rgb"],
            "front_camera_rgb": obs["image_front_rgb"],
            "right_camera_rgb": obs["image_right_rgb"],
            "instruction": instruction,
            "state": np.concatenate([obs["left_joint"], obs["right_joint"]])
        }
        log_collect_demos("Running policy inference...", "info")
        start_time = time.time()
        actions = policy.inference(input_dict)["actions"]
        inference_time = time.time() - start_time
        log_collect_demos(f"Policy inference completed in {inference_time:.3f}s", "success")
        log_collect_demos(f"Generated {len(actions)} action(s)", "data_info")
        for i in range(len(actions)):
            obs = smooth_move_while_inference_envstep(env, np.array(actions[i]))
        obs_index += 8
    logger.info("Finished policy inference")

def smooth_move_while_inference_envstep(env: RobotEnv, action):
    current_joint = env.get_obs()["joint_positions"]
    target_joint = action

    steps = 5
    obs = None
    for i in range(steps + 1):
        alpha = i / steps  # Interpolation factor
        interpolated_joint = (1 - alpha) * current_joint + alpha * target_joint  # Linear interpolation
        obs = env.step(interpolated_joint)
        time.sleep(0.5 / steps)

    return obs

if __name__ == "__main__":
    main()
