from policy_base import PolicyBase
from typing import List, Optional
import numpy as np
import time
import requests
import json_numpy
json_numpy.patch()
from gello.utils.logging_utils import get_molmoact_logger

DEFAULT_SERVER = "https://unachievable-tawana-subtransparent.ngrok-free.dev"


def _normalize_server_url(server: Optional[str]) -> str:
    """Accept ngrok URLs (``https://...``), bare IPs (``10.0.0.5``), or
    ``host:port`` strings, and return a full ``http(s)://host[:port]/act`` URL.

    - If empty/None, falls back to ``DEFAULT_SERVER``.
    - If no scheme is provided, ``http://`` is prepended (suitable for LAN IPs).
    - Trailing ``/act`` is appended unless the input already ends in ``/act``.
    """
    s = (server or DEFAULT_SERVER).strip().rstrip("/")
    if "://" not in s:
        s = "http://" + s
    if not s.endswith("/act"):
        s = s + "/act"
    return s


class MolmoAct(PolicyBase):
    def __init__(self, server: Optional[str] = None):
        self.logger = get_molmoact_logger()
        self.url = _normalize_server_url(server)
        self.multi_views = True
        self.action_horizon = 25

        # Log configuration
        self.logger.info(f"MolmoAct initialized with URL: {self.url}")
        self.logger.info(f"Multi-views enabled: {self.multi_views}")
        self.logger.info(f"Action horizon: {self.action_horizon}")

    def get_action_horizon(self):
        return self.action_horizon

    def prepare_input(self, obs, instruction):
        self.logger.info("Preparing input for MolmoAct inference")
        self.logger.info(f"Instruction: '{instruction}'")
        # self.logger.info(f"Camera keys - {obs['left_camera_rgb']}, {obs['front_camera_rgb']}, {obs['right_camera_rgb']}")
        self.logger.info(f"State: {obs['joint_positions']}")

        try:
            # Log image information
            if hasattr(obs['left_camera_rgb'], 'shape'):
                self.logger.info(f"Left image shape: {obs['left_camera_rgb'].shape}")
            if hasattr(obs['front_camera_rgb'], 'shape'):
                self.logger.info(f"Front image shape: {obs['front_camera_rgb'].shape}")
            if hasattr(obs['right_camera_rgb'], 'shape'):
                self.logger.info(f"Right image shape: {obs['right_camera_rgb'].shape}")

            input_dict = {
                "left_camera_rgb": obs["left_camera_rgb"],
                "front_camera_rgb": obs["front_camera_rgb"],
                "right_camera_rgb": obs["right_camera_rgb"],
                "instruction": instruction,
                "state": obs["joint_positions"]
            }

            self.logger.info("Input preparation completed successfully")
            return input_dict

        except KeyError as e:
            self.logger.error(f"Missing camera key in observation: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error preparing input: {e}")
            raise

    def inference(self, input_dict):
        self.logger.info("Starting MolmoAct inference")

        try:
            images = [input_dict["left_camera_rgb"], input_dict["front_camera_rgb"], input_dict["right_camera_rgb"]]
            lang = input_dict["instruction"]
            state = input_dict["state"]

            self.logger.info(f"Processing instruction: '{lang}'")
            self.logger.info(f"Number of images: {len(images)}")
            self.logger.info(f"Number of joints: {len(state)}")

            start_time = time.time()
            response = self.send_request(images, lang, state, self.url)
            request_time = time.time() - start_time

            self.logger.info(f"Server request completed in {request_time:.3f}s")
            self.logger.info(f"Raw actions received: {len(response['actions'])} actions")

            # processed_actions = self.prepare_output(actions)
            # self.logger.info(f"Processed {len(processed_actions)} actions")

            return response

        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            raise

    # def prepare_output(self, raw_actions):
    #     self.logger.info("Preparing output actions")

    #     try:
    #         result_actions = raw_actions.copy()

    #         if self.invert_gripper:
    #             self.logger.info("Applying gripper inversion")
    #             for i in range(len(raw_actions)):
    #                 action = raw_actions[i]
    #                 result_actions[i] = invert_gripper(action)
    #                 self.logger.debug(f"Action {i}: gripper value inverted")
    #         else:
    #             self.logger.info("No gripper inversion applied")

    #         self.logger.info(f"Output preparation completed: {len(result_actions)} actions")
    #         return result_actions

    #     except Exception as e:
    #         self.logger.error(f"Error preparing output: {e}")
    #         raise

    def send_request(self, images: List[np.ndarray], instruction: str, state: list, server_url: str):
        """
        Send the captured image and instruction to the inference server using json_numpy.
        Returns the action output as received from the server.
        """
        self.logger.info(f"Sending request to server: {server_url}")

        try:
            if not self.multi_views:
                self.logger.info("Using single view mode")
                # Convert PIL image to a NumPy array
                image_np = np.array(images[0])
                self.logger.info(f"Single image shape: {image_np.shape}")

                # Prepare the payload with the image and instruction from the script
                payload = {
                    "image": image_np, # scene cam
                    "instruction": instruction,
                    "state" : state
                }
            else:
                self.logger.info("Using multi-view mode")
                # Convert PIL image to a NumPy array
                left_img_np = np.array(images[0])
                front_img_np = np.array(images[1])
                right_img_np = np.array(images[2])

                self.logger.info(f"Left image shape: {left_img_np.shape}")
                self.logger.info(f"Front image shape: {front_img_np.shape}")
                self.logger.info(f"Right image shape: {right_img_np.shape}")

                # Prepare the payload with the image and instruction from the script
                payload = {
                    "left_cam": left_img_np,
                    "top_cam": front_img_np,
                    "right_cam": right_img_np,
                    "timestamp": time.time(), # add timestamp for debugging
                    "instruction": instruction,
                    "state": state,
                    "normalization_tag": "yam_dual_molmoact2"
                }

            self.logger.info("Preparing HTTP request")
            headers = {"Content-Type": "application/json"}

            # Serialize payload
            start_time = time.time()
            serialized_payload = json_numpy.dumps(payload)
            serialize_time = time.time() - start_time
            self.logger.info(f"Payload serialized in {serialize_time:.3f}s")

            # Send request
            start_time = time.time()
            response = requests.post(server_url, headers=headers, data=serialized_payload)
            request_time = time.time() - start_time

            self.logger.info(f"HTTP request completed in {request_time:.3f}s")
            self.logger.info(f"Response status code: {response.status_code}")

            if response.status_code != 200:
                error_msg = f"Server error: {response.text}"
                self.logger.error(error_msg)
                raise Exception(error_msg)

            # Parse response
            start_time = time.time()
            response_data = response.json()
            parse_time = time.time() - start_time
            self.logger.info(f"Response parsed in {parse_time:.3f}s")

            self.logger.info("Server request completed successfully")
            return response_data

        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error to server {server_url}: {e}")
            raise
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Request timeout to server {server_url}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error to server {server_url}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during server request: {e}")
            raise