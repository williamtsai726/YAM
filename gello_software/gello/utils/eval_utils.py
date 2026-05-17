"""Helpers for the eval system: per-rollout saving, live multi-camera viewer,
DROID-style labeling, end-of-session LeRobot conversion.

Each class/function is independent; the launch script wires them together.
"""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import h5py
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Per-rollout saver
# ---------------------------------------------------------------------------


def _save_png(image: np.ndarray, path: Path, compress_level: int) -> None:
    Image.fromarray(image).save(path, compress_level=compress_level)


class EvalRolloutSaver:
    """Buffer one rollout's frames in RAM, then flush to PNG + HDF5 on disk.

    Layout per rollout::

        rollout_dir/
        ├── left_rgb/{frame:06d}.png
        ├── right_rgb/{frame:06d}.png
        ├── front_rgb/{frame:06d}.png
        ├── episode.h5
        └── err.md          # only if write_err() is called

    The HDF5 file holds the joint trajectory (state, next_state) and the
    language instruction; the PNGs hold the per-frame RGB images. The DROID
    layout converter (``load_droid_layout_data``) walks this structure.
    """

    CAMERA_OBS_TO_KEY = {
        "left_camera_rgb": "left_rgb",
        "right_camera_rgb": "right_rgb",
        "front_camera_rgb": "front_rgb",
    }

    def __init__(
        self,
        rollout_dir: Path,
        instruction: str,
        max_workers: int = 2,
        png_compress_level: int = 1,
    ) -> None:
        self.rollout_dir = Path(rollout_dir)
        self.instruction = instruction
        self.max_workers = max(1, int(max_workers))
        self.png_compress_level = max(0, min(9, int(png_compress_level)))

        if self.rollout_dir.exists():
            raise FileExistsError(
                f"Rollout dir already exists: {self.rollout_dir}. "
                "Timestamps are expected to be unique."
            )
        self.rollout_dir.mkdir(parents=True)

        self._buffer: List[Dict[str, Any]] = []

    @property
    def num_steps(self) -> int:
        return len(self._buffer)

    def add_step(
        self,
        obs_pre: Dict[str, Any],
        obs_post: Dict[str, Any],
    ) -> None:
        """Buffer one control-step record.

        ``obs_pre`` is the observation snapshot at the start of the step
        (the image the policy "sees" for that step) and ``obs_post`` is the
        observation after applying the action. ``next_state`` mirrors the
        data-collection convention used in ``DataSaver``: the observed post-step
        joint positions, not the commanded action.
        """
        record: Dict[str, Any] = {
            "state": np.asarray(obs_pre["joint_positions"], dtype=np.float32).copy(),
            "next_state": np.asarray(obs_post["joint_positions"], dtype=np.float32).copy(),
        }
        for obs_key, cam_key in self.CAMERA_OBS_TO_KEY.items():
            img = obs_pre.get(obs_key)
            if img is not None:
                record[cam_key] = np.ascontiguousarray(img).copy()
        self._buffer.append(record)

    def flush(self) -> None:
        """Write buffered PNGs and ``episode.h5`` to ``rollout_dir``."""
        if not self._buffer:
            logger.warning("Empty buffer at %s; nothing to flush.", self.rollout_dir)
            return

        cam_keys_present = sorted(
            k for k in self.CAMERA_OBS_TO_KEY.values() if k in self._buffer[0]
        )
        for cam_key in cam_keys_present:
            (self.rollout_dir / cam_key).mkdir(exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futures = []
            for i, rec in enumerate(self._buffer):
                for cam_key in cam_keys_present:
                    img_path = self.rollout_dir / cam_key / f"{i:06d}.png"
                    futures.append(
                        exe.submit(_save_png, rec[cam_key], img_path, self.png_compress_level)
                    )
            for fut in futures:
                fut.result()

        states = np.stack([rec["state"] for rec in self._buffer]).astype(np.float32)
        next_states = np.stack([rec["next_state"] for rec in self._buffer]).astype(np.float32)
        cam_names_stripped = [k.replace("_rgb", "") for k in cam_keys_present]

        h5_path = self.rollout_dir / "episode.h5"
        with h5py.File(h5_path, "w") as f:
            f.attrs["language_instruction"] = self.instruction
            f.attrs["num_steps"] = len(self._buffer)
            f.attrs["camera_names"] = np.array(
                cam_names_stripped, dtype=h5py.string_dtype()
            )
            f.create_dataset("state", data=states, compression="gzip", compression_opts=4)
            f.create_dataset(
                "next_state", data=next_states, compression="gzip", compression_opts=4
            )

        logger.info(
            "Saved rollout: %s (%d steps, cameras=%s)",
            self.rollout_dir,
            len(self._buffer),
            cam_names_stripped,
        )

    def write_err(self, reason: str, step: int) -> None:
        """Drop a marker file explaining why this rollout is incomplete."""
        err_path = self.rollout_dir / "err.md"
        with open(err_path, "w") as f:
            f.write("# Incomplete rollout\n\n")
            f.write(f"- Reason: {reason}\n")
            f.write(f"- Step at interruption: {step}\n")
            f.write(f"- Steps actually saved: {self.num_steps}\n")
            f.write(f"- Instruction: {self.instruction}\n")
            f.write(f"- Written at: {datetime.now().isoformat(timespec='seconds')}\n")


# ---------------------------------------------------------------------------
# Live cv2 viewer
# ---------------------------------------------------------------------------


class LiveCameraView:
    """Single cv2 window showing the three policy-input frames hconcat'd, with
    a text header and key polling.

    ``update()`` returns the lowercase key character ``'y' | 'n' | 'q'`` if one
    of those is pressed, otherwise ``None``. Other keys are ignored.
    """

    WINDOW_NAME = "YAM Eval"
    OBS_KEYS = ("left_camera_rgb", "front_camera_rgb", "right_camera_rgb")
    OBS_LABELS = ("LEFT", "FRONT", "RIGHT")
    # Window grows 2x in each linear dimension on first frame -> 4x screen area.
    SCALE = 2

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._initialized = False
        self._sized = False

    def _ensure_window(self) -> None:
        if self._initialized or not self.enabled:
            return
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        self._initialized = True

    def update(
        self,
        obs: Dict[str, Any],
        rollout_idx: int,
        num_rollouts: int,
        step: int,
        max_steps: int,
        instruction: str,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        self._ensure_window()

        panes: List[np.ndarray] = []
        for obs_key, label in zip(self.OBS_KEYS, self.OBS_LABELS):
            rgb = obs.get(obs_key)
            if rgb is None:
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            cv2.putText(
                bgr, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
            )
            panes.append(bgr)

        if not panes:
            return None

        # Pad panes to the same height before hconcat (defensive — they should match).
        max_h = max(p.shape[0] for p in panes)
        padded = [
            np.pad(p, ((0, max_h - p.shape[0]), (0, 0), (0, 0))) if p.shape[0] < max_h else p
            for p in panes
        ]
        canvas = cv2.hconcat(padded)

        header_h = 90
        header = np.zeros((header_h, canvas.shape[1], 3), dtype=np.uint8)
        lines = [
            f"Rollout {rollout_idx + 1}/{num_rollouts}    Step {step}/{max_steps}",
            f"Instruction: {instruction}",
            "Keys:  y = success    n = failure    q = quit rollout (saves as eval)",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                header, line, (10, 24 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )

        final = cv2.vconcat([header, canvas])
        cv2.imshow(self.WINDOW_NAME, final)
        if not self._sized:
            h, w = final.shape[:2]
            cv2.resizeWindow(self.WINDOW_NAME, w * self.SCALE, h * self.SCALE)
            self._sized = True
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("y"), ord("n"), ord("q")):
            return chr(key)
        return None

    def close(self) -> None:
        if self._initialized:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
            except cv2.error:
                pass
            self._initialized = False


# ---------------------------------------------------------------------------
# Interactive prompts (stdin)
# ---------------------------------------------------------------------------


def prompt_instruction(rollout_idx: int, num_rollouts: int, last_prompt: str) -> str:
    """Ask user for the task instruction for this rollout.

    Empty input → reuse ``last_prompt``.
    """
    text = input(
        f"\n[rollout {rollout_idx + 1}/{num_rollouts}] "
        f"Task instruction (Enter to reuse '{last_prompt}'): "
    ).strip()
    return text if text else last_prompt


def prompt_label() -> Optional[str]:
    """DROID-style label prompt for timeout-ended rollouts.

    Returns ``"success"`` / ``"failure"`` / ``None`` (keep as eval).
    Reprompts on any other input.
    """
    while True:
        text = input(
            "Label rollout (y = success / n = failure / Enter = keep as eval): "
        ).strip().lower()
        if text == "y":
            return "success"
        if text == "n":
            return "failure"
        if text == "":
            return None
        print("Invalid input. Type y, n, or press Enter.")


# ---------------------------------------------------------------------------
# Rollout end-result -> filesystem move
# ---------------------------------------------------------------------------


@dataclass
class RolloutOutcome:
    """The outcome of one rollout — feeds the labeling/move logic."""

    end_reason: str  # 'success' | 'failure' | 'quit' | 'timeout'
    last_step: int

    def implicit_label(self) -> Optional[str]:
        """Label inferred from end_reason; ``None`` means 'ask the user'."""
        if self.end_reason == "success":
            return "success"
        if self.end_reason == "failure":
            return "failure"
        return None  # 'quit' or 'timeout' need either a stay-in-eval or stdin prompt

    def keep_in_eval(self) -> bool:
        """Quit means user explicitly wanted no label."""
        return self.end_reason == "quit"


def resolve_label(outcome: RolloutOutcome) -> Optional[str]:
    """Decide where a rollout goes given its end reason.

    Returns the label (``"success"`` / ``"failure"``) or ``None`` for stay-in-eval.
    Timeout triggers the stdin prompt; quit is treated as no-label.
    """
    if outcome.keep_in_eval():
        return None
    implicit = outcome.implicit_label()
    if implicit is not None:
        return implicit
    # Timeout: ask the user.
    return prompt_label()


def move_rollout(rollout_dir: Path, label: str, base_save_dir: Path) -> Path:
    """Move ``rollout_dir`` (under ``base_save_dir/eval/``) to
    ``base_save_dir/{label}/{YYYY-MM-DD}/{name}/``. Returns the new path.
    """
    if label not in ("success", "failure"):
        raise ValueError(f"Unknown label: {label!r}")
    date_str = datetime.now().strftime("%Y-%m-%d")
    dest_parent = Path(base_save_dir) / label / date_str
    dest_parent.mkdir(parents=True, exist_ok=True)
    dest = dest_parent / rollout_dir.name
    if dest.exists():
        # Defensive: same timestamp twice shouldn't happen, but don't clobber.
        suffix = datetime.now().strftime("_%f")
        dest = dest_parent / (rollout_dir.name + suffix)
    shutil.move(str(rollout_dir), str(dest))
    return dest


# ---------------------------------------------------------------------------
# End-of-session LeRobot conversion
# ---------------------------------------------------------------------------


def convert_session_to_lerobot(
    session_rollout_dirs: Sequence[Path],
    output_dir: Path,
    fps: int,
    robot_type: str,
    repo_id: str = "local/eval_session",
    action_mode: str = "next_joint_fields",
    vcodec: str = "libsvtav1",
    sanitize_online_viz_meta: bool = True,
    image_writer_processes: int = 0,
    image_writer_threads: int = 0,
    parallel_encoding: bool = True,
) -> Optional[Path]:
    """Convert the labeled rollouts from this eval session into one LeRobot v3.0 dataset.

    Calls into the existing ``create_lerobot_dataset_v30`` so the dataset schema
    stays identical to the data-collection pipeline. Returns the final output
    path (may differ from ``output_dir`` if a uniqueness suffix was applied to
    avoid a non-empty-directory collision).
    """
    if not session_rollout_dirs:
        logger.info("No labeled rollouts to convert.")
        return None

    # Late import — the converter pulls in lerobot which is heavy.
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from molmoact_to_lerobot_v30 import (  # noqa: WPS433 — intentional late import
        create_lerobot_dataset_v30,
        load_droid_layout_data,
    )

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        suffix = datetime.now().strftime("_%H%M%S")
        output_dir = output_dir.parent / f"{output_dir.name}{suffix}"
        logger.warning("Output dir non-empty; using %s instead.", output_dir)

    episodes = load_droid_layout_data(
        base_dir=None,
        explicit_paths=[Path(p) for p in session_rollout_dirs],
    )
    if not episodes:
        logger.error("No usable episodes from this session — skipping conversion.")
        return None

    create_lerobot_dataset_v30(
        episodes=episodes,
        output_dir=str(output_dir),
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        action_mode=action_mode,
        sanitize_online_viz_meta=sanitize_online_viz_meta,
        vcodec=vcodec,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        parallel_encoding=parallel_encoding,
    )
    logger.info("LeRobot dataset written: %s", output_dir)
    return output_dir
