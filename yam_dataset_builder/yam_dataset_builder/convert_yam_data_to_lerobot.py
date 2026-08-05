"""
Convert YAM bimanual raw data to LeRobot dataset v2.0 format for openpi fine-tuning.

Optimized: uses hardlinks + ffmpeg for video encoding (no PIL decode/re-encode).

Raw data structure (per episode):
  - rgb/{head,left_wrist,right_wrist}/XXXXXXXXXX.jpg   (1280x720 RGB images)
  - lowdim/{head,left_wrist,right_wrist}/XXXXXXXXXX.npz (joints: 14D, action: 26D cartesian)
  - metadata.json

Output LeRobot format:
  - observation.state: 14D joint positions [left_joint(6), left_grip(1), right_joint(6), right_grip(1)]
  - action: 14D joint positions (absolute target = joints[t+1])
  - observation.images.{head, left_wrist, right_wrist}: RGB video

Usage:
  uv run examples/yam/convert_yam_data_to_lerobot.py \
      --args.raw-dir /home/yuzhi/dataset/place_lock_simple_raw \
      --args.repo-id yuzhi/yam_place_lock_simple
"""

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import (
    DEFAULT_IMAGE_PATH,
    LeRobotDataset,
)
import numpy as np
import torch
import tqdm
import tyro


CAMERAS = ["scene_camera", "left_wrist_camera", "right_wrist_camera"]
FPS = 30

# 14D: [left_joint(6), left_grip(1), right_joint(6), right_grip(1)]
MOTORS = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_gripper",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_gripper",
]


@dataclasses.dataclass(frozen=True)
class Args:
    raw_dir: Path
    """Path to the raw YAM dataset directory."""
    repo_id: str
    """LeRobot repo ID (e.g., yuzhi/yam_place_lock_simple)."""
    push_to_hub: bool = False
    """Whether to push the dataset to HuggingFace Hub."""
    private: bool = True
    """Whether to create a private repo on HuggingFace Hub."""
    overwrite: bool = False
    """Whether to overwrite the existing dataset."""


def load_lowdim_bulk(ep_dir: Path, num_frames: int) -> np.ndarray:
    """Bulk-load all lowdim joints with threaded I/O. Returns (N, 14) array."""
    import pickle

    lowdim_dir = ep_dir / "lowdim"
    all_joints = np.empty((num_frames, 14), dtype=np.float32)

    def _load_one(i: int) -> None:
        #data = np.load(lowdim_dir / f"{i:010d}.npz", allow_pickle=True)
        #with open(lowdim_dir / f"{i:010d}.pkl", "rb") as f:
        #    data = pickle.load(f)
        #
        #all_joints[i] = data["joints"]
        fpath = lowdim_dir / f"{i:010d}.pkl"
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        joints = data["joints"]
        try:
            all_joints[i] = np.asarray(joints).flatten()[:14]
        except (ValueError, TypeError) as e:
            print(f"ERROR loading {fpath}:")
            print(f"  type={type(joints)}, len={len(joints) if hasattr(joints, '__len__') else 'N/A'}")
            print(f"  value={joints}")
            print(f"  np.asarray shape={np.asarray(joints).shape}")
            raise RuntimeError(f"Failed to load joints from {fpath}: {e}") from e


    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_load_one, range(num_frames)))

    return all_joints


def hardlink_frames(src_dir: Path, dst_dir: Path, num_frames: int) -> None:
    """Hardlink raw Framess to temp directory for ffmpeg (near-instant, zero-copy)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        #src = src_dir / f"{i:010d}.jpg"
        #dst = dst_dir / f"frame_{i:06d}.jpg"
        src = src_dir / f"{i:010d}.png"
        if not src.exists():
            src = src_dir / f"{i:010d}.jpg"
        dst = dst_dir / f"frame_{i:06d}.png"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)


def encode_video_ffmpeg(imgs_dir: Path, video_path: Path, num_frames: int, fps: int) -> None:
    """Encode JPEG frames to H.264 MP4 with ffmpeg directly."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(imgs_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "22",
        "-frames:v", str(num_frames),
        "-loglevel", "error",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed encoding {video_path}:\n{result.stderr}")


def process_episode(
    dataset: LeRobotDataset,
    ep_dir: Path,
    output_dir: Path,
) -> None:
    """Process one episode: bulk load lowdim, hardlink images, ffmpeg encode, save."""
    with open(ep_dir / "metadata.json") as f:
        metadata = json.load(f)

    num_frames_raw = metadata["num_frames"]
    task = metadata["language"]["prompt"][0]

    # --- Bulk load lowdim (threaded I/O) ---
    joints = load_lowdim_bulk(ep_dir, num_frames_raw)
    # action[t] = joints[t+1], drop last frame
    actions = joints[1:]
    joints = joints[:-1]
    num_frames = num_frames_raw - 1

    episode_index = dataset.meta.total_episodes
    tmp_frames_root = output_dir / "_tmp_frames"

    # --- Hardlink raw JPEGs for ffmpeg (parallel across cameras) ---
    def _hardlink_cam(cam: str) -> None:
        img_key = f"observation.images.{cam}"
        dst_dir = tmp_frames_root / img_key / f"episode_{episode_index:06d}"
        hardlink_frames(ep_dir / "rgb" / cam, dst_dir, num_frames)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(_hardlink_cam, CAMERAS))

    # --- Encode videos with ffmpeg (parallel across cameras) ---
    def _encode_cam(cam: str) -> None:
        img_key = f"observation.images.{cam}"
        imgs_dir = tmp_frames_root / img_key / f"episode_{episode_index:06d}"
        video_path = output_dir / dataset.meta.get_video_file_path(episode_index, img_key)
        encode_video_ffmpeg(imgs_dir, video_path, num_frames, FPS)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(_encode_cam, CAMERAS))

    # --- Build episode buffer manually (bypass add_frame to skip image I/O) ---
    ep_buffer = dataset.create_episode_buffer(episode_index)
    ep_buffer["size"] = num_frames
    ep_buffer["task"] = [task] * num_frames
    ep_buffer["frame_index"] = list(range(num_frames))
    ep_buffer["timestamp"] = [i / FPS for i in range(num_frames)]
    ep_buffer["observation.state"] = [joints[i] for i in range(num_frames)]
    ep_buffer["action"] = [actions[i] for i in range(num_frames)]

    # Video keys: store the hardlinked JPEG paths (LeRobot samples these for stats)
    for cam in CAMERAS:
        img_key = f"observation.images.{cam}"
        imgs_dir = tmp_frames_root / img_key / f"episode_{episode_index:06d}"
        ep_buffer[img_key] = [str(imgs_dir / f"frame_{i:06d}.png") for i in range(num_frames)]

    dataset.episode_buffer = ep_buffer
    dataset.save_episode()

    # Clean up temp frames for this episode
    for cam in CAMERAS:
        img_key = f"observation.images.{cam}"
        ep_frames_dir = tmp_frames_root / img_key / f"episode_{episode_index:06d}"
        if ep_frames_dir.exists():
            shutil.rmtree(ep_frames_dir)


def main(args: Args):
    raw_dir = args.raw_dir
    repo_id = args.repo_id

    ep_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f"Found {len(ep_dirs)} episodes in {raw_dir}")

    output_dir = HF_LEROBOT_HOME / repo_id
    if output_dir.exists():
        print(f"Output dir {output_dir} already exists. Use --args.overwrite to delete and restart.")                                                                                                                
        if args.overwrite:                                                                                                                                                                                           
            shutil.rmtree(output_dir)                                                                                                                                                                                
        else:                                                                                                                                                                                                        
            print("Exiting. Remove the directory manually or pass --args.overwrite.")                                                                                                                                
            sys.exit(1)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, 720, 1280),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="yam_bimanual",
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    skipped: list[tuple[str, str]] = []
    for ep_dir in tqdm.tqdm(ep_dirs, desc="Converting episodes"):
        try:
            process_episode(dataset, ep_dir, output_dir)
        except Exception as e:
            print(f"\nSKIPPING {ep_dir.name}: {e}")
            skipped.append((ep_dir.name, str(e)))

    if skipped:
        print(f"\n{'='*60}")
        print(f"Skipped {len(skipped)}/{len(ep_dirs)} episodes:")
        for name, err in skipped:
            print(f"  {name}: {err}")

    # Clean up any remaining temp dirs
    tmp_frames_root = output_dir / "_tmp_frames"
    if tmp_frames_root.exists():
        shutil.rmtree(tmp_frames_root)

    print(f"Dataset saved to {output_dir}")
    print(f"Total episodes: {dataset.num_episodes}, Total frames: {dataset.num_frames}")

    if args.push_to_hub:
        dataset.push_to_hub(private=args.private)
        print(f"Pushed to HuggingFace Hub: {repo_id} (private={args.private})")


if __name__ == "__main__":
    tyro.cli(main)
