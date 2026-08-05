"""
Verify the converted LeRobot dataset against the original raw data.

Checks:
  1. Episode count matches
  2. Frame counts match per episode
  3. Joint state values match (raw lowdim vs parquet observation.state)
  4. Action values match (joints[t+1] vs parquet action)
  5. Video frames match raw RGB images (pixel-level comparison via decoding)

Usage:
  uv run examples/yam/verify_dataset.py \
      --args.raw-dir /home/yuzhi/dataset/place_lock_simple_raw \
      --args.repo-id yuzhi/yam_place_lock_simple
"""

import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tqdm
import tyro

CAMERAS = ["head", "left_wrist", "right_wrist"]


@dataclasses.dataclass(frozen=True)
class Args:
    raw_dir: Path
    """Path to the original raw YAM dataset directory."""
    repo_id: str
    """LeRobot repo ID used during conversion."""
    check_videos: bool = True
    """Whether to decode and compare video frames (slower)."""
    num_video_samples: int = 5
    """Number of frames per episode to sample for video comparison."""
    atol: float = 1e-5
    """Absolute tolerance for float comparison."""


def load_raw_joints(ep_dir: Path, num_frames: int) -> np.ndarray:
    """Load raw joints from lowdim NPZ files."""
    all_joints = np.empty((num_frames, 14), dtype=np.float32)

    def _load(i: int) -> None:
        data = np.load(ep_dir / f"lowdim/head/{i:010d}.npz", allow_pickle=True)
        all_joints[i] = data["joints"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_load, range(num_frames)))
    return all_joints


def verify_episode_lowdim(
    ep_dir: Path, parquet_path: Path, ep_idx: int, atol: float
) -> list[str]:
    """Verify state/action values for one episode. Returns list of error messages."""
    import pandas as pd

    errors = []
    with open(ep_dir / "metadata.json") as f:
        metadata = json.load(f)
    num_frames_raw = metadata["num_frames"]

    df = pd.read_parquet(parquet_path)
    num_frames_converted = len(df)

    # Converted dataset drops last frame (no next state for action)
    expected_frames = num_frames_raw - 1
    if num_frames_converted != expected_frames:
        errors.append(
            f"  Episode {ep_idx}: frame count mismatch: "
            f"expected {expected_frames} (raw {num_frames_raw} - 1), got {num_frames_converted}"
        )
        return errors

    raw_joints = load_raw_joints(ep_dir, num_frames_raw)

    # Check observation.state == joints[:-1]
    for i in range(num_frames_converted):
        converted_state = np.array(df["observation.state"].iloc[i], dtype=np.float32)
        raw_state = raw_joints[i]
        if not np.allclose(converted_state, raw_state, atol=atol):
            max_diff = np.max(np.abs(converted_state - raw_state))
            errors.append(
                f"  Episode {ep_idx}, frame {i}: state mismatch (max diff={max_diff:.8f})"
            )
            break  # report first mismatch only

    # Check action == joints[1:]  (action[t] = joints[t+1])
    for i in range(num_frames_converted):
        converted_action = np.array(df["action"].iloc[i], dtype=np.float32)
        raw_action = raw_joints[i + 1]
        if not np.allclose(converted_action, raw_action, atol=atol):
            max_diff = np.max(np.abs(converted_action - raw_action))
            errors.append(
                f"  Episode {ep_idx}, frame {i}: action mismatch (max diff={max_diff:.8f})"
            )
            break

    # Check task
    task_raw = metadata["language"]["prompt"][0]
    # task_index should be consistent (just check it exists)
    if "task_index" not in df.columns:
        errors.append(f"  Episode {ep_idx}: missing task_index column")

    return errors


def verify_episode_video(
    ep_dir: Path, video_dir: Path, ep_idx: int, num_samples: int
) -> list[str]:
    """Verify video frames against raw RGB images. Returns list of error messages."""
    from PIL import Image

    try:
        import av
    except ImportError:
        try:
            import decord
        except ImportError:
            return [f"  Episode {ep_idx}: cannot verify video - install PyAV or decord"]

    errors = []

    with open(ep_dir / "metadata.json") as f:
        metadata = json.load(f)
    num_frames = metadata["num_frames"] - 1  # converted drops last frame

    # Sample frame indices
    if num_frames <= num_samples:
        sample_indices = list(range(num_frames))
    else:
        sample_indices = np.linspace(0, num_frames - 1, num_samples, dtype=int).tolist()

    for cam in CAMERAS:
        video_path = video_dir / f"observation.images.{cam}" / f"episode_{ep_idx:06d}.mp4"
        if not video_path.exists():
            errors.append(f"  Episode {ep_idx}: missing video {video_path.name} for camera {cam}")
            continue

        # Decode video frames
        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            decoded_frames = {}
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx in sample_indices:
                    decoded_frames[frame_idx] = frame.to_ndarray(format="rgb24")
                if frame_idx >= max(sample_indices):
                    break
            container.close()
        except Exception:
            # Fallback to decord
            import decord
            vr = decord.VideoReader(str(video_path))
            decoded_frames = {}
            for idx in sample_indices:
                decoded_frames[idx] = vr[idx].asnumpy()

        # Compare with raw images
        for idx in sample_indices:
            if idx not in decoded_frames:
                errors.append(f"  Episode {ep_idx}, cam {cam}: could not decode frame {idx}")
                continue

            raw_img = np.array(Image.open(ep_dir / f"rgb/{cam}/{idx:010d}.jpg"))
            decoded_img = decoded_frames[idx]

            if raw_img.shape != decoded_img.shape:
                errors.append(
                    f"  Episode {ep_idx}, cam {cam}, frame {idx}: "
                    f"shape mismatch raw={raw_img.shape} vs decoded={decoded_img.shape}"
                )
                continue

            # Video compression introduces some loss, allow tolerance
            diff = np.abs(raw_img.astype(float) - decoded_img.astype(float))
            mean_diff = diff.mean()
            max_diff = diff.max()
            # CRF 22 H.264 should have very small differences
            if mean_diff > 10.0:
                errors.append(
                    f"  Episode {ep_idx}, cam {cam}, frame {idx}: "
                    f"large pixel diff (mean={mean_diff:.2f}, max={max_diff:.0f})"
                )

    return errors


def main(args: Args):
    from lerobot.common.constants import HF_LEROBOT_HOME

    raw_dir = args.raw_dir
    converted_dir = HF_LEROBOT_HOME / args.repo_id

    print(f"Raw data:      {raw_dir}")
    print(f"Converted data: {converted_dir}")
    print()

    # Check converted dataset exists
    if not converted_dir.exists():
        print(f"ERROR: Converted dataset not found at {converted_dir}")
        return

    # Find episodes
    ep_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f"Raw episodes: {len(ep_dirs)}")

    # Check meta
    info_path = converted_dir / "meta" / "info.json"
    if not info_path.exists():
        print("ERROR: meta/info.json not found")
        return

    with open(info_path) as f:
        info = json.load(f)
    print(f"Converted episodes: {info['total_episodes']}")
    print(f"Converted frames:   {info['total_frames']}")
    print(f"FPS: {info['fps']}, Robot: {info.get('robot_type', 'unknown')}")
    print()

    if info["total_episodes"] != len(ep_dirs):
        print(f"WARNING: Episode count mismatch: raw={len(ep_dirs)}, converted={info['total_episodes']}")

    num_episodes = min(len(ep_dirs), info["total_episodes"])

    # Verify lowdim (state/action)
    print("--- Verifying state/action values ---")
    all_errors = []
    for ep_idx in tqdm.tqdm(range(num_episodes), desc="Checking lowdim"):
        ep_dir = ep_dirs[ep_idx]
        parquet_path = converted_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        if not parquet_path.exists():
            all_errors.append(f"  Episode {ep_idx}: parquet file not found")
            continue
        errors = verify_episode_lowdim(ep_dir, parquet_path, ep_idx, args.atol)
        all_errors.extend(errors)

    if all_errors:
        print(f"FAILED: {len(all_errors)} errors found:")
        for e in all_errors[:20]:
            print(e)
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")
    else:
        print(f"PASSED: All {num_episodes} episodes match (state + action)")
    print()

    # Verify videos
    if args.check_videos:
        print(f"--- Verifying video frames ({args.num_video_samples} samples/episode) ---")
        video_dir = converted_dir / "videos" / "chunk-000"
        video_errors = []
        for ep_idx in tqdm.tqdm(range(num_episodes), desc="Checking videos"):
            ep_dir = ep_dirs[ep_idx]
            errors = verify_episode_video(ep_dir, video_dir, ep_idx, args.num_video_samples)
            video_errors.extend(errors)

        if video_errors:
            print(f"WARNINGS: {len(video_errors)} issues found:")
            for e in video_errors[:20]:
                print(e)
        else:
            print(f"PASSED: Video frames match across {num_episodes} episodes")
    print()
    print("Verification complete.")


if __name__ == "__main__":
    tyro.cli(main)
