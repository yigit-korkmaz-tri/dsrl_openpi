# YAM Dataset Builder

Convert raw YAM bimanual robot demonstrations into [LeRobot v2.0](https://github.com/huggingface/lerobot) dataset format for OpenPI fine-tuning.

## Quick Start

```bash
# Convert raw data to LeRobot format
uv run yam_dataset_builder/convert_yam_data_to_lerobot.py \
    --args.raw-dir /path/to/raw_dataset \
    --args.repo-id <hf_user>/yam_<task_name>

# Verify the conversion
uv run yam_dataset_builder/verify_dataset.py \
    --args.raw-dir /path/to/raw_dataset \
    --args.repo-id <hf_user>/yam_<task_name>
```

## Expected Input Structure

The raw dataset directory should contain numbered episode folders with the following layout:

```
raw_dataset_dir/
├── 0/                              # Episode folder (numeric name)
│   ├── metadata.json
│   ├── rgb/
│   │   ├── head/
│   │   │   ├── 0000000000.jpg      # 1280x720 RGB images
│   │   │   ├── 0000000001.jpg
│   │   │   └── ...
│   │   ├── left_wrist/
│   │   │   └── ...
│   │   └── right_wrist/
│   │       └── ...
│   └── lowdim/
│       ├── head/
│       │   ├── 0000000000.npz      # Joint data per frame
│       │   ├── 0000000001.npz
│       │   └── ...
│       ├── left_wrist/
│       │   └── ...
│       └── right_wrist/
│           └── ...
├── 1/
│   └── (same structure)
└── ...
```

### Images

- Format: JPEG, 1280x720 RGB
- Naming: 10-digit zero-padded frame index (e.g., `0000000000.jpg`)
- 3 cameras: `head`, `left_wrist`, `right_wrist`

### Low-dimensional Data (NPZ)

Each `.npz` file contains:
- **`joints`**: 14D float32 array — `[left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]`
- **`action`**: 26D cartesian array (present in raw data but not used during conversion)

### Metadata (per episode)

`metadata.json`:
```json
{
  "num_frames": 150,
  "language": {
    "prompt": ["pick up the lock and place it on the hook"]
  }
}
```

## Output Structure

The converter produces a LeRobot v2.0 dataset under `~/.cache/huggingface/datasets/hub/<repo_id>/`:

```
<repo_id>/
├── meta/
│   ├── info.json
│   ├── features.json
│   └── data_index.json
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.head/
        │   ├── episode_000000.mp4
        │   └── ...
        ├── observation.images.left_wrist/
        │   └── ...
        └── observation.images.right_wrist/
            └── ...
```

### Parquet columns

| Column | Type | Description |
|--------|------|-------------|
| `observation.state` | 14D float32 | Joint positions at frame `t` |
| `action` | 14D float32 | Joint positions at frame `t+1` (absolute target) |
| `frame_index` | int32 | Frame number within episode |
| `timestamp` | float32 | `frame_index / 30` (seconds) |
| `task` | string | Language instruction from metadata |

### Key transformations

- The last frame of each episode is dropped (no next-state for action), so `output_frames = raw_frames - 1`.
- Actions are computed as `joints[t+1]` (absolute joint targets), not deltas.
- Videos are encoded as H.264 MP4 (libx264, yuv420p, CRF 22) at 30 FPS.

## CLI Reference

### convert_yam_data_to_lerobot.py

| Argument | Required | Description |
|----------|----------|-------------|
| `--args.raw-dir` | Yes | Path to raw YAM dataset directory |
| `--args.repo-id` | Yes | HuggingFace repo ID (e.g., `yuzhi/yam_task`) |
| `--args.push-to-hub` | No | Push to HuggingFace Hub after conversion |
| `--args.private` | No | Make Hub repo private (default: True) |

### verify_dataset.py

| Argument | Required | Description |
|----------|----------|-------------|
| `--args.raw-dir` | Yes | Path to original raw dataset directory |
| `--args.repo-id` | Yes | LeRobot repo ID used during conversion |
| `--args.check-videos` | No | Enable video frame verification (default: True) |
| `--args.num-video-samples` | No | Frames to sample per episode for video check (default: 5) |
| `--args.atol` | No | Float comparison tolerance (default: 1e-5) |

## Pushing to HuggingFace Hub

### During conversion

Pass `--args.push-to-hub` when running the converter:

```bash
uv run yam_dataset_builder/convert_yam_data_to_lerobot.py \
    --args.raw-dir /path/to/raw_dataset \
    --args.repo-id <hf_user>/yam_<task_name> \
    --args.push-to-hub
```

By default the repo is created as private. To make it public, add `--args.no-private`.

### Pushing an already-converted dataset

If you have already converted a dataset locally and want to push it to the Hub without re-running the conversion:

```bash
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset(repo_id='<hf_user>/yam_<task_name>').push_to_hub(private=True)"
```

The converted dataset is stored locally under `~/.cache/huggingface/datasets/hub/<repo_id>/`. Make sure the `repo_id` matches the one used during conversion.
