"""Scan all processed episodes and flag frames with non-14D joints."""

import pickle
import sys
from pathlib import Path

import numpy as np


def check_episode(ep_dir: Path) -> list[str]:
    """Check all pkl files in an episode's lowdim dir. Returns list of issues."""
    lowdim_dir = ep_dir / "lowdim"
    if not lowdim_dir.exists():
        return [f"  NO lowdim/ directory"]

    issues = []
    files = sorted(lowdim_dir.glob("*.pkl"))
    if not files:
        return [f"  NO pkl files found"]

    for fpath in files:
        try:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            issues.append(f"  {fpath.name}: LOAD ERROR: {e}")
            continue

        if "joints" not in data:
            issues.append(f"  {fpath.name}: missing 'joints' key, keys={list(data.keys())}")
            continue

        arr = np.asarray(data["joints"])
        if arr.shape != (14,):
            issues.append(f"  {fpath.name}: shape={arr.shape}")

    return issues


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_lowdim.py /path/to/processed/data")
        print("  e.g. python check_lowdim.py /home/robot-lab/data/processed")
        sys.exit(1)

    data_root = Path(sys.argv[1])
    if not data_root.exists():
        print(f"ERROR: {data_root} does not exist")
        sys.exit(1)

    # Support both: passing the top-level processed dir (task/episode structure)
    # or passing a single task dir directly (episode structure)
    ep_dirs_direct = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()])

    total_episodes = 0
    bad_episodes = 0
    total_bad_frames = 0

    if ep_dirs_direct:
        # data_root is a task dir containing episode dirs directly
        for ep_dir in ep_dirs_direct:
            total_episodes += 1
            issues = check_episode(ep_dir)
            if issues:
                bad_episodes += 1
                total_bad_frames += len(issues)
                print(f"\n{ep_dir.name}:")
                for issue in issues:
                    print(issue)
    else:
        # data_root is the top-level dir containing task dirs
        task_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
        for task_dir in task_dirs:
            ep_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir() and d.name.isdigit()])
            if not ep_dirs:
                continue
            for ep_dir in ep_dirs:
                total_episodes += 1
                issues = check_episode(ep_dir)
                if issues:
                    bad_episodes += 1
                    total_bad_frames += len(issues)
                    print(f"\n{ep_dir.relative_to(data_root)}:")
                    for issue in issues:
                        print(issue)

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total episodes scanned: {total_episodes}")
    print(f"  Episodes with issues:   {bad_episodes}")
    print(f"  Total bad frames:       {total_bad_frames}")


if __name__ == "__main__":
    main()
