"""Upload a locally cached LeRobot dataset to S3 so training jobs can mount it.

LeRobot resolves a dataset to ``HF_LEROBOT_HOME/<repo_id>``, and the training container points
``HF_LEROBOT_HOME`` at the mounted input channel. So the S3 prefix has to keep the
``<namespace>/<dataset_name>/`` layout, which is exactly what this script writes.

Usage:

    uv run --group sagemaker python sagemaker/upload_dataset.py \
        --repo-id ykorkmaz/yam_hang_mug_on_mug_tree \
        --s3-prefix s3://my-bucket/lerobot

Then launch with `--data-source s3 --s3-data-uri s3://my-bucket/lerobot`.
"""

import dataclasses
import os
import pathlib
import subprocess

import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    # LeRobot dataset id, e.g. "ykorkmaz/yam_hang_mug_on_mug_tree".
    repo_id: str
    # Destination prefix. The dataset is written to <s3_prefix>/<repo_id>/.
    s3_prefix: str
    # Local dataset root. Defaults to HF_LEROBOT_HOME, then ~/.cache/huggingface/lerobot.
    lerobot_home: pathlib.Path | None = None
    # AWS profile used for the upload. Datasets live in manip-cluster buckets, so that profile
    # writes to them directly rather than relying on cross-account access.
    profile: str = "manip-cluster"
    # Region of the destination bucket.
    region: str = "us-west-2"
    # Print the sync without transferring anything.
    dry_run: bool = False


def main(args: Args) -> None:
    root = args.lerobot_home or pathlib.Path(
        os.environ.get("HF_LEROBOT_HOME", pathlib.Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "lerobot")
    )
    source = pathlib.Path(root).expanduser() / args.repo_id
    if not (source / "meta").is_dir():
        raise FileNotFoundError(
            f"{source} does not look like a LeRobot dataset (no meta/ directory). Download it locally "
            "first, or pass --lerobot-home."
        )

    destination = f"{args.s3_prefix.rstrip('/')}/{args.repo_id}"
    cmd = [
        "aws",
        "s3",
        "sync",
        str(source),
        destination,
        "--profile",
        args.profile,
        "--region",
        args.region,
        "--delete",
    ]
    if args.dry_run:
        cmd.append("--dryrun")

    print(f"=> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nLaunch with: --data-source s3 --s3-data-uri {args.s3_prefix.rstrip('/')}")


if __name__ == "__main__":
    main(tyro.cli(Args))
