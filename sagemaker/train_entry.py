#!/usr/bin/env python
"""In-container entrypoint for SageMaker training jobs.

The SageMaker training toolkit invokes this with the image's system python, so before anything
else it re-execs into the uv venv that the Dockerfile built (see sagemaker/Dockerfile for why
openpi does not share the DLC's python).

Everything the job needs arrives in a single base64-encoded JSON hyperparameter written by
launch_training.py. It travels with the job rather than being baked into the image, so reusing an
image with --skip-build still runs the spec you just launched. From there this script sets up the
cache and checkpoint locations, optionally computes norm stats, and finally execs scripts/train.py.
"""

import os
import sys

VENV_PYTHON = "/opt/ml/code/.venv/bin/python"

if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    _env = dict(os.environ)
    # The training toolkit exports PYTHONPATH pointing at the DLC's python3.12 stdlib
    # (/usr/local/lib/python3.12, its lib-dynload and site-packages). Inheriting that into the
    # venv's 3.11 interpreter makes the very first stdlib import load 3.12's `re` against 3.11's
    # `_sre` extension module, which dies with "AssertionError: SRE module mismatch". The venv
    # needs none of it -- openpi is installed into it -- so drop both interpreter path variables
    # and use execve to hand over the cleaned environment.
    _env.pop("PYTHONPATH", None)
    _env.pop("PYTHONHOME", None)
    os.execve(VENV_PYTHON, [VENV_PYTHON, os.path.abspath(__file__), *sys.argv[1:]], _env)

import argparse  # noqa: E402
import base64  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402

CODE_DIR = pathlib.Path("/opt/ml/code")
# The ML storage volume is mounted at /opt/ml, so caches live there rather than in the (small)
# container filesystem. pi05_base alone is ~7GB.
CACHE_DIR = pathlib.Path("/opt/ml/cache")
# Checkpoints deliberately do NOT live in /opt/ml/checkpoints. That path is watched by SageMaker's
# checkpoint sync agent, which copies files to S3 as they appear -- and it races orbax, which
# writes into .orbax-checkpoint-tmp-* directories and reads its own metadata back during finalize.
# The observed symptom is orbax writing array_metadatas/process_0 and then reading it back empty a
# few seconds later ("json.decoder.JSONDecodeError: Expecting value: line 1 column 1"), failing the
# save and killing the job. Writing to unsynced instance storage and uploading ourselves (see
# _upload_checkpoints) removes the race entirely.
CHECKPOINT_DIR = CACHE_DIR / "checkpoints"
# Where SageMaker mounts the "train" input channel when --data-source s3 is used.
S3_DATA_DIR = pathlib.Path("/opt/ml/input/data/train")
# How often the background uploader mirrors checkpoints to S3. A pi0.5 checkpoint is ~50GiB and
# uploads in a few minutes, so this stays well clear of one sync overlapping the next.
SYNC_INTERVAL_SECONDS = 15 * 60

logger = logging.getLogger("openpi.sagemaker")

# Held for the duration of an upload so a slow sync cannot stack up behind the next tick.
_upload_lock = threading.Lock()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [entry] %(message)s",
        datefmt="%H:%M:%S",
    )


def _setup_env(spec: dict) -> None:
    """Point openpi's caches at the ML storage volume and wire up the data source."""
    openpi_cache = CACHE_DIR / "openpi"
    hf_cache = CACHE_DIR / "huggingface"
    jax_cache = CACHE_DIR / "jax"
    for path in (openpi_cache, hf_cache, jax_cache, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OPENPI_DATA_HOME", str(openpi_cache))
    os.environ.setdefault("HF_HOME", str(hf_cache))
    # train.py points the JAX compilation cache at ~/.cache/jax unconditionally; giving HOME a
    # path on the ML volume keeps that (and anything else writing to ~) off the container disk.
    os.environ.setdefault("HOME", str(CACHE_DIR / "home"))
    pathlib.Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)

    # JAX is the only thing using the GPUs here; let it preallocate nearly all of HBM.
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

    if spec["data_source"] == "s3":
        if not S3_DATA_DIR.exists():
            raise FileNotFoundError(
                f"--data-source s3 was requested but the 'train' input channel is not mounted at "
                f"{S3_DATA_DIR}. Check the --s3-data-uri passed to launch_training.py."
            )
        # LeRobot resolves a dataset to HF_LEROBOT_HOME/<repo_id>, so the S3 prefix must be laid
        # out as <prefix>/<namespace>/<dataset_name>/. upload_dataset.py does that for you.
        os.environ["HF_LEROBOT_HOME"] = str(S3_DATA_DIR)
        repo_id = spec.get("repo_id")
        if repo_id and not (S3_DATA_DIR / repo_id).exists():
            present = sorted(str(p.relative_to(S3_DATA_DIR)) for p in S3_DATA_DIR.glob("*/*"))
            raise FileNotFoundError(
                f"Dataset '{repo_id}' not found under the mounted channel {S3_DATA_DIR}. "
                f"Found instead: {present}. The S3 prefix must contain <namespace>/<dataset_name>/."
            )
        logger.info(f"Reading LeRobot datasets from the mounted S3 channel: {S3_DATA_DIR}")
    else:
        logger.info("Reading LeRobot datasets from the HuggingFace Hub")
        if "HF_TOKEN" not in os.environ:
            logger.warning("HF_TOKEN is not set; only public datasets will be reachable.")

    diff_file = os.environ.get("OPENPI_GIT_DIFF_FILE")
    if diff_file and pathlib.Path(diff_file).exists():
        logger.info(f"Launched with uncommitted changes, diff saved at {diff_file}")


def _run(argv: list[str]) -> None:
    logger.info(f"=> {' '.join(argv)}")
    subprocess.run(argv, check=True, cwd=CODE_DIR)


# ── Checkpoint mirroring ────────────────────────────────────────────────────────────────────────
# Replaces SageMaker's built-in checkpoint sync (see the CHECKPOINT_DIR comment). We own both
# halves: pull the latest checkpoint down at startup so --resume works, and push periodically plus
# once at exit so a preempted job does not lose everything.


def _s3(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["aws", "s3", *argv], capture_output=True, text=True, check=False)


def _latest_remote_step(run_uri: str) -> str | None:
    """Highest numbered step directory under the run's S3 prefix, or None if there are none."""
    listing = _s3(["ls", f"{run_uri.rstrip('/')}/"])
    if listing.returncode != 0:
        # A prefix that does not exist yet is the normal first-run case: `aws s3 ls` exits 1 with
        # nothing on stderr. Only a real error (permissions, bad bucket) is worth warning about.
        if listing.stderr.strip():
            logger.warning(f"Could not list {run_uri}: {listing.stderr.strip()[:200]}")
        return None
    steps = [m.group(1) for line in listing.stdout.splitlines() if (m := re.match(r"\s+PRE (\d+)/", line))]
    return max(steps, key=int) if steps else None


def _restore_checkpoints(run_uri: str, run_dir: pathlib.Path) -> None:
    """Download only the newest checkpoint, plus the run's top-level files.

    Downloading every step would work but is unbounded: at ~50GiB each, a run with several retained
    checkpoints would exhaust the volume. openpi only ever restores the latest step, so that plus
    wandb_id.txt (which keeps the W&B run continuous) is all a resume needs.
    """
    step = _latest_remote_step(run_uri)
    if step is None:
        logger.info(f"No existing checkpoints under {run_uri}; starting fresh.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    # Top-level files only -- "*/*" excludes everything inside the step directories.
    top = _s3(["sync", f"{run_uri.rstrip('/')}/", str(run_dir), "--exclude", "*/*"])
    if top.returncode != 0:
        logger.warning(f"Failed to restore run metadata: {top.stderr.strip()[:300]}")

    logger.info(f"Restoring checkpoint step {step} from {run_uri}")
    got = _s3(["sync", f"{run_uri.rstrip('/')}/{step}", str(run_dir / step)])
    if got.returncode != 0:
        raise RuntimeError(f"Failed to restore checkpoint step {step}: {got.stderr.strip()[:300]}")
    logger.info(f"Restored {run_dir / step}")


def _upload_checkpoints(run_uri: str, run_dir: pathlib.Path, *, blocking: bool = False) -> None:
    if not run_dir.exists():
        return
    # Never mirror an empty directory: with --delete that would erase good checkpoints from S3, so
    # the guard matters even though we do not pass it today.
    if not any(run_dir.iterdir()):
        return
    if not _upload_lock.acquire(blocking=blocking):
        logger.info("Previous checkpoint upload still running; skipping this interval.")
        return
    try:
        logger.info(f"Uploading checkpoints to {run_uri}")
        result = _s3(["sync", str(run_dir), f"{run_uri.rstrip('/')}/"])
        if result.returncode != 0:
            # A failed upload must not kill a healthy training run; the next tick retries.
            logger.error(f"Checkpoint upload failed: {result.stderr.strip()[:300]}")
        else:
            logger.info("Checkpoint upload complete")
    finally:
        _upload_lock.release()


def _uploader_loop(run_uri: str, run_dir: pathlib.Path, stop: threading.Event) -> None:
    while not stop.wait(SYNC_INTERVAL_SECONDS):
        _upload_checkpoints(run_uri, run_dir)


def main() -> None:
    _setup_logging()
    # Records which interpreter actually won the re-exec, and whether a stray PYTHONPATH survived.
    logger.info(f"Python {sys.version.split()[0]} at {sys.executable}")
    logger.info(f"PYTHONPATH={os.environ.get('PYTHONPATH', '(unset)')}")

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--args_b64", help="base64-encoded JSON job spec from launch_training.py.")
    group.add_argument("--args_path", help="Path to a JSON job spec. Only used for local debugging.")
    args = parser.parse_args()

    if args.args_b64:
        spec = json.loads(base64.b64decode(args.args_b64).decode())
    else:
        spec = json.loads(pathlib.Path(args.args_path).read_text())
    logger.info(f"Job spec:\n{json.dumps(spec, indent=2)}")

    _setup_env(spec)

    config_name = spec["config"]
    exp_name = spec["exp_name"]

    if spec.get("compute_norm_stats"):
        # Writes to assets_base_dir/<config>/<asset_id>, i.e. the assets/ tree baked into the image.
        # Only useful when the norm stats for this config were not committed to the repo.
        norm_argv = [sys.executable, "scripts/compute_norm_stats.py", f"--config-name={config_name}"]
        if spec.get("norm_stats_max_frames") is not None:
            norm_argv.append(f"--max-frames={spec['norm_stats_max_frames']}")
        _run(norm_argv)

    train_argv = [
        sys.executable,
        "scripts/train.py",
        config_name,
        f"--exp-name={exp_name}",
        f"--checkpoint-base-dir={CHECKPOINT_DIR}",
        f"--assets-base-dir={CODE_DIR / 'assets'}",
        *spec.get("overrides", []),
    ]

    run_dir = CHECKPOINT_DIR / config_name / exp_name
    run_uri = spec.get("checkpoint_s3_uri")
    if run_uri:
        run_uri = f"{run_uri.rstrip('/')}/{config_name}/{exp_name}"
        _restore_checkpoints(run_uri, run_dir)
    else:
        logger.warning("No checkpoint_s3_uri in the job spec; checkpoints stay on local disk only.")

    # A requeued job (or a reclaimed spot instance) restarts this container from scratch, so resume
    # from whatever we just restored rather than refusing to run against a non-empty directory.
    # openpi downgrades this to a fresh start on its own if no step has been written yet.
    already_specified = any(a.startswith(("--resume", "--no-resume", "--overwrite")) for a in train_argv)
    if run_dir.exists() and any(run_dir.iterdir()) and not already_specified:
        logger.info(f"Found an existing checkpoint directory at {run_dir}, resuming.")
        train_argv.append("--resume")

    logger.info(f"=> {' '.join(train_argv)}")
    os.chdir(CODE_DIR)

    # Run training as a child rather than exec'ing it: exec would replace this process, leaving
    # nothing to upload the final checkpoint or to flush on a spot interruption.
    proc = subprocess.Popen(train_argv, cwd=CODE_DIR)

    def _forward(signum, _frame):
        # SageMaker sends SIGTERM ~2 minutes before reclaiming a spot instance. Pass it on so
        # training shuts down, then the final upload below still runs.
        logger.info(f"Received signal {signum}; forwarding to training.")
        proc.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    stop = threading.Event()
    uploader = None
    if run_uri:
        uploader = threading.Thread(
            target=_uploader_loop, args=(run_uri, run_dir, stop), daemon=True, name="checkpoint-uploader"
        )
        uploader.start()

    returncode = proc.wait()
    stop.set()
    if uploader is not None:
        uploader.join(timeout=10)
    logger.info(f"Training exited with code {returncode}")

    if run_uri:
        # blocking=True so this waits out an in-flight periodic sync instead of skipping the final
        # upload, which is the one that must not be lost.
        _upload_checkpoints(run_uri, run_dir, blocking=True)

    sys.exit(returncode)


if __name__ == "__main__":
    main()
