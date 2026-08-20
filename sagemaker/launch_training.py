"""Build the openpi training image and submit a job to a SageMaker training queue.

Usage:

    uv run --group sagemaker python sagemaker/launch_training.py \
        --user firstname.lastname \
        --config pi05_yam_mugontree \
        --exp-name mugontree_full_v1 \
        -- --batch-size 256 --num-train-steps 30000

Everything after the bare `--` is forwarded verbatim to scripts/train.py.

Run `sagemaker/smoke_test.py` first to check the AWS prerequisites.
"""

import base64
import dataclasses
from datetime import datetime
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time
from uuid import uuid4

import boto3
import tyro

import openpi.training.config as _config
import sagemaker
from sagemaker.aws_batch.training_queue import TrainingQueue
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

NAME = "openpi"
REPO_ROOT = pathlib.Path(__file__).parent.parent
CONFIG_DIR = REPO_ROOT / "sagemaker" / "configs"
GIT_DIFF_DIR = REPO_ROOT / "sagemaker" / "git_diffs"

# Jobs run in the shared-compute account, which owns the batch queues and the execution role.
SHARED_COMPUTE_ACCOUNT = "141701954645"
DEFAULT_PROFILE = "shared-compute"
DEFAULT_ARN = f"arn:aws:iam::{SHARED_COMPUTE_ACCOUNT}:role/CAM-Robotics-Sagemaker-role-us-west-2"
DEFAULT_QUEUE = "tri-cam-robotics"
# Data, checkpoints and artifacts belong in manip-cluster buckets instead; cross-account access is
# already configured. Creating buckets in the shared-compute account breaks budget tracking, so
# every S3 location is set explicitly rather than left to the SDK's default bucket.
ARTIFACT_ACCOUNT = "682769330988"

INSTANCE_MAPPER = {
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
    "p5e": "ml.p5e.48xlarge",
    "p5en": "ml.p5en.48xlarge",
    "p6": "ml.p6-b200.48xlarge",
}
# Every instance type above exposes 8 GPUs, which is the device count JAX will see in the job.
GPUS_PER_INSTANCE = 8

# Maps full instance type to the queue name suffix shared across queue families.
# Full queue name: fss-{queue_name}-{suffix}, where queue_name is a family such as "ml", "vla",
# "tri-cam-robotics" or "vla-spot".
QUEUE_SUFFIX_MAPPER = {
    "us-west-2": {
        "ml.p5.48xlarge": "p5-48xlarge-us-west-2",
        # The p5e queue is backed by a flexible training plan and carries a "-training-plan"
        # suffix in its name, unlike the on-demand queues.
        "ml.p5e.48xlarge": "p5e-48xlarge-us-west-2-training-plan",
        "ml.p5en.48xlarge": "p5en-48xlarge-us-west-2",
        "ml.p4de.24xlarge": "p4de-24xlarge-us-west-2",
        "ml.p4d.24xlarge": "p4d-24xlarge-us-west-2",
        "ml.p6-b200.48xlarge": "p6-b200-48xlarge-us-west-2",
    },
}


@dataclasses.dataclass(frozen=True)
class Args:
    # ── What to train ──
    # Name of an openpi config from src/openpi/training/config.py.
    config: str
    # Experiment name. Names the checkpoint directory and the W&B run.
    exp_name: str
    # Your TRI username, e.g. "firstname.lastname". Namespaces the ECR repo and the job name.
    user: str
    # Extra flags for scripts/train.py as one shell-quoted string, e.g. "--batch-size 256
    # --num-train-steps 40000". Prefer putting them after a bare `--` instead: argparse rejects a
    # value here that is a single token starting with a dash, such as "--overwrite".
    train_args: str = ""
    # Override the base checkpoint the weight loader reads. Useful to point at an S3 mirror of
    # gs://openpi-assets/checkpoints/pi05_base/params instead of pulling from GCS on every job.
    base_checkpoint: str | None = None
    # Run scripts/compute_norm_stats.py in the container before training. Only needed when the
    # norm stats for this config are not committed under assets/.
    compute_norm_stats: bool = False
    # Cap on the frames scanned by compute_norm_stats.
    norm_stats_max_frames: int | None = None

    # ── Data ──
    # "hub" pulls the config's repo_id from HuggingFace inside the container (needs HF_TOKEN in
    # secrets.env for private datasets). "s3" mounts a prefix you uploaded with upload_dataset.py.
    data_source: str = "hub"
    # S3 prefix holding <namespace>/<dataset_name>/ trees. Required when data_source is "s3".
    s3_data_uri: str | None = None

    # ── AWS ──
    # Region the queue, ECR repo and DLC base image live in.
    region: str = "us-west-2"
    # Local AWS profile used to build and push the image, and to submit the job.
    profile: str = DEFAULT_PROFILE
    # SageMaker execution role. Falls back to $SAGEMAKER_ARN, then the CAM robotics role.
    arn: str | None = None
    # Prefix in a manip-cluster (682769330988) bucket where checkpoints, model artifacts and the
    # packaged source are written. Falls back to $OPENPI_S3_BASE_URI. Required: leaving these to the
    # SDK would create a bucket in the shared-compute account, which is not allowed.
    s3_base_uri: str | None = None
    # Overrides the checkpoint location derived from s3_base_uri. Must stay stable across relaunches
    # of the same run, since it is what makes a requeued job resumable.
    checkpoint_s3_uri: str | None = None

    # ── Instance ──
    # Key of INSTANCE_MAPPER, e.g. "p5" for ml.p5.48xlarge.
    instance_type: str = "p5"
    # Must be 1: scripts/train.py has no jax.distributed.initialize().
    instance_count: int = 1
    # EBS volume size in GB. Only relevant for instance types without local NVMe.
    volume_size: int = 500
    # Max job runtime in days.
    max_run: int = 5

    # ── Queue ──
    # Queue family; the instance type and region complete the name.
    queue_name: str = DEFAULT_QUEUE
    # Higher runs sooner within the queue.
    priority: int = 400
    # ARN of a flexible training plan backing the queue, when it has one. Required by any queue
    # whose name ends in "-training-plan"; see the guard in validate().
    training_plan_arn: str | None = None
    # Attempts a job gets on infrastructure faults. Application errors exit immediately regardless,
    # so this does not re-run a doomed config N times.
    max_attempts: int = 5
    # Managed spot training. Required by the "-spot" queue families and rejected by on-demand ones.
    use_spot: bool = False

    # ── Misc ──
    # Prepended to the SageMaker job name.
    name_prefix: str | None = None
    # W&B project the run reports to.
    wandb_project: str = "openpi"
    # Skip the image build and reuse whatever is already tagged latest in ECR.
    skip_build: bool = False
    # Validate everything and write the job spec, but neither build the image nor submit the job.
    dry_run: bool = False


def run_command(command: str) -> None:
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def _run_git(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def get_git_env_vars() -> dict[str, str]:
    return {
        "OPENPI_GIT_COMMIT": _run_git(["git", "rev-parse", "HEAD"]),
        "OPENPI_GIT_BRANCH": _run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def remove_old_files(path: pathlib.Path, pattern: str, expiration_days: int = 3) -> None:
    for file in path.glob(pattern):
        if file.stat().st_mtime < time.time() - expiration_days * 24 * 60 * 60:
            file.unlink()


def save_git_diff(uuid: str) -> str:
    """Save the working-tree diff into the image for provenance. Returns the in-container path."""
    GIT_DIFF_DIR.mkdir(parents=True, exist_ok=True)
    remove_old_files(GIT_DIFF_DIR, "git_diff_*.txt")

    git_diff_file = GIT_DIFF_DIR / f"git_diff_{uuid}.txt"
    sagemaker_path = f"/opt/ml/code/sagemaker/git_diffs/{git_diff_file.name}"

    if not _run_git(["git", "status", "--porcelain"]):
        return sagemaker_path

    diff = _run_git(["git", "diff", "HEAD"])
    if untracked := _run_git(["git", "ls-files", "--others", "--exclude-standard"]):
        diff += "\n\n# Untracked files:\n" + untracked
    git_diff_file.write_text(diff)
    print(f"Saved git diff to {git_diff_file} ({len(diff)} bytes)")
    return sagemaker_path


def get_image(user: str, profile: str, region: str, *, skip_build: bool) -> str:
    os.environ["AWS_PROFILE"] = profile
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    assert account.isdigit(), f"Invalid account value: {account}"

    algorithm_name = f"{user}-{NAME}"
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"
    if skip_build:
        print(f"Skipping build, reusing {fullname}")
        return fullname

    dockerfile = pathlib.Path(__file__).parent / "Dockerfile"
    login_cmd = (
        f"aws ecr get-login-password --region {region} --profile {profile} | "
        f"docker login --username AWS --password-stdin"
    )

    print("Building container")
    commands = [
        # Log in to the AWS Deep Learning Containers account to pull the base image.
        f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
        f"docker build --progress=plain -f {dockerfile} --build-arg AWS_REGION={region} -t {algorithm_name} .",
        f"docker tag {algorithm_name} {fullname}",
        f"{login_cmd} {fullname}",
        (
            f"aws --region {region} ecr describe-repositories --repository-names {algorithm_name} --no-cli-pager || "
            f"aws --region {region} ecr create-repository --repository-name {algorithm_name} --no-cli-pager"
        ),
    ]
    run_command("\n".join(f"{x} || exit 1" for x in commands))
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return fullname


def sanitize_name(name: str) -> str:
    name = name.replace("_", "-").replace(".", "-")
    clean = "".join(c if c.isalnum() or c == "-" else "" for c in name).strip("-")
    return clean or "job"


def get_job_name(base: str) -> str:
    # SageMaker job names must match [a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}.
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # noqa: DTZ005
    job_name = f"{sanitize_name(base)}-{date_str}".lstrip("-")
    return job_name[:63].rstrip("-")


def read_secrets() -> dict[str, str]:
    """Load secrets.env (WANDB_API_KEY, optionally HF_TOKEN) from the repo root."""
    secrets_path = REPO_ROOT / "secrets.env"
    if not secrets_path.exists():
        print(f"Warning: {secrets_path} not found, no secrets will be injected into the job.")
        return {}
    secrets = {}
    for raw_line in secrets_path.read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            secrets[key.strip()] = value.strip().strip("\"'")
    return secrets


def resolve_s3_locations(args: Args) -> dict[str, str]:
    """Work out where checkpoints, artifacts and the packaged source go.

    Everything is keyed on config/exp_name rather than the (timestamped) job name so that
    relaunching the same experiment reuses the same checkpoint prefix and resumes.
    """
    base = args.s3_base_uri or os.environ.get("OPENPI_S3_BASE_URI")
    if not base:
        raise ValueError(
            "Pass --s3-base-uri (or set OPENPI_S3_BASE_URI) pointing at a prefix in a manip-cluster "
            f"({ARTIFACT_ACCOUNT}) bucket. Without it the SDK would default to creating a bucket in "
            f"the shared-compute account ({SHARED_COMPUTE_ACCOUNT}), which is not allowed. Ask in "
            "#cam-robotics or #ie if you need a bucket."
        )
    if not base.startswith("s3://"):
        raise ValueError(f"--s3-base-uri must be an s3:// URI, got '{base}'")

    # The SDK's default bucket is sagemaker-{region}-{account}; naming it explicitly would satisfy
    # the check above while still putting data in the wrong account.
    bucket = base.removeprefix("s3://").split("/")[0]
    if bucket == f"sagemaker-{args.region}-{SHARED_COMPUTE_ACCOUNT}":
        raise ValueError(
            f"'{bucket}' is the SageMaker default bucket in the shared-compute account. Checkpoints "
            f"and artifacts must live in a manip-cluster ({ARTIFACT_ACCOUNT}) bucket."
        )

    run_prefix = f"{base.rstrip('/')}/{args.config}/{args.exp_name}"
    return {
        "checkpoints": args.checkpoint_s3_uri or f"{run_prefix}/checkpoints",
        "output": f"{run_prefix}/output",
        "code": f"{run_prefix}/code",
    }


def validate(args: Args, train_config: _config.TrainConfig, overrides: list[str]) -> None:
    if args.instance_type not in INSTANCE_MAPPER:
        raise ValueError(f"Unknown instance type '{args.instance_type}'. Available: {list(INSTANCE_MAPPER)}")

    instance = INSTANCE_MAPPER[args.instance_type]
    if args.region not in QUEUE_SUFFIX_MAPPER:
        raise ValueError(f"No queues mapped for region '{args.region}'. Available: {list(QUEUE_SUFFIX_MAPPER)}")
    if instance not in QUEUE_SUFFIX_MAPPER[args.region]:
        raise ValueError(
            f"Instance '{instance}' has no queue in region '{args.region}'. "
            f"Available: {list(QUEUE_SUFFIX_MAPPER[args.region])}"
        )

    # A reserved-capacity queue submitted without its plan ARN is ACCEPTED by Batch -- it returns a
    # normal "queued" result and a real job -- and is then never scheduled. It holds head-of-line,
    # blocking every job behind it, until a queue operator kills it. Nothing at submit time hints at
    # this, so refuse it here. `aws sagemaker list-training-plans` lists the live ARNs.
    queue_name = f"fss-{args.queue_name}-{QUEUE_SUFFIX_MAPPER[args.region][instance]}"
    if queue_name.endswith("-training-plan") and not args.training_plan_arn:
        raise ValueError(
            f"Queue '{queue_name}' is backed by a reserved-capacity training plan. Submitting "
            "without --training-plan-arn is accepted but never scheduled, and blocks the queue for "
            "everyone. Pass --training-plan-arn (see `aws sagemaker list-training-plans`), or pick "
            "an on-demand instance type."
        )

    # scripts/train.py has no jax.distributed.initialize call, so a multi-node job would start N
    # independent single-node trainings that each overwrite the same checkpoint directory.
    if args.instance_count != 1:
        raise ValueError(
            "Only single-node jobs are supported. Multi-node requires adding jax.distributed.initialize() "
            "to scripts/train.py driven by SM_HOSTS/SM_CURRENT_HOST."
        )

    # A spot queue's service environment only accepts managed-spot jobs and an on-demand queue
    # rejects them. Both mismatches fail after the ~10 minute image build and push, so catch them now.
    queue_is_spot = "-spot" in args.queue_name
    if queue_is_spot != args.use_spot:
        raise ValueError(
            f"Queue family '{args.queue_name}' is {'spot' if queue_is_spot else 'on-demand'} but "
            f"--use-spot is {args.use_spot}. Pass --use-spot / --no-use-spot to match, or switch queues."
        )

    if args.data_source not in ("hub", "s3"):
        raise ValueError(f"--data-source must be 'hub' or 's3', got '{args.data_source}'")
    if args.data_source == "s3" and not args.s3_data_uri:
        raise ValueError("--data-source s3 requires --s3-data-uri (see sagemaker/upload_dataset.py)")

    # Mirror train.py's own check here so a bad batch size fails in seconds instead of after the
    # image build, the queue wait and the model init.
    batch_size = train_config.batch_size
    for i, flag in enumerate(overrides):
        if flag == "--batch-size":
            batch_size = int(overrides[i + 1])
        elif flag.startswith("--batch-size="):
            batch_size = int(flag.split("=", 1)[1])
    if batch_size % GPUS_PER_INSTANCE != 0:
        raise ValueError(f"Batch size {batch_size} must be divisible by {GPUS_PER_INSTANCE} GPUs.")


def main(args: Args, passthrough: list[str]) -> None:
    train_config = _config.get_config(args.config)  # Raises with suggestions on a typo.
    overrides = [*shlex.split(args.train_args), *passthrough]
    validate(args, train_config, overrides)
    s3 = resolve_s3_locations(args)

    # Shard across the whole node. A pi0.5 full fine-tune at batch 32 sits near 75GB of an 80GB card
    # even with 4-way FSDP, so 1-way (pure data parallelism) is not a safe default here, and this is
    # the setting our runs have actually completed under.
    if not any(a.startswith("--fsdp-devices") for a in overrides):
        overrides += ["--fsdp-devices", str(GPUS_PER_INSTANCE)]
    # openpi's default of 2 dataloader workers starves the GPUs on video-backed datasets. Measured
    # on ykorkmaz/yam_hang_mug_on_mug_tree (3x 720p h264 cameras, random-seek decoded per sample):
    # 656 ms/sample, i.e. 1.5 samples/s per worker. At batch 32 that predicts 32/(2*1.5) = 10.7 s/it,
    # which matched an observed ~11 s/it almost exactly -- the loader was the entire step time.
    # 32 workers gives ~48 samples/s, enough for the GPU to become the limit again, and a
    # p5.48xlarge has 192 vCPUs. Image-backed datasets do not need anything like this.
    if not any(a.startswith("--num-workers") for a in overrides):
        overrides += ["--num-workers", "32"]
    if args.base_checkpoint:
        overrides += [f"--weight-loader.params-path={args.base_checkpoint}"]

    # assets/ is gitignored, so a fresh clone has no such directory and the image build's COPY would
    # fail. Launching without norm stats is legitimate when --compute-norm-stats is set.
    assets_dir = REPO_ROOT / "assets"
    assets_dir.mkdir(exist_ok=True)
    (assets_dir / ".gitkeep").touch()

    uuid = str(uuid4())
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    remove_old_files(CONFIG_DIR, "train_args_*.json")
    spec = {
        "config": args.config,
        "exp_name": args.exp_name,
        "data_source": args.data_source,
        "repo_id": train_config.data.repo_id,
        "compute_norm_stats": args.compute_norm_stats,
        "norm_stats_max_frames": args.norm_stats_max_frames,
        "overrides": overrides,
        # train_entry.py mirrors checkpoints to and from here itself; see the CHECKPOINT_DIR
        # comment there for why SageMaker's own checkpoint sync is not used.
        "checkpoint_s3_uri": s3["checkpoints"],
    }
    # Kept on disk purely as a local record of what was launched; the container gets the spec
    # through the hyperparameter below, NOT from this file. Baking it into the image would tie
    # every launch to a rebuild and silently break --skip-build, since a reused image would only
    # ever contain some earlier launch's spec.
    (CONFIG_DIR / f"train_args_{uuid}.json").write_text(json.dumps(spec, indent=2))
    print(f"Job spec:\n{json.dumps(spec, indent=2)}")

    # base64 so the value survives the trip through the SageMaker hyperparameter map and the
    # toolkit's argv reconstruction untouched -- raw JSON would need quote escaping at both hops.
    spec_b64 = base64.b64encode(json.dumps(spec).encode()).decode()
    if len(spec_b64) > 2400:  # SageMaker caps a hyperparameter value at 2500 characters.
        raise ValueError(
            f"Encoded job spec is {len(spec_b64)} characters, too large for a SageMaker "
            "hyperparameter. Shorten the train.py overrides passed after `--`."
        )

    # Written before the build so it lands in the image.
    git_diff_sagemaker_path = save_git_diff(uuid)

    secrets = read_secrets()
    wandb_enabled = "--no-wandb-enabled" not in overrides
    if wandb_enabled and "WANDB_API_KEY" not in secrets:
        raise ValueError(
            f"{REPO_ROOT / 'secrets.env'} must define WANDB_API_KEY, or pass "
            "`-- --no-wandb-enabled` to train without logging."
        )
    if args.data_source == "hub" and "HF_TOKEN" not in secrets:
        print("Warning: HF_TOKEN is not in secrets.env; only public HuggingFace datasets will be reachable.")

    arn = args.arn or os.environ.get("SAGEMAKER_ARN") or DEFAULT_ARN
    instance_type = INSTANCE_MAPPER[args.instance_type]
    queue_name = f"fss-{args.queue_name}-{QUEUE_SUFFIX_MAPPER[args.region][instance_type]}"

    print(f"\nExecution role: {arn}")
    print(f"Instance:       {instance_type} x{args.instance_count}")
    print(f"Queue:          {queue_name} (priority {args.priority})")
    print(f"Checkpoints:    {s3['checkpoints']}")
    print(f"Artifacts:      {s3['output']}")

    if args.dry_run:
        print("\nDry run: everything validated, no image built and no job submitted.")
        return

    image = get_image(args.user, profile=args.profile, region=args.region, skip_build=args.skip_build)
    os.environ["AWS_DEFAULT_REGION"] = args.region

    sagemaker_session = sagemaker.Session(boto_session=boto3.session.Session(region_name=args.region))
    account = boto3.client("sts").get_caller_identity()["Account"]
    print(f"AWS account: {account}")
    if account != SHARED_COMPUTE_ACCOUNT:
        # The queues and the execution role live in shared-compute; submitting from anywhere else
        # fails at the queue, after the image has already been built and pushed.
        print(
            f"Warning: authenticated as {account}, but the batch queues live in the shared-compute "
            f"account ({SHARED_COMPUTE_ACCOUNT}). Use --profile {DEFAULT_PROFILE}."
        )

    base_job_name = sanitize_name(f"{args.name_prefix + '-' if args.name_prefix else ''}{args.user}-{NAME}")
    job_name = get_job_name(base_job_name)

    environment = {
        "SM_USE_RESERVED_CAPACITY": "0" if args.training_plan_arn or args.use_spot else "1",
        "WANDB_PROJECT": args.wandb_project,
        "SAGEMAKER_PROGRAM": "/opt/ml/code/sagemaker/train_entry.py",
        "FI_EFA_FORK_SAFE": "1",
        "OPENPI_GIT_DIFF_FILE": git_diff_sagemaker_path,
        "OPENPI_LAUNCHED_BY": args.user,
        **get_git_env_vars(),
        **secrets,
    }

    max_run_seconds = args.max_run * 24 * 60 * 60

    estimator = PyTorch(
        entry_point="sagemaker/train_entry.py",
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters={"args_b64": spec_b64},
        role=arn,
        image_uri=image,
        instance_count=args.instance_count,
        instance_type=INSTANCE_MAPPER[args.instance_type],
        # The job's name is set by queue.map(job_names=...) below -- the estimator has no such
        # parameter, so passing one here would be silently swallowed by its **kwargs.
        #
        # NOTE: checkpoint_local_path / checkpoint_s3_uri are deliberately NOT set. They enable
        # SageMaker's checkpoint sync agent, which races orbax's writes and corrupts saves --
        # train_entry.py mirrors checkpoints itself instead. The remaining S3 locations are passed
        # explicitly so nothing lands in a default bucket the SDK would create in shared-compute.
        output_path=s3["output"],
        code_location=s3["code"],
        max_run=max_run_seconds,
        # Spot bills only running time, so the wait window can equal the run budget: capacity waits
        # are free and the job still stops after max_run of compute.
        max_wait=max_run_seconds if args.use_spot else None,
        use_spot_instances=args.use_spot,
        # File mode (a full download to local disk) rather than FastFile: LeRobot random-seeks into
        # parquet shards and mp4 episodes every step, which a FUSE mount serves badly.
        input_mode="File",
        environment=environment,
        # Warm pools cannot be combined with spot instances.
        keep_alive_period_in_seconds=None if args.use_spot else 5 * 60,
        # SageMaker Debugger/Profiler is in maintenance mode and closed to new customers, so the
        # SDK's default of attaching a ProfilerConfig makes the API reject the job outright with
        # "SageMaker Debugger is in maintenance mode ... (Status Code: 400)". Nothing here uses it
        # -- training metrics go to W&B -- so turn both off.
        disable_profiler=True,
        debugger_hook_config=False,
        training_plan=args.training_plan_arn,
        tags=[
            {"Key": "tri.project", "Value": "MM:PJ-0077"},
            {"Key": "tri.owner.email", "Value": f"{args.user}@tri.global"},
        ],
        volume_size=args.volume_size,
    )

    inputs = None
    if args.data_source == "s3":
        inputs = {"train": TrainingInput(args.s3_data_uri, input_mode="File")}

    queue = TrainingQueue(queue_name=queue_name)

    # Retry only on infrastructure faults. Without evaluateOnExit, a bad config burns every attempt
    # re-running an identical, identically-doomed job. A retried attempt restarts the container,
    # which restores the latest checkpoint from S3 and resumes (see train_entry.py).
    retry_config = {
        "attempts": args.max_attempts,
        "evaluateOnExit": [
            {"action": "RETRY", "onStatusReason": "Host EC2*"},
            {"action": "RETRY", "onStatusReason": "*InternalServerError*"},
            {"action": "RETRY", "onStatusReason": "*CapacityError*"},
            {"action": "EXIT", "onStatusReason": "*"},
        ],
    }

    try:
        queue.map(
            estimator,
            inputs=[inputs],
            job_names=[job_name],
            priority=args.priority,
            share_identifier="default",
            retry_config=retry_config,
            timeout={"attemptDurationSeconds": max_run_seconds},
        )
        print(f"Queued {job_name} to {queue_name} at priority {args.priority}")
    except Exception as e:
        print(f"Failed to queue {job_name}: {e}")
        raise


if __name__ == "__main__":
    # Split on a bare `--` before tyro sees it, so train.py flags can be written plainly instead of
    # squeezed into a quoted --train-args string.
    _argv = sys.argv[1:]
    _passthrough: list[str] = []
    if "--" in _argv:
        _split = _argv.index("--")
        _argv, _passthrough = _argv[:_split], _argv[_split + 1 :]
    main(tyro.cli(Args, args=_argv), _passthrough)
