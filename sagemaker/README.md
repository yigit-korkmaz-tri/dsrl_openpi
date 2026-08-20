# Training openpi on SageMaker

Full fine-tuning of pi0.5 on a single `ml.p5.48xlarge` (8x H100), submitted to a TRI SageMaker
training queue.

## How it fits together

| File | Role |
| --- | --- |
| `Dockerfile` | AWS PyTorch DLC base + a uv-managed venv holding openpi's JAX/torch pins |
| `launch_training.py` | Runs locally: builds & pushes the image, writes the job spec, submits to the queue |
| `train_entry.py` | Runs in the container: sets caches up, optionally computes norm stats, execs `scripts/train.py` |
| `upload_dataset.py` | Copies a local LeRobot dataset into S3 for `--data-source s3` |
| `smoke_test.py` | Preflight checks, no build and no submission |

openpi is JAX, not PyTorch, so there is no `torch_distributed` in the estimator. `scripts/train.py`
has no `jax.distributed.initialize()` call, so **only single-node jobs are supported**; the launcher
rejects `--instance-count > 1` rather than silently starting N independent trainings that fight over
one checkpoint directory.

## Accounts and storage

Jobs run in the **shared-compute** account (`141701954645`), which owns the batch queues and the
execution role. Checkpoints, artifacts and datasets live in **manip-cluster** (`682769330988`)
buckets. Do not create buckets in the shared-compute account — it breaks budget tracking.

Use the team's shared bucket, namespaced by your username:

```bash
export OPENPI_S3_BASE_URI=s3://robotics-cam-checkpoints/<firstname-lastname>/openpi
```

Two quirks of that bucket. The `Robo-SharedCompute-BatchOperator` permission set grants `GetObject`
and `PutObject` but deliberately **not** `DeleteObject`, so use `--profile manip-cluster` to remove
anything. And the smoke test's write probe therefore cannot clean up after itself; it leaves one
fixed-name object that the next run overwrites.

**Creating your own bucket is not self-service.** Cross-account S3 needs permission on both sides:
a bucket policy in manip-cluster (which you can add) *and* an identity policy in shared-compute
(which you cannot — the SSO permission set allowlists buckets by ARN and is managed centrally). A
bucket you create yourself will be denied even with a correct bucket policy. To get one wired up,
ask in `#cam-robotics` or `#ie` to add its ARN to both the `Robo-SharedCompute-BatchOperator`
permission set and the `CAM-Robotics-Sagemaker-role-us-west-2` execution role, and copy the bucket
policy from an already-working bucket such as `claire-yang-us-west-2`.

This is also why `--s3-base-uri` is mandatory: left to itself, the SageMaker SDK writes checkpoints,
model artifacts and the packaged source to a default bucket it creates in whichever account you
authenticated to. The launcher sets all three locations explicitly and rejects an `--s3-base-uri`
pointing at the shared-compute default bucket.

## One-time setup

**1. Docker.** The launcher builds and pushes an image, so you need a working daemon:

```bash
bash scripts/docker/install_docker_ubuntu22.sh
```

The installer adds you to the `docker` group, but group membership is stamped onto a session at
login — so **log out and back in**, or run `newgrp docker` for a single shell. Until you do,
`docker info` fails with `permission denied ... /var/run/docker.sock` even though the daemon is
running and `/etc/group` lists you. Verify with `id -nG | grep docker`.

**2. AWS.** `~/.aws/config` needs the `sso` session plus the `manip-cluster` and `shared-compute`
profiles.

```bash
aws sso login --profile shared-compute
export AWS_PROFILE=shared-compute
export OPENPI_S3_BASE_URI=s3://robotics-cam-checkpoints/<firstname-lastname>/openpi
```

**3. Secrets.** `secrets.env` is gitignored and gets injected into the job's environment.

```bash
cat > secrets.env <<'EOF'
WANDB_API_KEY=...
HF_TOKEN=...          # needed for private HuggingFace datasets
EOF
```

**4. Dependencies.**

```bash
uv sync --group sagemaker
```

**5. Preflight.** Every line must be OK before you spend build time.

```bash
uv run --group sagemaker python sagemaker/smoke_test.py \
    --user firstname.lastname --config pi05_yam_mugontree \
    --s3-base-uri $OPENPI_S3_BASE_URI
```

Defaults are already set for this environment, so you rarely pass them: profile `shared-compute`,
queue family `tri-cam-robotics`, region `us-west-2`, instance `p5`, and role
`arn:aws:iam::141701954645:role/CAM-Robotics-Sagemaker-role-us-west-2` (overridable with `--arn` or
`$SAGEMAKER_ARN`).

## Running a job

**Dry run first.** Validates the config name, norm stats, batch size, queue/spot pairing and data
source, and prints the resolved S3 destinations. Takes about a minute, builds nothing.

```bash
uv run --group sagemaker python sagemaker/launch_training.py \
    --user firstname.lastname --config pi05_yam_mugontree --exp-name smoke_v0 \
    --dry-run -- --batch-size 32 --num-train-steps 200
```

**Then a short proving run.** Do not make your first launch the real 30k-step job — send a cheap one
to exercise the whole chain: image build, ECR push, queue admission, dataset pull, the ~7GB
`pi05_base` download, and checkpoint sync back to S3.

```bash
uv run --group sagemaker python sagemaker/launch_training.py \
    --user firstname.lastname --config pi05_yam_mugontree --exp-name smoke_v0 \
    --max-run 1 \
    -- --batch-size 32 --num-train-steps 200 --save-interval 100
```

The first build is slow — roughly 10GB of DLC base image plus 10GB of CUDA wheels. Later builds
reuse the dependency layer; editing anything under `src/`, `scripts/` or `assets/` only invalidates
the source layer. Note that the queue submission happens *after* the build and push, so killing the
launcher during the build submits nothing.

**Then the real run.**

```bash
uv run --group sagemaker python sagemaker/launch_training.py \
    --user firstname.lastname --config pi05_yam_mugontree --exp-name mugontree_full_v1 \
    -- --num-train-steps 30000 --save-interval 1000
```

On batch size: the config declares 32 and the SLURM script in `scripts/` used 64 on 4x H200. Leave
it at the default for the first real run and raise it only once you have seen actual memory headroom
in the smoke job's logs — full fine-tuning of a ~3B model plus AdamW state is the memory-hungry
case, and an OOM hours into a queued job is expensive.

## Monitoring

The launcher prints the job name on submission.

```bash
# recent jobs
aws sagemaker list-training-jobs --region us-west-2 \
    --sort-by CreationTime --sort-order Descending --max-results 5

# status of one job
aws sagemaker describe-training-job --region us-west-2 --training-job-name <job-name> \
    --query '{Status:TrainingJobStatus,Secondary:SecondaryStatus,Msg:StatusMessage}'

# live logs
aws logs tail /aws/sagemaker/TrainingJobs --region us-west-2 \
    --log-stream-name-prefix <job-name> --follow
```

A queued period before anything starts is normal on a shared queue. `SecondaryStatus` walks through
`Starting` → `Downloading` → `Training`. Loss curves land in W&B under the `--wandb-project`
(default `openpi`).

While a job is still queued it has no training job yet, so look for it on the Batch queue instead —
note it is a `SAGEMAKER_TRAINING` queue, which needs `list-service-jobs`, not `list-jobs`:

```bash
aws batch list-service-jobs --region us-west-2 \
    --job-queue fss-tri-cam-robotics-p5-48xlarge-us-west-2 --job-status RUNNABLE
```

## Getting checkpoints back

openpi nests its own `<config>/<exp_name>/<step>` structure inside the checkpoint prefix, so the
path repeats:

```bash
aws s3 ls --profile manip-cluster \
  $OPENPI_S3_BASE_URI/pi05_yam_mugontree/mugontree_full_v1/checkpoints/pi05_yam_mugontree/mugontree_full_v1/

aws s3 sync --profile manip-cluster \
  $OPENPI_S3_BASE_URI/pi05_yam_mugontree/mugontree_full_v1/checkpoints/pi05_yam_mugontree/mugontree_full_v1/29999 \
  ./checkpoints/pi05_yam_mugontree/mugontree_full_v1/29999
```

## Launcher reference

Everything after the bare `--` is forwarded verbatim to `scripts/train.py`, so anything tyro
exposes there works (`--lr-schedule.peak-lr 5e-5`, `--ema-decay 0.999`, `--num-workers 8`,
`--overwrite`, ...). `--exp-name`, `--checkpoint-base-dir` and `--assets-base-dir` are set by
`train_entry.py` and should not be passed through.

Two defaults the launcher injects unless you override them:

- **`--fsdp-devices 8`** — shard across the whole node. A pi0.5 full fine-tune at batch 32 sits near
  75GB of an 80GB card even with 4-way FSDP, so pure data parallelism is not a safe default.
- **`--num-workers 32`** — openpi's default of 2 starves the GPUs on video-backed datasets.
  `ykorkmaz/yam_hang_mug_on_mug_tree` stores three 720p h264 cameras and is random-seek decoded per
  sample, measured at **656 ms/sample — 1.5 samples/s per worker**. At batch 32 that predicts
  `32 / (2 x 1.5) = 10.7 s/it`, matching an observed ~11 s/it: the loader was the entire step time,
  with the GPUs idle. 32 workers gives ~48 samples/s against 192 vCPUs.

  Datasets stored as `image` features rather than `video` (the LIBERO ones here, for example) skip
  decoding entirely and are fine on far fewer workers.

  The deeper fix for video datasets is re-encoding near the model's input size: openpi resizes every
  frame to 224x224 (`model.py:47`), so 720p decodes ~18x more pixels per camera than are kept.
  Re-encoding to ~256x256 would cut decode cost roughly an order of magnitude and shrink the dataset.

If a run OOMs, cut EMA first (`--ema-decay None`) — it keeps a second copy of every parameter.

Note that `--batch-size` is the **global** batch: `train.py` requires it to be divisible by the
device count and shards it across all 8 GPUs, so `--batch-size 32` is 4 samples per H100.

Useful launcher flags:

- `--instance-type p5|p5e|p5en|p4de|p6` and `--queue-name <family>` — the full queue name is
  `fss-{queue_name}-{instance}-{region}`, so the default resolves to
  `fss-tri-cam-robotics-p5-48xlarge-us-west-2`. Spot queue families (`*-spot`) additionally need
  `--use-spot`; the launcher fails fast on a mismatch rather than after the build.
- `--training-plan-arn` — **required** for any queue ending in `-training-plan` (currently `p5e`).
  Such a queue is reserved-capacity: a job submitted without the ARN is *accepted* by Batch, gets a
  real job id, and is then never scheduled — holding head-of-line and blocking everyone behind it
  until an operator kills it. Nothing at submit time hints at this, so the launcher refuses it.
  `aws sagemaker list-training-plans` lists the live ARNs and free capacity.
- `--max-attempts` (default 5) — retries are restricted to infrastructure faults (`Host EC2*`,
  `*InternalServerError*`, `*CapacityError*`); anything else exits immediately, so a bad config
  cannot burn every attempt. A retried attempt restores the latest checkpoint from S3 and resumes.
- `--compute-norm-stats` — runs `scripts/compute_norm_stats.py` in the container before training.
  Needed when `assets/<config>/<asset_id>/norm_stats.json` does not exist locally, since `assets/`
  is gitignored and gets baked into the image from your working tree.
- `--base-checkpoint s3://...` — overrides the weight loader's path. See "GCS egress" below.
- `--skip-build` — reuse the image already tagged `latest` in your ECR repo. Only safe when
  nothing under `src/`, `scripts/` or `assets/` changed.
- `--checkpoint-s3-uri s3://...` — overrides the checkpoint prefix derived from `--s3-base-uri`.

## Data

`--data-source hub` (default) pulls the config's `repo_id` from HuggingFace inside the container on
every job start, using `HF_TOKEN` from `secrets.env`.

`--data-source s3` mounts a prefix you uploaded beforehand:

```bash
uv run --group sagemaker python sagemaker/upload_dataset.py \
    --repo-id ykorkmaz/yam_hang_mug_on_mug_tree \
    --s3-prefix s3://robotics-cam-checkpoints/<firstname-lastname>/lerobot

# ... then add to the launch command:
#   --data-source s3 --s3-data-uri s3://robotics-cam-checkpoints/<firstname-lastname>/lerobot
```

The prefix must preserve LeRobot's `<namespace>/<dataset_name>/` layout, because the container
points `HF_LEROBOT_HOME` at the mount and LeRobot resolves a dataset to
`HF_LEROBOT_HOME/<repo_id>`. `upload_dataset.py` writes that layout for you. The channel uses
SageMaker `File` mode rather than `FastFile`: LeRobot random-seeks into parquet shards and mp4
episodes on every step, which a FUSE mount serves badly.

## Checkpoints and resuming

Checkpoints are written to `/opt/ml/cache/checkpoints/<config>/<exp_name>` and mirrored by
`train_entry.py` to `<s3-base-uri>/<config>/<exp_name>/checkpoints`: once every 15 minutes while
training runs, and once more after it exits (including on failure, and on the SIGTERM that precedes
a spot reclaim).

**SageMaker's own checkpointing is deliberately not used.** Setting `checkpoint_local_path` /
`checkpoint_s3_uri` starts a sync agent that watches `/opt/ml/checkpoints` and uploads files as
they appear — which races orbax, since orbax writes into `.orbax-checkpoint-tmp-*` directories and
reads its own metadata back during finalize. In practice orbax wrote `array_metadatas/process_0`
and read it back empty seconds later:

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

That fails the save and kills the job, and it hit on the very first save. Writing to unsynced
instance storage and uploading explicitly removes the race. Do not re-enable those estimator
arguments.

On startup `train_entry.py` restores **only the newest** step from S3, plus the run's top-level
files. Pulling every retained step would be unbounded — at ~50GiB each that exhausts the volume —
and openpi only ever restores the latest. `wandb_id.txt` comes down with the top-level files, which
is what keeps the W&B run continuous across a restart.

If anything was restored, `--resume` is appended automatically, so a requeued or spot-interrupted
job picks up where it left off (openpi downgrades this to a fresh start on its own if no step was
written yet). The prefix is keyed on config and `--exp-name` rather than the timestamped job name,
so relaunching the same experiment resumes it instead of starting over.

To force a clean restart under the same `--exp-name`, pass `-- --overwrite`.

## GCS egress

Configs default their weight loader to `gs://openpi-assets/checkpoints/pi05_base/params` (~7GB).
`openpi.shared.download` routes that bucket through `gsutil`, which the image installs, and the
download happens once per job from a container in AWS. If that is slow or blocked in your VPC,
mirror it into S3 once and pass `--base-checkpoint`:

```bash
gsutil -m cp -r "gs://openpi-assets/checkpoints/pi05_base/params" ./pi05_base_params
aws s3 sync ./pi05_base_params s3://robotics-cam-checkpoints/<firstname-lastname>/openpi/pi05_base/params --profile manip-cluster
# ... then: --base-checkpoint s3://robotics-cam-checkpoints/<firstname-lastname>/openpi/pi05_base/params
```

`s3fs` is installed in the image specifically so fsspec can read those `s3://` URIs (openpi itself
only ships `fsspec[gcs]`).

## Notes

- The image's `/opt/ml/code/.venv` is a separate uv environment from the DLC's python. The DLC's
  toolkit launches `train_entry.py` with the system python and the script re-execs into the venv.
  Anything you add to `pyproject.toml` needs `uv lock` before the next build, since the Dockerfile
  installs with `--frozen`.
- Caches (`OPENPI_DATA_HOME`, `HF_HOME`, `HOME`) live under `/opt/ml/cache` on the ML storage
  volume, not the container filesystem. `--volume-size` defaults to 500GB; p5/p4de instances back
  this with local NVMe.
- The launcher writes your uncommitted diff into the image and exposes it as
  `OPENPI_GIT_DIFF_FILE`, alongside `OPENPI_GIT_COMMIT` / `OPENPI_GIT_BRANCH`.
