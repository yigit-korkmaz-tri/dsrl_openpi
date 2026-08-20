ARG AWS_REGION
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker

# Used by the SageMaker training toolkit to locate our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# ffmpeg: torchcodec / LeRobot decode the episode videos at every training step.
# gsutil: openpi.shared.download routes gs://openpi-assets (the pi05_base weights) through gsutil
#   and only falls back to gcsfs -- which fails to auth against that bucket -- when it is missing.
# git: uv resolves `lerobot` and `dlimp` from git revisions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/* \
    # awscli: train_entry.py mirrors checkpoints to S3 itself rather than using SageMaker's
    # checkpoint sync. The DLC normally ships it, but this makes that a guarantee.
    && pip install --no-cache-dir --upgrade uv gsutil awscli

# openpi pins jax[cuda12]==0.5.3 and torch==2.7.1, which conflict with what the DLC bakes into its
# python. So it gets its own uv-managed venv rather than installing over the image's environment;
# train_entry.py re-execs into that venv on startup.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/ml/code/.venv

WORKDIR /opt/ml/code

# Dependency layer: manifests only, so edits under src/ don't re-resolve or re-download the ~10GB
# of CUDA wheels. README.md and LICENSE are here because pyproject.toml references both.
COPY pyproject.toml uv.lock .python-version README.md LICENSE /opt/ml/code/
COPY packages/openpi-client/pyproject.toml /opt/ml/code/packages/openpi-client/
RUN uv sync --frozen --no-dev --no-install-workspace \
    && rm -rf /root/.cache/uv

# Source layer. assets/ carries the precomputed norm stats; without them training aborts unless the
# job is launched with --compute-norm-stats.
COPY packages/ /opt/ml/code/packages/
COPY src/ /opt/ml/code/src/
COPY scripts/ /opt/ml/code/scripts/
COPY assets/ /opt/ml/code/assets/
COPY sagemaker/ /opt/ml/code/sagemaker/
RUN uv sync --frozen --no-dev \
    # s3fs is not an openpi dependency (it only ships fsspec[gcs]), but it lets weight loaders and
    # AssetsConfig read s3:// URIs, which is how you mirror pi05_base out of GCS into your account.
    && uv pip install --no-cache-dir s3fs \
    && rm -rf /root/.cache/uv

ENV SAGEMAKER_PROGRAM=/opt/ml/code/sagemaker/train_entry.py
