"""SageMaker launch smoke test.

Verifies prerequisites without building an image or submitting a job.

Usage:
    uv run --group sagemaker python sagemaker/smoke_test.py --user firstname.lastname
"""

import argparse
import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent
# Jobs run in the shared-compute account; checkpoints and artifacts go to manip-cluster buckets.
SM_PROFILE = "shared-compute"
SM_ARN = "arn:aws:iam::141701954645:role/CAM-Robotics-Sagemaker-role-us-west-2"
SM_REGION = "us-west-2"
SHARED_COMPUTE_ACCOUNT = "141701954645"

_OK = "\033[32m OK\033[0m"
_FAIL = "\033[31m FAIL\033[0m"


class _Missing:
    """Stand-in result for a command whose binary is not installed at all."""

    returncode = 127

    def __init__(self, cmd):
        self.stdout = ""
        self.stderr = f"{cmd[0]}: command not found"


def _run(cmd, **kw):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)
    except FileNotFoundError:
        return _Missing(cmd)


def _check(label, cmd, hint):
    r = _run(cmd)
    ok = r.returncode == 0
    print(f"  [{_OK if ok else _FAIL}] {label}")
    if not ok:
        print(f"        hint: {hint}")
        if r.stderr.strip():
            print(f"        {r.stderr.strip()[:200]}")
    return ok


def _report(label, ok, hint=""):
    print(f"  [{_OK if ok else _FAIL}] {label}")
    if not ok and hint:
        print(f"        hint: {hint}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True)
    p.add_argument("--sm-profile", default=SM_PROFILE)
    p.add_argument("--region", default=SM_REGION)
    # Queue families live in different accounts, so the account a correct setup authenticates to
    # depends on where the job will be submitted.
    p.add_argument("--sm-arn", default=SM_ARN, help="Execution role ARN the jobs will run under.")
    p.add_argument("--config", default=None, help="openpi config name to validate, e.g. pi05_yam_mugontree.")
    p.add_argument(
        "--s3-base-uri",
        default=os.environ.get("OPENPI_S3_BASE_URI"),
        help="Prefix in a manip-cluster bucket for checkpoints and artifacts. Checked for write access.",
    )
    args = p.parse_args()

    profile, region = args.sm_profile, args.region
    print(f"SageMaker smoke test  (user={args.user}, profile={profile}, region={region})\n")

    passed = True

    # SSO session is valid — required for all subsequent AWS calls.
    passed &= _check(
        f"AWS profile '{profile}' authenticated",
        ["aws", "--profile", profile, "sts", "get-caller-identity"],
        f"aws sso login --profile {profile}",
    )

    # Docker must be running before we attempt image build or ECR login.
    passed &= _check("Docker daemon running", ["docker", "info"], "sudo systemctl start docker")

    # ECR login: fetch a short-lived token, pipe it to docker login against the account registry
    # hostname (not a full image URI — that was a past bug).
    acct = _run(
        [
            "aws",
            "--profile",
            profile,
            "--region",
            region,
            "sts",
            "get-caller-identity",
            "--query",
            "Account",
            "--output",
            "text",
        ]
    )
    if acct.returncode == 0:
        account = acct.stdout.strip()
        registry = f"{account}.dkr.ecr.{region}.amazonaws.com"
        token = _run(["aws", "ecr", "get-login-password", "--region", region, "--profile", profile])
        login = (
            _run(["docker", "login", "--username", "AWS", "--password-stdin", registry], input=token.stdout)
            if token.returncode == 0
            else token
        )
        # Also check stderr: some Docker versions exit 0 but fail to save credentials (broken
        # credential helper), which causes docker push to fail later.
        cred_err = "error storing credentials" in login.stderr
        ecr_ok = login.returncode == 0 and not cred_err
        print(f"  [{_OK if ecr_ok else _FAIL}] ECR login ({registry})")
        if not ecr_ok:
            print(f"        {(login.stderr.strip() or login.stdout.strip())[:200]}")
            if cred_err:
                print("        hint: broken credential helper — edit ~/.docker/config.json, remove 'credsStore'")
        passed &= ecr_ok

        # No check that the DLC base image exists: the BatchOperator role can authenticate to the
        # Deep Learning Containers registry (763104351884) and pull from it, but not call
        # DescribeImages on it, so any such probe fails regardless of whether the tag is valid. A
        # wrong tag surfaces immediately as a docker pull error at the start of the build anyway.

        # Verify the authenticated account matches the ARN we submit jobs under.
        arn_account = args.sm_arn.split(":")[4]
        passed &= _report(
            f"ARN account matches ({account})",
            account == arn_account,
            f"--sm-arn targets {arn_account}; pass the ARN for the account owning your queue",
        )
        passed &= _report(
            f"submitting from shared-compute ({SHARED_COMPUTE_ACCOUNT})",
            account == SHARED_COMPUTE_ACCOUNT,
            f"the batch queues live there; use --sm-profile {SM_PROFILE}",
        )
    else:
        passed &= _report("ECR login (could not resolve account)", ok=False)

    # Checkpoints must land in a manip-cluster bucket, so confirm this profile can actually write
    # there cross-account before a multi-day job discovers otherwise at its first save_interval.
    if args.s3_base_uri:
        probe = f"{args.s3_base_uri.rstrip('/')}/.openpi_smoke_test"
        wrote = _run(
            ["aws", "s3", "cp", "-", probe, "--profile", profile, "--region", region],
            input="openpi smoke test\n",
        )
        if _report(f"S3 base writable ({args.s3_base_uri})", wrote.returncode == 0, wrote.stderr.strip()[:200]):
            # The BatchOperator permission set grants Get/Put but not Delete, so this usually fails.
            # The probe key is fixed, so at worst one harmless object is left behind and overwritten
            # by the next run.
            if _run(["aws", "s3", "rm", probe, "--profile", profile, "--region", region]).returncode != 0:
                print(f"        note: could not remove the probe object, left at {probe}")
        else:
            passed = False
    else:
        print("  [ SKIP ] S3 base writable (pass --s3-base-uri or set OPENPI_S3_BASE_URI)")

    # sagemaker + boto3 live in the optional uv dep group, not installed by default.
    passed &= _check(
        "sagemaker + boto3 importable",
        ["uv", "run", "--group", "sagemaker", "python", "-c", "import sagemaker, boto3"],
        "uv sync --group sagemaker",
    )

    # secrets.env is read unconditionally by launch_training.py at startup.
    secrets_path = REPO_ROOT / "secrets.env"
    secrets = {}
    if secrets_path.exists():
        for raw_line in secrets_path.read_text().splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                secrets[k.strip()] = v.strip()
    passed &= _report(
        "secrets.env defines WANDB_API_KEY",
        "WANDB_API_KEY" in secrets,
        f"create {secrets_path} with WANDB_API_KEY=... (and HF_TOKEN=... for private datasets)",
    )

    # Informational, not a failure: launch_training.py falls back to the CAM robotics role when
    # neither --arn nor SAGEMAKER_ARN is given.
    effective_arn = os.environ.get("SAGEMAKER_ARN", args.sm_arn)
    source = "SAGEMAKER_ARN" if "SAGEMAKER_ARN" in os.environ else "default"
    print(f"  [{_OK}] execution role ({source}): {effective_arn}")

    if args.config:
        # Catches config typos and missing norm stats before a ~10 minute build and push.
        r = _run(
            [
                "uv",
                "run",
                "python",
                "-c",
                "import sys, openpi.training.config as c;"
                "cfg = c.get_config(sys.argv[1]);"
                "d = cfg.data.create(cfg.assets_dirs, cfg.model);"
                "print(cfg.assets_dirs / d.asset_id)",
                args.config,
            ],
            cwd=REPO_ROOT,
        )
        if _report(f"config '{args.config}' resolves", r.returncode == 0, r.stderr.strip()[-300:]):
            assets_dir = pathlib.Path(r.stdout.strip().splitlines()[-1])
            passed &= _report(
                f"norm stats present ({assets_dir.relative_to(REPO_ROOT)})",
                (assets_dir / "norm_stats.json").exists(),
                f"run scripts/compute_norm_stats.py --config-name={args.config}, or launch with --compute-norm-stats",
            )
        else:
            passed = False

    print()
    print("All checks passed." if passed else "One or more checks failed.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
