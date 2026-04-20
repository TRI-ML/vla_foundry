import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import boto3
import draccus
from sagemaker.pytorch import PyTorch

import sagemaker
from vla_foundry.params.base_params import BaseParams
from vla_foundry.params.train_experiment_params import TrainExperimentParams

NAME = "vla_foundry"


def get_git_env_vars() -> dict[str, str]:
    """Capture git info as environment variables for passing to SageMaker workers."""
    cwd = os.path.dirname(os.path.dirname(__file__))

    def run_git(args: list[str]) -> str | None:
        result = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run_git(["git", "rev-parse", "HEAD"])
    if not commit:
        return {}

    changes = run_git(["git", "status", "--porcelain"])
    return {
        "VLA_GIT_COMMIT_HASH": commit,
        "VLA_GIT_BRANCH": run_git(["git", "branch", "--show-current"]) or "DETACHED",
        "VLA_GIT_REMOTE_URL": run_git(["git", "config", "--get", "remote.origin.url"]) or "unknown",
        "VLA_GIT_HAS_LOCAL_CHANGES": "true" if changes else "false",
    }


INSTANCE_MAPPER = {
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
    "p5en": "ml.p5en.48xlarge",
    "p6": "ml.p6-b200.48xlarge",
}


@dataclass(frozen=True)
class SageMakerRunParams(BaseParams):
    local: bool = field(default=False)
    user: str = field(default=None)
    name_prefix: str = field(default=None)

    # Volume size in GB
    volume_size: int = field(default=30)

    # AWS profile args
    region: str = field(default="us-west-2")
    profile: str = field(default="default")
    arn: str = field(default=None)

    # Instance args
    instance_count: int = field(default=1)
    instance_type: str = field(default="p4de")
    max_run: int = field(default=10)

    # Testing flags -- exit early before making expensive or remote calls.
    # check_config: parse and validate config, dump the hyperparameters yaml, then exit.
    # build_only: build and push the docker image, then exit without submitting to SageMaker.
    check_config: bool = field(default=False)
    build_only: bool = field(default=False)


@dataclass(frozen=True)
class SageMakerParams(TrainExperimentParams):
    sagemaker: SageMakerRunParams = field(default_factory=SageMakerRunParams)

    def __post_init__(self):
        if self.save_path is not None:
            logging.warning(
                "Save path is not None, but SageMaker will override it to /tmp"
                " because Sagemaker has its own volume mounted at /tmp."
            )
        object.__setattr__(self, "save_path", "/tmp")


def run_command(command):
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def remove_old_hyperparameters(path, expiration_days=3):
    """Remove old hyperparameters files."""
    for file in Path(path).glob("hyperparameters_*.yaml"):
        if file.stat().st_mtime < time.time() - expiration_days * 24 * 60 * 60:
            file.unlink()


GIT_DIFF_DIR = "sagemaker/git_diffs"


def remove_old_git_diffs(path, expiration_days=3):
    """Remove old git diff files."""
    for file in Path(path).glob("git_diff_*.txt"):
        if file.stat().st_mtime < time.time() - expiration_days * 24 * 60 * 60:
            file.unlink()


def save_git_diff(uuid: str) -> str:
    """Save git diff to a file that will be included in the Docker image.

    Returns the path to the file (for use in SageMaker container).
    """
    os.makedirs(GIT_DIFF_DIR, exist_ok=True)
    remove_old_git_diffs(GIT_DIFF_DIR, expiration_days=3)

    git_diff_file = f"{GIT_DIFF_DIR}/git_diff_{uuid}.txt"
    sagemaker_path = f"/opt/ml/code/{git_diff_file}"

    def run_git(cmd):
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""

    changes = run_git(["git", "status", "--porcelain"])
    if not changes:
        return sagemaker_path  # No diff file created, but return path anyway

    diff = run_git(["git", "diff", "HEAD"])
    untracked = run_git(["git", "ls-files", "--others", "--exclude-standard"])
    if untracked:
        diff += "\n\n# Untracked files:\n" + untracked

    with open(git_diff_file, "w") as f:
        f.write(diff)
    print(f"Saved git diff to {git_diff_file} ({len(diff)} bytes)")
    return sagemaker_path


def get_image(user, profile="default", region="us-east-1"):
    os.environ["AWS_PROFILE"] = f"{profile}"
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    assert account.isdigit(), f"Invalid account value: {account}"
    docker_dir = Path(__file__).parent
    algorithm_name = f"{user}-{NAME}"
    dockerfile_base = docker_dir / "Dockerfile"
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"

    login_cmd = (
        f"aws ecr get-login-password --region {region} --profile {profile} | "
        f"docker login --username AWS --password-stdin"
    )

    print("Building container")
    commands = [
        # Log in to Sagemaker account to get image.
        f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
        f"docker build --progress=plain -f {dockerfile_base} --build-arg AWS_REGION={region} -t {algorithm_name} .",
        f"docker tag {algorithm_name} {fullname}",
        f"{login_cmd} {fullname}",
        (
            f"aws --region {region} ecr describe-repositories --repository-names {algorithm_name} || "
            f"aws --region {region} ecr create-repository --repository-name {algorithm_name}"
        ),
    ]

    # Create command, making sure to exit if any part breaks.
    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"


SECRETS_FILE = "secrets.env"


def check_secrets_file():
    """Exit with a helpful message if secrets.env is missing."""
    if os.path.exists(SECRETS_FILE):
        return
    print(
        f"ERROR: {SECRETS_FILE} not found in the current directory ({os.getcwd()}).\n"
        f"Create it in the repo root with entries like:\n"
        f"  WANDB_API_KEY=<your wandb key>\n"
        f"  HF_TOKEN=<your huggingface token>\n"
        f"Lines starting with '#' and blank lines are ignored.\n"
        f"See sagemaker/README.md for details.",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    args = draccus.parse(config_class=SageMakerParams)

    assert args.sagemaker.user is not None, "--sagemaker.user is required (e.g., --sagemaker.user firstname.lastname)"
    check_secrets_file()

    # Save hyperparameters to a yaml file. Files are deleted after 3 days.
    uuid = str(uuid4())
    temp_file_path = f"sagemaker/configs/hyperparameters_{uuid}.yaml"
    hyperparameter_sagemaker_path = f"/opt/ml/code/configs/hyperparameters_{uuid}.yaml"
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    with open(temp_file_path, "w") as f:
        args_dict = draccus.parsers.encoding.encode(args)
        del args_dict["sagemaker"]
        print(args_dict)
        draccus.cfgparsing.save_config(args_dict, f)
    remove_old_hyperparameters("sagemaker/configs", expiration_days=3)

    # Save git diff to file (before docker build so it's included in image)
    git_diff_sagemaker_path = save_git_diff(uuid)

    # We probably want wandb logging and S3 saving for sagemaker runs
    assert args.remote_sync is not None
    assert args.wandb

    # Check that batch sizes align. We do this here because the world_size is not known during `__post_init__`.
    world_size = args.sagemaker.instance_count * 8
    combined_batch_size = world_size * args.hparams.per_gpu_batch_size
    assert args.hparams.global_batch_size % combined_batch_size == 0

    args = args.sagemaker
    assert args.instance_type in INSTANCE_MAPPER
    if args.arn is None:
        assert "SAGEMAKER_ARN" in os.environ, "Please specify --arn or set the SAGEMAKER_ARN environment variable"
        object.__setattr__(args, "arn", os.environ["SAGEMAKER_ARN"])

    if args.check_config:
        print(f"Config valid. Hyperparameters written to {temp_file_path}.")
        print("--sagemaker.check_config set -- exiting before docker build.")
        return

    image = get_image(
        args.user,
        region=args.region,
        profile=args.profile,
    )
    os.environ["AWS_DEFAULT_REGION"] = args.region

    if args.build_only:
        print(f"Image built and pushed: {image}")
        print("--sagemaker.build_only set -- exiting before SageMaker submission.")
        return

    ##########
    # Create session and make sure of account and region
    ##########
    sagemaker_session = sagemaker.Session(boto_session=boto3.session.Session(region_name=args.region))

    if args.local:
        from sagemaker.local import LocalSession

        sagemaker_session = LocalSession()

    role = args.arn
    # provide a pre-existing role ARN as an alternative to creating a new role
    role_name = role.split("/")[-1]
    print(f"SageMaker Execution Role:{role}")
    print(f"The name of the Execution role: {role_name}")

    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
    print(f"AWS account:{account}")

    ##########
    # Configure the training
    ##########
    def sanitize_name(name):
        name = name.replace("_", "-")
        clean = "".join(c if c.isalnum() or c == "-" else "" for c in name)
        clean = clean.strip("-")
        return clean or "job"

    base_job_name = sanitize_name(
        f"{args.name_prefix + '-' if args.name_prefix else ''}{args.user.replace('.', '-')}-{NAME}"
    )
    checkpoint_local_path = "/opt/ml/checkpoints"

    def get_job_name(base):
        now = datetime.now()
        # Format example: 2023-03-03-10-14-02-324
        date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}"
        # Ensure the job name follows SageMaker naming constraints: [a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}
        clean_base = sanitize_name(base)
        job_name = f"{clean_base}-{date_str}"
        job_name = job_name.lstrip("-")
        # Truncate if too long (SageMaker limit is 63 characters)
        if len(job_name) > 63:
            job_name = job_name[:63]
        # Remove trailing hyphens if any (truncation may have left some)
        job_name = job_name.rstrip("-")
        return job_name

    job_name = get_job_name(base_job_name)

    environment = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "WANDB_PROJECT": "vla_foundry",
        "NCCL_DEBUG": "INFO",
        "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "1",
        "SAGEMAKER_PROGRAM": "/opt/ml/code/vla_foundry/main.py",
        "FI_EFA_FORK_SAFE": "1",
        "VLA_GIT_DIFF_FILE": git_diff_sagemaker_path,
        "VLA_LAUNCHED_BY": args.user,
        **get_git_env_vars(),
    }
    with open("secrets.env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                environment[key.strip()] = value.strip().strip("\"'")

    estimator = PyTorch(
        entry_point="vla_foundry/main.py",
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters={"config_path": hyperparameter_sagemaker_path},
        role=role,
        image_uri=image,
        instance_count=args.instance_count,
        instance_type="local_gpu" if args.local else INSTANCE_MAPPER[args.instance_type],
        checkpoint_local_path=None if args.local else checkpoint_local_path,
        # Training using SMDataParallel Distributed Training Framework
        distribution={"torch_distributed": {"enabled": True}},
        # Max run 5 days
        max_run=args.max_run * 24 * 60 * 60,  # max_run days
        input_mode="FastFile",
        environment=environment,
        keep_alive_period_in_seconds=5 * 60,  # 5 minutes
        tags=[
            {"Key": "project", "Value": "vla_foundry"},
            {"Key": "owner", "Value": args.user},
        ],
        volume_size=args.volume_size,
    )

    # Validate job name before submission
    if not job_name or len(job_name) > 63:
        raise ValueError(f"Invalid job name length ({len(job_name)}): {job_name}")
    invalid_chars = [c for c in job_name if not (c.isalnum() or c == "-")]
    if invalid_chars:
        raise ValueError(f"Job name contains invalid characters {set(invalid_chars)}: {job_name}")

    try:
        estimator.fit(job_name=job_name, wait=False)
        print(f"Submitted {job_name}")
    except Exception as e:
        print(f"Failed to submit {job_name}: {str(e)}")
        raise


if __name__ == "__main__":
    main()
