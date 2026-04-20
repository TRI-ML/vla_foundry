"""Upload and download VLA Foundry experiments to/from Hugging Face Hub.

Paths starting with ``hf://`` are resolved transparently via the HF cache:

    hf://your-org/my-vla-model                  → repo root (snapshot_download)
    hf://your-org/my-vla-model/config.yaml       → single file (hf_hub_download)
    hf://your-org/my-vla-model/checkpoints/checkpoint_11.pt → single file

Upload from S3::

    python -m vla_foundry.hf_hub push \\
        s3://your-bucket/your-path/vla_foundry/model_checkpoints/.../ \\
        your-org/my-vla-model --checkpoint 11

Load in code::

    from vla_foundry.hf_hub import resolve_hf_path
    local = resolve_hf_path("hf://your-org/my-vla-model/config.yaml")  # cached file
    local_dir = resolve_hf_path("hf://your-org/my-vla-model")          # cached dir
"""

import argparse
import logging
import os
import shutil
import subprocess
import tempfile

from huggingface_hub import HfApi, add_collection_item, hf_hub_download, snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_PREFIX = "hf://"

# Default collection that groups all Foundry-* releases. Passing --collection ""
# on the push CLI disables the auto-add.
DEFAULT_COLLECTION_SLUG = "TRI-ML/vla-foundry-69e2e472078e75c271a486d6"

# Config files to upload from the experiment root.
# Keys are the canonical repo name; values list source filenames to try (first match wins).
# This handles legacy names (e.g. preprocessing_configs.yaml → preprocessing_config.yaml).
EXPERIMENT_ROOT_FILES = {
    "config.yaml": ["config.yaml"],
    "config_model.yaml": ["config_model.yaml"],
    "config_normalizer.yaml": ["config_normalizer.yaml"],
    "config_processor.yaml": ["config_processor.yaml"],
    "preprocessing_config.yaml": ["preprocessing_config.yaml", "preprocessing_configs.yaml"],
    "processing_metadata.json": ["processing_metadata.json"],
    "stats.json": ["stats.json"],
}


def is_hf_path(path: str) -> bool:
    """Check if a path is an HF Hub path (hf://repo_id/...)."""
    return path.startswith(HF_PREFIX)


def normalize_checkpoint_locator(path: str) -> str:
    """Normalize a checkpoint locator for downstream loaders.

    Resolution rules, in order:
      1. ``hf://...``, ``s3://...``, and absolute local paths are left as-is.
      2. A path that exists on the local filesystem (relative or absolute) is
         treated as a local checkpoint directory. This keeps the common
         "train locally, eval locally" dev flow working with paths like
         ``experiments/my_run``.
      3. Anything else is assumed to be a bare HuggingFace repo ID
         (e.g. ``TRI-ML/my-model``) and gets the ``hf://`` prefix.
    """
    if not path.startswith(("hf://", "s3://", "/")) and not os.path.exists(path):
        return f"hf://{path}"
    return path


def parse_hf_path(path: str) -> tuple[str, str | None]:
    """Parse ``hf://repo_id[/subpath]`` into (repo_id, subpath).

    Returns:
        (repo_id, subpath) where subpath is None for repo root.
    """
    stripped = path[len(HF_PREFIX) :]
    # repo_id is owner/name (first two segments)
    parts = stripped.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Invalid HF path, expected hf://owner/repo[/subpath], got: {path}")
    repo_id = f"{parts[0]}/{parts[1]}"
    subpath = parts[2] if len(parts) > 2 else None
    if subpath is not None and subpath.strip() == "":
        subpath = None
    return repo_id, subpath


def resolve_hf_path(path: str, token: str | None = None) -> str:
    """Resolve an ``hf://`` path to a local cached file or directory.

    Uses the HF Hub cache so files are downloaded once and reused.

    Args:
        path: ``hf://repo_id`` for the whole repo, or ``hf://repo_id/file.yaml`` for a single file.
        token: Optional HF token.

    Returns:
        Local filesystem path (cached).
    """
    repo_id, subpath = parse_hf_path(path)

    if subpath is None:
        # Download the whole repo snapshot
        return snapshot_download(repo_id=repo_id, token=token)
    else:
        # Download a single file
        return hf_hub_download(repo_id=repo_id, filename=subpath, token=token)


def _find_local_checkpoint(
    experiment_path: str, checkpoint_num: int | None, include_optimizer: bool = False
) -> tuple[int, list[str]]:
    """Find checkpoint files in a local experiment directory.

    Searches both root and checkpoints/ subdirectory for checkpoint_*.pt files.

    Returns:
        (checkpoint_num, list of (local_path, repo_path) tuples)
    """
    candidates = []
    for dirpath in [experiment_path, os.path.join(experiment_path, "checkpoints")]:
        if not os.path.isdir(dirpath):
            continue
        for f in os.listdir(dirpath):
            if f.startswith("checkpoint_") and f.endswith(".pt"):
                num = f.replace("checkpoint_", "").replace(".pt", "")
                if num.isdigit():
                    candidates.append((int(num), os.path.join(dirpath, f)))

    if not candidates:
        raise RuntimeError(f"No checkpoint_*.pt files found in {experiment_path}")

    if checkpoint_num is None:
        checkpoint_num = max(c[0] for c in candidates)
        logger.info(f"Latest checkpoint: {checkpoint_num}")

    # Collect all files for this checkpoint number (checkpoint, ema, and optionally optimizer)
    result = []
    base_dir = os.path.dirname(next(path for num, path in candidates if num == checkpoint_num))
    prefixes = ["checkpoint_", "ema_"]
    if include_optimizer:
        prefixes.append("optimizer_")
    for prefix in prefixes:
        local_path = os.path.join(base_dir, f"{prefix}{checkpoint_num}.pt")
        if os.path.exists(local_path):
            result.append((local_path, f"checkpoints/{prefix}{checkpoint_num}.pt"))

    return checkpoint_num, result


def _find_s3_checkpoint(s3_path: str, checkpoint_num: int | None, include_optimizer: bool) -> tuple[int, list[str]]:
    """Find checkpoint files in an S3 experiment directory.

    Returns:
        (checkpoint_num, list of (s3_path, repo_path) tuples)
    """
    if checkpoint_num is None:
        result = subprocess.run(
            ["aws", "s3", "ls", f"{s3_path}/checkpoints/"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to list checkpoints: {result.stderr}")

        checkpoint_nums = []
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split()
            if not parts:
                continue
            name = parts[-1]
            if name.startswith("checkpoint_") and name.endswith(".pt"):
                num = name.replace("checkpoint_", "").replace(".pt", "")
                if num.isdigit():
                    checkpoint_nums.append(int(num))

        if not checkpoint_nums:
            raise RuntimeError("No checkpoints found")
        checkpoint_num = max(checkpoint_nums)
        logger.info(f"Latest checkpoint: {checkpoint_num}")

    prefixes = ["checkpoint_"]
    if include_optimizer:
        prefixes.append("optimizer_")

    # Check for EMA checkpoint
    ema_check = subprocess.run(
        ["aws", "s3", "ls", f"{s3_path}/checkpoints/ema_{checkpoint_num}.pt"],
        capture_output=True,
    )
    if ema_check.returncode == 0:
        prefixes.append("ema_")

    result = []
    for prefix in prefixes:
        filename = f"{prefix}{checkpoint_num}.pt"
        result.append((f"{s3_path}/checkpoints/{filename}", f"checkpoints/{filename}"))

    return checkpoint_num, result


def _bundle_vlm_config_if_applicable(experiment_path: str, is_s3: bool, tmpdir: str, api: HfApi, repo_id: str) -> None:
    """If the experiment's config_model.yaml references a VLM backbone checkpoint,
    fetch that VLM's config_model.yaml and upload it to the VLA repo as
    ``vlm_config_model.yaml`` so the published model is self-contained.
    """
    import yaml

    src_cfg = os.path.join(tmpdir, "config_model.yaml")
    if not os.path.exists(src_cfg):
        if is_s3:
            subprocess.run(
                ["aws", "s3", "cp", f"{experiment_path}/config_model.yaml", src_cfg], capture_output=True, check=False
            )
        else:
            candidate = os.path.join(experiment_path, "config_model.yaml")
            if os.path.exists(candidate):
                shutil.copyfile(candidate, src_cfg)
    if not os.path.exists(src_cfg):
        return

    with open(src_cfg) as f:
        cfg = yaml.safe_load(f) or {}
    vlb = cfg.get("vision_language_backbone", {}) if isinstance(cfg, dict) else {}
    if not isinstance(vlb, dict) or vlb.get("type") != "vlm_foundry_backbone":
        return

    # Find the VLM's experiment dir from vlm_experiment_dir or resume_from_checkpoint.
    vlm_exp_dir = vlb.get("vlm_experiment_dir")
    if not vlm_exp_dir:
        rfc = vlb.get("resume_from_checkpoint")
        if rfc:
            vlm_exp_dir = os.path.dirname(os.path.dirname(rfc))
    if not vlm_exp_dir:
        logger.warning("Could not locate VLM experiment dir for bundling; skipping vlm_config_model.yaml")
        return

    vlm_cfg_src = f"{vlm_exp_dir}/config_model.yaml"
    vlm_cfg_local = os.path.join(tmpdir, "vlm_config_model.yaml")
    if vlm_exp_dir.startswith("s3"):
        r = subprocess.run(["aws", "s3", "cp", vlm_cfg_src, vlm_cfg_local], capture_output=True)
        if r.returncode != 0:
            logger.warning(f"Failed to fetch VLM config from {vlm_cfg_src}: {r.stderr.decode()}")
            return
    else:
        if not os.path.exists(vlm_cfg_src):
            logger.warning(f"VLM config not found at {vlm_cfg_src}")
            return
        shutil.copyfile(vlm_cfg_src, vlm_cfg_local)

    api.upload_file(path_or_fileobj=vlm_cfg_local, path_in_repo="vlm_config_model.yaml", repo_id=repo_id)
    logger.info("Uploaded vlm_config_model.yaml")


def push_to_hub(
    experiment_path: str,
    repo_id: str,
    checkpoint_num: int | None = None,
    include_optimizer: bool = False,
    private: bool = False,
    token: str | None = None,
    collection_slug: str | None = DEFAULT_COLLECTION_SLUG,
):
    """Upload an experiment from a local directory or S3 to Hugging Face Hub.

    Args:
        experiment_path: Local path or S3 path to the experiment directory.
        repo_id: HF Hub repo ID (e.g. "your-org/my-vla-model").
        checkpoint_num: Specific checkpoint number to upload. If None, uploads the latest.
        include_optimizer: Whether to upload optimizer state (large, only needed for resuming training).
        private: Whether to create a private repo.
        token: HF token. If None, uses the cached token.
    """
    experiment_path = experiment_path.rstrip("/")
    is_s3 = experiment_path.startswith("s3")
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id, exist_ok=True, private=private)
    logger.info(f"Repo: https://huggingface.co/{repo_id}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Upload config files from experiment root
        for repo_name, source_candidates in EXPERIMENT_ROOT_FILES.items():
            local_path = None
            for candidate in source_candidates:
                if is_s3:
                    src = f"{experiment_path}/{candidate}"
                    candidate_path = os.path.join(tmpdir, candidate)
                    result = subprocess.run(["aws", "s3", "cp", src, candidate_path], capture_output=True)
                    if result.returncode == 0:
                        local_path = candidate_path
                        break
                else:
                    candidate_path = os.path.join(experiment_path, candidate)
                    if os.path.exists(candidate_path):
                        local_path = candidate_path
                        break
            if local_path is None:
                logger.debug(f"Skipped {repo_name} (not found)")
                continue

            api.upload_file(path_or_fileobj=local_path, path_in_repo=repo_name, repo_id=repo_id)
            logger.info(f"Uploaded {repo_name}")

        # 1b. If this is a VLA with a VLM backbone, bundle the VLM's config_model.yaml
        # as vlm_config_model.yaml so published repos are self-contained.
        _bundle_vlm_config_if_applicable(experiment_path, is_s3, tmpdir, api, repo_id)

        # 2. Find and upload checkpoint files
        if is_s3:
            checkpoint_num, checkpoint_files = _find_s3_checkpoint(experiment_path, checkpoint_num, include_optimizer)
        else:
            checkpoint_num, checkpoint_files = _find_local_checkpoint(
                experiment_path, checkpoint_num, include_optimizer
            )

        for src_path, repo_path in checkpoint_files:
            if is_s3:
                local_path = os.path.join(tmpdir, os.path.basename(src_path))
                logger.info(f"Downloading {os.path.basename(src_path)} from S3...")
                result = subprocess.run(["aws", "s3", "cp", src_path, local_path], capture_output=True)
                if result.returncode != 0:
                    logger.warning(f"Failed to download: {result.stderr.decode()}")
                    continue
            else:
                local_path = src_path

            size_gb = os.path.getsize(local_path) / 1e9
            logger.info(f"Uploading {repo_path} (~{size_gb:.1f} GB)...")
            api.upload_file(path_or_fileobj=local_path, path_in_repo=repo_path, repo_id=repo_id)
            logger.info(f"Uploaded {repo_path}")

            if is_s3:
                os.remove(local_path)

    if collection_slug:
        try:
            add_collection_item(
                collection_slug=collection_slug,
                item_id=repo_id,
                item_type="model",
                exists_ok=True,
                token=token,
            )
            logger.info(f"Added to collection: https://huggingface.co/collections/{collection_slug}")
        except Exception as e:
            logger.warning(f"Could not add to collection {collection_slug}: {e}")

    logger.info(f"Done! Model available at: https://huggingface.co/{repo_id}")
    logger.info(f"Load with: hf://{repo_id}")
    return repo_id


def main():
    parser = argparse.ArgumentParser(description="Upload/download VLA Foundry experiments to/from HF Hub")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Push command
    push_parser = subparsers.add_parser("push", help="Upload experiment to HF Hub")
    push_parser.add_argument("experiment_path", help="Local or S3 path to experiment directory")
    push_parser.add_argument("repo_id", help="HF Hub repo ID (e.g. your-org/my-vla-model)")
    push_parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint number (default: latest)")
    push_parser.add_argument("--include-optimizer", action="store_true", help="Include optimizer state")
    push_parser.add_argument("--private", action="store_true", help="Create private repo")
    push_parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_SLUG,
        help=f"HF collection slug to add the repo to (default: {DEFAULT_COLLECTION_SLUG}). Pass '' to skip.",
    )

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Download experiment from HF Hub to local cache")
    pull_parser.add_argument("repo_id", help="HF Hub repo ID (e.g. your-org/my-vla-model)")

    args = parser.parse_args()

    if args.command == "push":
        push_to_hub(
            experiment_path=args.experiment_path,
            repo_id=args.repo_id,
            checkpoint_num=args.checkpoint,
            include_optimizer=args.include_optimizer,
            private=args.private,
            collection_slug=args.collection or None,
        )
    elif args.command == "pull":
        local_path = resolve_hf_path(f"hf://{args.repo_id}")
        print(local_path)


if __name__ == "__main__":
    main()
