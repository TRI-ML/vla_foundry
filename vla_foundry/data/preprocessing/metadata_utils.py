### These functions are used to create metadata for preprocessing robotics data.

import ast
import datetime
import hashlib
import os
import platform
import subprocess
import sys
from typing import Any

import fsspec
from draccus.parsers import encoding as _draccus_encoding

from vla_foundry.data.preprocessing.robotics.preprocess_params import PreprocessParams


def find_repo_root(start_path):
    current_path = os.path.abspath(start_path)
    while current_path != "/":
        if (
            os.path.isdir(os.path.join(current_path, ".git"))
            or os.path.isfile(os.path.join(current_path, "pyproject.toml"))
            or os.path.isfile(os.path.join(current_path, "setup.py"))
        ):
            return current_path
        current_path = os.path.dirname(current_path)
    return None


def get_python_dependencies(file_path: str, repo_root: str | None = None, visited: set | None = None) -> set:
    """Dynamically extract Python file dependencies by parsing imports."""
    if visited is None:
        visited = set()

    if file_path in visited:
        return set()

    visited.add(file_path)
    dependencies = {file_path}

    # Try to determine repo root if not provided
    if repo_root is None:
        repo_root = find_repo_root(file_path)

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    module_name = node.module
                    if module_name and module_name.startswith("vla_foundry"):
                        # Convert module path to file path relative to repo root
                        module_parts = module_name.split(".")
                        potential_paths = [
                            os.path.join(repo_root, "/".join(module_parts) + ".py"),
                            os.path.join(repo_root, "/".join(module_parts), "__init__.py"),
                        ]

                        for potential_path in potential_paths:
                            if os.path.exists(potential_path):
                                # Recursively get dependencies
                                sub_deps = get_python_dependencies(potential_path, repo_root, visited)
                                dependencies.update(sub_deps)
                                break

                    # Handle relative imports from current directory
                    if not module_name:  # relative import like "from . import"
                        current_dir = os.path.dirname(file_path)
                        for alias in node.names:
                            if alias.name != "*":
                                rel_path = os.path.join(current_dir, alias.name + ".py")
                                if os.path.exists(rel_path):
                                    sub_deps = get_python_dependencies(rel_path, repo_root, visited)
                                    dependencies.update(sub_deps)

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        if module_name.startswith("vla_foundry"):
                            # Convert module path to file path relative to repo root
                            module_parts = module_name.split(".")
                            potential_paths = [
                                os.path.join(repo_root, "/".join(module_parts) + ".py"),
                                os.path.join(repo_root, "/".join(module_parts), "__init__.py"),
                            ]

                            for potential_path in potential_paths:
                                if os.path.exists(potential_path):
                                    sub_deps = get_python_dependencies(potential_path, repo_root, visited)
                                    dependencies.update(sub_deps)
                                    break

        # Also check for direct file imports (like spartan_data_explorer)
        current_dir = os.path.dirname(file_path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and not node.module.startswith(".")
                and "." not in node.module
            ):
                # Handle direct imports from files in same directory
                # Check if it's a module file in the same directory
                potential_file = os.path.join(current_dir, node.module + ".py")
                if os.path.exists(potential_file):
                    sub_deps = get_python_dependencies(potential_file, repo_root, visited)
                    dependencies.update(sub_deps)

    except Exception as e:
        print(f"Warning: Could not analyze dependencies for {file_path}: {e}")

    return dependencies


def check_preprocessing_related_changes() -> tuple[bool, list[str]]:
    """Check if uncommitted changes affect preprocessing code or dependencies."""

    # Get the main preprocessing script path
    script_path = os.path.abspath(__file__)

    # Dynamically discover Python dependencies
    print("🔍 Analyzing Python dependencies...")
    repo_root = find_repo_root(script_path)
    python_dependencies = get_python_dependencies(script_path, repo_root)

    # Also include package management files
    additional_critical_files = ["requirements.txt", "pyproject.toml", "setup.py", "environment.yml"]

    # Convert to relative paths and add additional files
    critical_files = set()

    for dep_path in python_dependencies:
        if os.path.exists(dep_path):
            try:
                rel_path = os.path.relpath(dep_path, repo_root)
                critical_files.add(rel_path)
            except ValueError:
                # If relative path calculation fails, use absolute path
                critical_files.add(dep_path)

    # Add additional critical files
    for additional_file in additional_critical_files:
        additional_file_path = os.path.join(repo_root, additional_file)
        if os.path.exists(additional_file_path):
            critical_files.add(additional_file)

    print(f"📋 Found {len(critical_files)} critical dependency files")

    # Files that are relevant but less critical
    relevant_files = [
        "vla_foundry/data/preprocessing/create_data.sh",  # Processing configuration
    ]

    # Files to explicitly exclude (documentation, tests, etc.)
    exclude_patterns = [
        "README",
        ".md",
        "_test.py",
        "test_",
        "/tests/",
        ".txt",  # Like episode_description.txt
        "validate_",
        "example_",
        "_old.py",
        "pickle_to_raw.py",  # Utility script, not used by preprocessing
        "create_test_data.py",  # Test utility
    ]

    try:
        # Get list of changed files
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=repo_root)
        if result.returncode != 0:
            return False, []

        changed_files = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                # Extract filename (skip the status prefix)
                filename = line[3:].strip()
                # Make filename relative to repo_root (in case it's not already)
                abs_path = os.path.abspath(os.path.join(repo_root, filename))
                rel_path = os.path.relpath(abs_path, repo_root)
                changed_files.append(rel_path)

        # Check for critical changes (require tagging)
        critical_changes = []
        relevant_changes = []

        for file in changed_files:
            # First check if file should be excluded
            should_exclude = False
            for exclude_pattern in exclude_patterns:
                if exclude_pattern in file:
                    should_exclude = True
                    break

            if should_exclude:
                continue

            # Check if file is in critical dependencies (exact match or path contains)
            file_matched = False

            # Exact match first
            if file in critical_files:
                critical_changes.append(file)
                file_matched = True
            else:
                # Check if any critical file path contains this changed file
                for critical_path in critical_files:
                    if critical_path in file or file in critical_path:
                        critical_changes.append(file)
                        file_matched = True
                        break

            if not file_matched:
                # Check relevant but non-critical files
                for relevant_path in relevant_files:
                    if relevant_path in file:
                        relevant_changes.append(file)
                        break

        # Return True if there are critical changes, and all relevant files
        all_relevant = critical_changes + relevant_changes
        return len(critical_changes) > 0, all_relevant, critical_changes, relevant_changes

    except Exception:
        return False, [], [], []


def create_preprocessing_tag(dataset_name: str = None, preprocessing_type: str = None) -> str:
    """Create a git tag for the current preprocessing code state with more explicit naming."""
    try:
        # Generate tag name with timestamp and context
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        tag_parts = ["preprocessing"]

        # Add preprocessing type for clarity
        if preprocessing_type:
            clean_type = preprocessing_type.replace("_", "-")
            tag_parts.append(clean_type)

        # Add dataset name
        if dataset_name:
            # Clean dataset name for tag
            clean_name = dataset_name.replace("/", "_").replace(" ", "_").replace(":", "_")
            tag_parts.append(clean_name[:20])  # Limit length

        # Add timestamp
        tag_parts.append(f"v{timestamp}")

        tag_name = "_".join(tag_parts)

        # Create descriptive commit message
        commit_msg_parts = [f"Preprocessing code snapshot {timestamp}"]
        if preprocessing_type:
            commit_msg_parts.append(f"for {preprocessing_type}")
        if dataset_name:
            commit_msg_parts.append(f"dataset: {dataset_name}")

        commit_message = " ".join(commit_msg_parts)

        # Create the tag
        result = subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", commit_message],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )
        if result.returncode != 0:
            print(f"Warning: Failed to create git tag: {result.stderr}")
            return ""

        # Try to push the tag (don't fail if this doesn't work)
        result = subprocess.run(
            ["git", "push", "origin", tag_name], capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        if result.returncode != 0:
            print(f"Warning: Failed to push git tag (continuing anyway): {result.stderr}")
        else:
            print(f"✅ Created and pushed git tag: {tag_name}")

        return tag_name

    except Exception as e:
        print(f"Warning: Failed to create git tag: {e}")
        return ""


def get_git_info(auto_tag: bool = True) -> dict[str, str]:
    """Get git repository information for code version tracking."""
    git_info = {}

    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        if result.returncode == 0:
            git_info["commit_hash"] = result.stdout.strip()
    except Exception:
        git_info["commit_hash"] = "unknown"

    try:
        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()
    except Exception:
        git_info["branch"] = "unknown"

    try:
        # Check if there are uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        if result.returncode == 0:
            has_changes = bool(result.stdout.strip())
            git_info["has_uncommitted_changes"] = has_changes

            # If there are changes, check if they're preprocessing-related
            if has_changes:
                requires_tag, all_files, critical_files, relevant_files = check_preprocessing_related_changes()
                git_info["has_preprocessing_related_changes"] = len(all_files) > 0
                git_info["preprocessing_related_files"] = all_files
                git_info["critical_preprocessing_files"] = critical_files
                git_info["relevant_preprocessing_files"] = relevant_files

                # Show all files but distinguish between critical and relevant
                if all_files:
                    if critical_files:
                        print("⚠️  Found uncommitted changes to CRITICAL preprocessing code:")
                        for file in critical_files:
                            print(f"    🔴 {file}")

                    if relevant_files:
                        print("📝 Found uncommitted changes to relevant preprocessing files:")
                        for file in relevant_files:
                            print(f"    🟡 {file}")

                # Create a tag only if there are critical changes and auto_tag is enabled
                if requires_tag:
                    if auto_tag:
                        print("🏷️  Creating git tag for reproducibility (critical changes detected)...")
                        # Extract dataset info for more descriptive tag
                        dataset_name = None
                        preprocessing_type = "robotics_data"  # Default type
                        tag_name = create_preprocessing_tag(dataset_name, preprocessing_type)
                        if tag_name:
                            git_info["preprocessing_tag"] = tag_name
                            git_info["commit_hash_for_reproduction"] = git_info["commit_hash"]
                            print(f"📌 Use git tag '{tag_name}' to reproduce this exact code state")
                        else:
                            git_info["preprocessing_tag"] = "failed_to_create"
                    else:
                        print(
                            "🏷️  Auto-tagging disabled. "
                            "Consider manually creating a git tag for reproducibility of critical changes."
                        )
                        git_info["auto_tag_disabled"] = True
                elif all_files:
                    print("ℹ️  Only non-critical files changed. No git tag needed for reproducibility.")

            else:
                git_info["has_preprocessing_related_changes"] = False
                git_info["preprocessing_related_files"] = []
                git_info["critical_preprocessing_files"] = []
                git_info["relevant_preprocessing_files"] = []

    except Exception:
        git_info["has_uncommitted_changes"] = "unknown"
        git_info["has_preprocessing_related_changes"] = "unknown"

    try:
        # Get latest commit message
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%s"], capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        if result.returncode == 0:
            git_info["latest_commit_message"] = result.stdout.strip()
    except Exception:
        git_info["latest_commit_message"] = "unknown"

    # Add remote URL for full reproducibility info
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )
        if result.returncode == 0:
            git_info["remote_url"] = result.stdout.strip()
    except Exception:
        git_info["remote_url"] = "unknown"

    return git_info


def get_source_data_info(source_path: str, episodes: list[str]) -> dict[str, str]:
    """Get information about the source data."""
    source_info = {
        "source_path": source_path,
        "num_episodes": len(episodes),
        "episode_paths": episodes[:10],  # Store first 10 for reference
        "total_episodes_available": len(episodes),
    }

    # Try to get modification times of some episodes for data versioning
    try:
        fs, _ = fsspec.core.url_to_fs(source_path)
        sample_episodes = episodes[:3]  # Check first 3 episodes
        mod_times = []

        for episode in sample_episodes:
            try:
                if source_path.startswith("s3://"):
                    # For S3, try to get object info
                    fs_path = episode[5:] if episode.startswith("s3://") else episode
                    info = fs.info(fs_path)
                    if "LastModified" in info:
                        mod_times.append(info["LastModified"].isoformat())
                else:
                    # For local files
                    stat = fs.stat(episode)
                    mod_times.append(datetime.datetime.fromtimestamp(stat["mtime"]).isoformat())
            except Exception:
                continue

        source_info["sample_episode_modification_times"] = mod_times

        # Create a simple hash of episode paths for data version tracking
        episode_hash = hashlib.md5("\n".join(sorted(episodes)).encode()).hexdigest()
        source_info["episode_list_hash"] = episode_hash

    except Exception as e:
        source_info["source_data_info_error"] = str(e)

    return source_info


def create_processing_metadata(args: PreprocessParams, episodes: list[str]) -> dict[str, Any]:
    """Create comprehensive metadata about the processing run."""

    # Get command line information
    command_line = {
        "script_name": sys.argv[0],
        "full_command": " ".join(sys.argv),
        "arguments": _draccus_encoding.encode(args),
    }

    # Get environment information
    environment = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "processor": platform.processor(),
        "python_executable": sys.executable,
        "working_directory": os.getcwd(),
        "user": os.environ.get("USER", "unknown"),
        "timestamp_captured": datetime.datetime.now().isoformat(),
    }

    # Get git information (skip if testing flag is set)
    if args.skip_git_tagging:
        git_info = {"skip_git_tagging": True, "commit_hash": "test", "branch": "test"}
    else:
        git_info = get_git_info(auto_tag=args.auto_tag)

    # Get source data information
    source_data_info = get_source_data_info(args.source_episodes, episodes)

    # Package versions (try to get key dependencies)
    try:
        # Use modern importlib.metadata instead of deprecated pkg_resources
        try:
            from importlib.metadata import PackageNotFoundError, version
        except ImportError:
            # Fallback for Python < 3.8
            from importlib_metadata import PackageNotFoundError, version

        key_packages = ["numpy", "fsspec", "PIL", "tqdm", "boto3", "webdataset"]
        package_versions = {}
        for pkg in key_packages:
            try:
                # Handle special case for PIL package name
                pkg_name = "Pillow" if pkg == "PIL" else pkg
                package_versions[pkg] = version(pkg_name)
            except PackageNotFoundError:
                package_versions[pkg] = "not_found"
            except Exception:
                package_versions[pkg] = "unknown"
        environment["package_versions"] = package_versions
    except ImportError:
        # If importlib.metadata is not available, fall back gracefully
        environment["package_versions"] = "unavailable_importlib_metadata_missing"
    except Exception:
        environment["package_versions"] = "unavailable"

    # Create reproducibility instructions based on git state
    reproducibility_notes = []

    if git_info.get("preprocessing_tag"):
        # If we created a tag, use that for reproduction
        reproducibility_notes.extend(
            [
                f"EXACT REPRODUCTION: Use git tag '{git_info['preprocessing_tag']}'",
                "Commands to reproduce:",
                f"  git clone {git_info.get('remote_url', 'REPO_URL')}",
                f"  git checkout {git_info['preprocessing_tag']}",
                f"  {command_line['full_command']}",
                "",
                "This tag captures the exact code state including uncommitted changes used for this dataset.",
            ]
        )
    elif git_info.get("has_uncommitted_changes"):
        # If there are uncommitted changes but no tag was created
        reproducibility_notes.extend(
            [
                f"WARNING: Dataset created with uncommitted changes to commit {git_info.get('commit_hash', 'unknown')}",
                "For exact reproduction, the following files had uncommitted changes:",
            ]
        )
        for file in git_info.get("preprocessing_related_files", []):
            reproducibility_notes.append(f"  - {file}")
        reproducibility_notes.extend(
            [
                "",
                "Basic reproduction (may differ due to uncommitted changes):",
                f"  git clone {git_info.get('remote_url', 'REPO_URL')}",
                f"  git checkout {git_info.get('commit_hash', 'COMMIT_HASH')}",
                f"  {command_line['full_command']}",
            ]
        )
    else:
        # Clean state - straightforward reproduction
        reproducibility_notes.extend(
            [
                "CLEAN REPRODUCTION: No uncommitted changes",
                "Commands to reproduce:",
                f"  git clone {git_info.get('remote_url', 'REPO_URL')}",
                f"  git checkout {git_info.get('commit_hash', 'COMMIT_HASH')}",
                f"  {command_line['full_command']}",
            ]
        )

    reproducibility_notes.extend(
        [
            "",
            "Additional requirements:",
            "- Ensure the source data at the specified paths is unchanged (check episode_list_hash)",
            "- Use the same package versions if possible for identical results",
            "- Use the same hardware/OS for identical performance characteristics",
        ]
    )

    # Combine all metadata
    metadata = {
        "metadata_version": "1.0",
        "created_at": datetime.datetime.now().isoformat(),
        "command_line": command_line,
        "environment": environment,
        "git_info": git_info,
        "source_data": source_data_info,
        "processing": {},
        "reproducibility_notes": reproducibility_notes,
    }

    return metadata
