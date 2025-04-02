import datetime
import json
import os
import re
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union

# Import from llm_scoring
import llm_scoring
import pandas as pd
import typer
import yaml
from huggingface_hub import HfApi
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Import from submission.py for authentication
from submission import get_hf_user

console = Console()


# from maxb2: https://github.com/tiangolo/typer/issues/86#issuecomment-996374166
def conf_callback(ctx: typer.Context, param: typer.CallbackParam, config: Optional[str] = None):
    if config is not None:
        typer.echo(f"Loading config file: {config}\n")
        try:
            with open(config, "r") as f:  # Load config file
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}  # Initialize the default map
            ctx.default_map.update(conf)  # Merge the config dict into default_map
        except Exception as ex:
            raise typer.BadParameter(str(ex))
    return config


class ScoreCategory(str, Enum):
    """Score categories used in evaluations.

    All categories are stored in uppercase but accessed as lowercase in code.
    """

    GRAMMAR = "GRAMMAR"
    COHERENCE = "COHERENCE"
    CREATIVITY = "CREATIVITY"
    CONSISTENCY = "CONSISTENCY"
    PLOT = "PLOT"
    OVERALL = "OVERALL"


class UploadTracker:
    """Tracks which files have been uploaded to avoid redundant uploads."""

    def __init__(self, state_file="state/upload_state.json", is_test_mode=False):
        # If default state file is used, apply test mode suffix
        if state_file == "state/upload_state.json" and is_test_mode:
            state_file = "state/upload_state_test.json"

        self.state_file = Path(state_file)
        # Create the state directory if it doesn't exist
        self.state_file.parent.mkdir(exist_ok=True, parents=True)

        # Check for old state file in root directory for backward compatibility
        old_state_file = Path(".upload_state.json")
        if old_state_file.exists() and not self.state_file.exists():
            # Migrate state from old location to new location
            console.print(f"[yellow]Migrating upload state from {old_state_file} to {self.state_file}[/yellow]")
            self.state_file.parent.mkdir(exist_ok=True, parents=True)
            old_state_file.rename(self.state_file)

        self.state = self._load_state()

    def _load_state(self):
        """Load saved upload state or create empty state."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except json.JSONDecodeError:
                console.print(f"[yellow]Warning: Could not parse upload state file. Creating new state.[/yellow]")
                return {"last_upload": {}, "last_readme_update": None}
        return {"last_upload": {}, "last_readme_update": None}

    def save_state(self):
        """Save current state to disk."""
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def should_upload_file(self, file_path: Path) -> bool:
        """Check if file needs upload based on modification time."""
        file_path_str = str(file_path)
        last_upload_time = self.state["last_upload"].get(file_path_str)

        if not file_path.exists():
            return False

        if last_upload_time is None:
            return True

        # Get file modification time as ISO timestamp
        mtime = datetime.datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        return mtime > last_upload_time

    def mark_file_uploaded(self, file_path: Path):
        """Mark a file as uploaded with current timestamp."""
        file_path_str = str(file_path)
        mtime = datetime.datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        self.state["last_upload"][file_path_str] = mtime
        self.save_state()

    def mark_readme_updated(self):
        """Mark the README as updated with current timestamp."""
        self.state["last_readme_update"] = datetime.datetime.now().isoformat()
        self.save_state()

    def should_update_readme(self, min_interval_hours=12) -> bool:
        """Check if README should be updated based on time interval."""
        last_update = self.state.get("last_readme_update")
        if last_update is None:
            return True

        last_update_time = datetime.datetime.fromisoformat(last_update)
        hours_since_update = (datetime.datetime.now() - last_update_time).total_seconds() / 3600
        return hours_since_update >= min_interval_hours


class ScoringTracker:
    """Tracks which submissions have been scored to avoid redundant scoring."""

    def __init__(self, state_file="state/scoring_state.json", is_test_mode=False):
        # If default state file is used, apply test mode suffix
        if state_file == "state/scoring_state.json" and is_test_mode:
            state_file = "state/scoring_state_test.json"

        self.state_file = Path(state_file)
        # Create the state directory if it doesn't exist
        self.state_file.parent.mkdir(exist_ok=True, parents=True)

        # Check for old state file in root directory for backward compatibility
        old_state_file = Path(".scoring_state.json")
        if old_state_file.exists() and not self.state_file.exists():
            # Migrate state from old location to new location
            console.print(f"[yellow]Migrating scoring state from {old_state_file} to {self.state_file}[/yellow]")
            self.state_file.parent.mkdir(exist_ok=True, parents=True)
            old_state_file.rename(self.state_file)

        self.state = self._load_state()

    def _load_state(self):
        """Load saved scoring state or create empty state."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except json.JSONDecodeError:
                console.print(f"[yellow]Warning: Could not parse scoring state file. Creating new state.[/yellow]")
                return {"scored_submissions": {}}
        return {"scored_submissions": {}}

    def save_state(self):
        """Save current state to disk with collision avoidance for parallel processing."""
        # Re-read the current state file to get any changes made by other processes
        if self.state_file.exists():
            try:
                # Read the current state from disk
                disk_state = json.loads(self.state_file.read_text())

                # Merge the disk state with our in-memory state
                # For each username in our state
                for username, user_data in self.state["scored_submissions"].items():
                    # Ensure username exists in disk state
                    if username not in disk_state["scored_submissions"]:
                        disk_state["scored_submissions"][username] = {}

                    # For each submission in our state
                    for submission_id, submission_data in user_data.items():
                        # If submission doesn't exist in disk state, copy it
                        if submission_id not in disk_state["scored_submissions"][username]:
                            disk_state["scored_submissions"][username][submission_id] = submission_data
                        else:
                            # Submission exists, merge the models list
                            disk_models = disk_state["scored_submissions"][username][submission_id].get("models", [])
                            our_models = submission_data.get("models", [])

                            # Combine models from both states (use set to avoid duplicates)
                            combined_models = list(set(disk_models + our_models))

                            # Update the disk state with combined models
                            disk_state["scored_submissions"][username][submission_id]["models"] = combined_models
                            # Update timestamp to most recent
                            disk_state["scored_submissions"][username][submission_id]["timestamp"] = datetime.datetime.now().isoformat()

                # Use the merged state for writing
                self.state = disk_state

            except json.JSONDecodeError:
                # If the file is corrupted, we'll overwrite it with our state
                console.print(f"[yellow]Warning: Could not parse scoring state file for merging. Overwriting with current state.[/yellow]")

        # Write the merged state back to disk
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def is_submission_scored(self, username: str, submission_id: str, model_name: Optional[str] = None) -> bool:
        """Check if a submission has already been scored.

        Args:
            username: The username of the submission
            submission_id: The ID of the submission
            model_name: Optional model name to check if this specific model has scored the submission

        Returns:
            True if the submission has been scored (by the specified model if model_name is provided)
        """
        user_submissions = self.state["scored_submissions"].get(username, {})

        # If no submission entry exists, it's not scored
        if submission_id not in user_submissions:
            return False

        # If no specific model is requested, check if any model has scored it
        if model_name is None:
            return bool(user_submissions.get(submission_id, {}).get("models", []))

        # Check if the specific model has scored this submission
        return model_name in user_submissions.get(submission_id, {}).get("models", [])

    def get_scored_models(self, username: str, submission_id: str) -> List[str]:
        """Get the list of models that have scored a submission.

        Args:
            username: The username of the submission
            submission_id: The ID of the submission

        Returns:
            List of model names that have scored this submission
        """
        user_submissions = self.state["scored_submissions"].get(username, {})
        return user_submissions.get(submission_id, {}).get("models", [])

    def mark_submission_scored(self, username: str, submission_id: str, model_name: str):
        """Mark a submission as scored with a specific model.

        Args:
            username: Username of the submitter
            submission_id: The ID of the submission
            model_name: Name of the model used for scoring
        """
        # Initialize user entry if needed
        if username not in self.state["scored_submissions"]:
            self.state["scored_submissions"][username] = {}

        # Initialize submission entry if needed
        if submission_id not in self.state["scored_submissions"][username]:
            # Use extracted timestamp instead of current time
            self.state["scored_submissions"][username][submission_id] = {
                "timestamp": extract_submission_datetime(submission_id),
                "models": [],
            }

        # Add model to list if not already present
        models = self.state["scored_submissions"][username][submission_id].get("models", [])
        if model_name not in models:
            models.append(model_name)
            self.state["scored_submissions"][username][submission_id]["models"] = models
            # Update timestamp to use submission timestamp
            self.state["scored_submissions"][username][submission_id]["timestamp"] = extract_submission_datetime(submission_id)

        # Save the updated state
        self.save_state()

    def sync_with_repository(self, repo_id: str, api: HfApi):
        """Sync scoring state with HF repository to find already scored submissions."""
        try:
            # First try to download the scoring state file directly
            try:
                remote_state_path = "state/scoring_state.json"
                if "test" in self.state_file.name:
                    remote_state_path = "state/scoring_state_test.json"

                # Create a temporary file for downloading
                with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
                    temp_path = Path(temp_file.name)

                try:
                    # Try to download the state file
                    api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=remote_state_path, local_dir=temp_path.parent)

                    downloaded_path = temp_path.parent / remote_state_path
                    if downloaded_path.exists():
                        # Read the remote state
                        try:
                            remote_state = json.loads(downloaded_path.read_text())
                            console.print(f"[green]Downloaded scoring state from repository[/green]")

                            # Merge with our local state
                            if "scored_submissions" in remote_state:
                                if "scored_submissions" not in self.state:
                                    self.state["scored_submissions"] = {}

                                # Merge each user's submissions
                                for username, user_data in remote_state["scored_submissions"].items():
                                    if username not in self.state["scored_submissions"]:
                                        self.state["scored_submissions"][username] = {}

                                    # Merge each submission
                                    for submission_id, submission_data in user_data.items():
                                        if submission_id not in self.state["scored_submissions"][username]:
                                            self.state["scored_submissions"][username][submission_id] = submission_data
                                        else:
                                            # Merge models list
                                            local_models = self.state["scored_submissions"][username][submission_id].get("models", [])
                                            remote_models = submission_data.get("models", [])

                                            # Combine models from both states
                                            self.state["scored_submissions"][username][submission_id]["models"] = list(
                                                set(local_models + remote_models)
                                            )

                            # Save the merged state
                            self.save_state()
                        except json.JSONDecodeError:
                            console.print(
                                f"[yellow]Could not parse downloaded scoring state file. Will rebuild from repository files.[/yellow]"
                            )
                except Exception as e:
                    console.print(
                        f"[yellow]Could not download scoring state file ({remote_state_path}): {e}. Will rebuild from repository files.[/yellow]"
                    )
            finally:
                # Clean up the temp file if it exists
                if "temp_path" in locals() and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass

            # Check for scored submissions in repository (as before)
            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

            # Track how many new entries we found
            new_entries = 0

            # Look for submission summary files which indicate completed scoring
            for file_path in files:
                if "submission_summary.csv" in file_path:
                    # Extract username and submission_id from path
                    # Pattern: submissions/{username}/{submission_id}/submission_summary.csv
                    parts = file_path.split("/")
                    if len(parts) >= 4:
                        username = parts[1]
                        submission_id = parts[2]

                        # Also check for model CSV files to determine which models have scored this
                        model_files = [
                            f
                            for f in files
                            if f.startswith(f"submissions/{username}/{submission_id}/")
                            and f.endswith(".csv")
                            and not f.endswith("submission_summary.csv")
                        ]
                        model_names = [Path(f).stem for f in model_files]

                        # Mark as scored
                        if username not in self.state["scored_submissions"]:
                            self.state["scored_submissions"][username] = {}

                        if submission_id not in self.state["scored_submissions"][username]:
                            new_entries += 1
                            self.state["scored_submissions"][username][submission_id] = {
                                "timestamp": extract_submission_datetime(submission_id),
                                "models": model_names,
                            }
                        else:
                            # Update model list for existing entries
                            existing_models = self.state["scored_submissions"][username][submission_id].get("models", [])
                            updated_models = list(set(existing_models + model_names))
                            if len(updated_models) > len(existing_models):
                                new_entries += 1
                                self.state["scored_submissions"][username][submission_id]["models"] = updated_models
                                # Update timestamp to use submission timestamp
                                self.state["scored_submissions"][username][submission_id]["timestamp"] = extract_submission_datetime(
                                    submission_id
                                )

            self.save_state()
            if new_entries > 0:
                console.print(
                    f"[green]Synchronized scoring state with repository: found {new_entries} new or updated submission scores[/green]"
                )
            else:
                console.print(f"[green]Scoring state synchronized with repository (no new entries found)[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not sync scoring state with repository: {e}[/yellow]")

    def upload_to_repository(self, repo_id: str, api: HfApi):
        """Upload scoring state to HF repository."""
        try:
            # Determine remote path based on filename
            remote_path = "state/scoring_state.json"
            if "test" in self.state_file.name:
                remote_path = "state/scoring_state_test.json"

            # Ensure state directory exists in repository
            try:
                files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
                if "state" not in files:
                    # Create empty file in state directory to ensure it exists
                    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
                        temp_file.write("")
                        temp_path = temp_file.name

                    api.upload_file(path_or_fileobj=temp_path, path_in_repo="state/.gitkeep", repo_id=repo_id, repo_type="dataset")

                    # Clean up temp file
                    Path(temp_path).unlink()
            except Exception as e:
                console.print(f"[yellow]Warning: Could not create state directory in repository: {e}[/yellow]")

            # Upload the scoring state file
            api.upload_file(path_or_fileobj=str(self.state_file), path_in_repo=remote_path, repo_id=repo_id, repo_type="dataset")
            console.print(f"[green]Uploaded scoring state to repository: {remote_path}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Error uploading scoring state to repository: {e}[/red]")
            return False

    def _merge_dictionaries(self, disk_state: Dict[str, Any]) -> None:
        """Merge disk state with memory state."""
        # Merge scored_submissions
        if "scored_submissions" in disk_state:
            for username, user_data in disk_state["scored_submissions"].items():
                if username not in self.state["scored_submissions"]:
                    self.state["scored_submissions"][username] = {}

                for submission_id, submission_data in user_data.items():
                    if submission_id not in self.state["scored_submissions"][username]:
                        self.state["scored_submissions"][username][submission_id] = submission_data
                    else:
                        # Merge models lists
                        disk_models = disk_state["scored_submissions"][username][submission_id].get("models", [])
                        local_models = self.state["scored_submissions"][username][submission_id].get("models", [])
                        combined_models = list(set(disk_models).union(set(local_models)))

                        # Update models list
                        self.state["scored_submissions"][username][submission_id]["models"] = combined_models

                        # Update timestamp to use the submission date instead of current time
                        self.state["scored_submissions"][username][submission_id]["timestamp"] = extract_submission_datetime(submission_id)


def download_new_submissions(
    dataset_id: str = "cluster-of-stars/TinyStoriesHackathon_Submissions",
    output_dir: Union[str, Path] = "downloaded_submissions",
    is_test_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Download all new submissions from HF dataset.

    Args:
        dataset_id: Hugging Face dataset ID
        output_dir: Directory to save downloaded submissions
        is_test_mode: Whether to download test submissions

    Returns:
        List of dictionaries with information about downloaded files
    """
    # Get HF API
    _, api = get_hf_user()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create tracking file for processed submissions
    processed_filename = "processed_test.json" if is_test_mode else "processed.json"
    processed_file = output_dir / processed_filename
    if processed_file.exists():
        processed = set(json.loads(processed_file.read_text()))
    else:
        processed = set()

    # Get all files in the dataset that match our pattern
    files = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")
    submission_files = [f for f in files if f.startswith("submissions/") and f.endswith(".csv")]
    metadata_files = [f for f in files if f.startswith("submissions/") and f.endswith("metadata.json")]

    # Download new submissions
    new_files = []

    # Process all metadata files first
    for remote_path in metadata_files:
        # Extract username
        parts = remote_path.split("/")
        if len(parts) != 3 or parts[2] != "metadata.json":
            continue

        username = parts[1]

        # Create user directory
        user_dir = output_dir / username
        user_dir.mkdir(exist_ok=True)

        # Download metadata file
        local_path = user_dir / "metadata.json"
        console.print(f"Downloading metadata [yellow]{remote_path}[/yellow] to [blue]{local_path}[/blue]")

        # Download to the correct path
        downloaded_path = api.hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=remote_path, local_dir=output_dir)

        # Move file to the correct location if needed
        if Path(downloaded_path) != local_path:
            Path(downloaded_path).rename(local_path)

    # Now process all submission files
    for remote_path in submission_files:
        if remote_path in processed:
            continue

        # Extract username and filename
        parts = remote_path.split("/")
        if len(parts) != 3:
            continue  # Skip unexpected formats

        username = parts[1]
        filename = parts[2]

        # Create user directory
        user_dir = output_dir / username
        user_dir.mkdir(exist_ok=True)

        # Download file
        local_path = user_dir / filename
        console.print(f"Downloading [yellow]{remote_path}[/yellow] to [blue]{local_path}[/blue]")

        # Download to the correct path
        downloaded_path = api.hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=remote_path, local_dir=output_dir)

        # Move file to the correct location if needed
        if Path(downloaded_path) != local_path:
            Path(downloaded_path).rename(local_path)

        # Add to processed list
        processed.add(remote_path)
        new_files.append(dict(username=username, filename=filename, remote_path=remote_path, local_path=local_path))

    # Update processed file
    processed_file.write_text(json.dumps(list(processed)))

    return new_files


def extract_submission_datetime(submission_id: str) -> str:
    """Extract timestamp from submission ID using regex pattern.

    Args:
        submission_id: The submission ID string

    Returns:
        Formatted timestamp string from the submission ID (YYYY-MM-DD HH:MM),
        or current time if no timestamp pattern found
    """
    # Use the same regex pattern from submission.py
    timestamp_match = re.search(r"(\d{8}_\d{6})", submission_id)
    if timestamp_match:
        try:
            ts_str = timestamp_match.group(1)
            # Parse timestamp in format YYYYmmdd_HHMMSS
            ts_datetime = datetime.datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            # Format as YYYY-MM-DD HH:MM
            return ts_datetime.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            # If parsing fails, return current time in simplified format
            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    else:
        # If no timestamp found, return current time in simplified format
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def get_authenticated_api() -> HfApi:
    """Get authenticated HF API instance."""
    _, api = get_hf_user()
    return api


def extract_scores(response: str) -> Tuple[Dict[str, float], bool]:
    """Extract scores from the model's response.

    Uses regex patterns to find scores for each category in the response.
    Stores them as lowercase keys (e.g., 'grammar', 'creativity').
    COHERENCE is optional and will be included if present in the response.

    Args:
        response: Text response from the model

    Returns:
        Tuple of (scores dict, success boolean)
    """
    scores = {}

    # Pre-process response to remove formatting characters
    # Keep only alphanumeric, whitespace, colons, slashes, and periods (for decimals)
    cleaned_response = re.sub(r"[^a-zA-Z0-9\s:/.()]", "", response)

    # Extract individual category scores with a more flexible regex
    for category in ScoreCategory:
        # Pattern for standard scores (7 or 7.5)
        standard_pattern = f"{category.value}\\s*[:=>]\\s*(\\d+(?:\\.\\d+)?)"
        # Pattern for scores in X/10 format
        ratio_pattern = f"{category.value}\\s*[:=>]\\s*(\\d+(?:\\.\\d+)?)\\s*\\/\\s*10"

        # First check for X/10 format
        ratio_matches = re.findall(ratio_pattern, cleaned_response, re.IGNORECASE)
        if ratio_matches:
            # Use the last match if multiple exist
            scores[category.lower()] = float(ratio_matches[-1])
        else:
            # Then check for standard format
            standard_matches = re.findall(standard_pattern, cleaned_response, re.IGNORECASE)
            if standard_matches:
                # Use the last match if multiple exist
                scores[category.lower()] = float(standard_matches[-1])

    # Define required categories (all except COHERENCE)
    required_categories = [cat for cat in ScoreCategory if cat != ScoreCategory.COHERENCE]

    # Check if all required categories have scores
    success = all(category.lower() in scores for category in required_categories)
    return scores, success


def write_csv(path: Path, scores: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """Write scores to a CSV file using pandas.

    Args:
        path: Path to output CSV file
        scores: Dictionary containing score data
    """
    # Convert nested dictionary to a list of records for DataFrame
    records = []
    for username in scores:
        for submission_id in scores[username]:
            for model_arch in scores[username][submission_id]:
                model_data = scores[username][submission_id][model_arch]
                score = model_data["score"]

                # Add a record for each item detail
                if "details" in model_data:
                    for item in model_data["details"]:
                        item_id = item["item_id"]
                        # Create a record with all the basic info
                        record = {
                            "username": username,
                            "submission_id": submission_id,
                            "model_arch": model_arch,
                            "score": score,
                            "item_id": item_id,
                        }
                        # Add individual scores
                        for category, value in item["scores"].items():
                            record[category] = value
                        records.append(record)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    console.print(f"[green]Scores written to {path}[/green]")


def read_csv(path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Read scores from a CSV file using pandas.

    Args:
        path: Path to input CSV file

    Returns:
        Dictionary containing score data
    """
    # Read CSV into DataFrame
    df = pd.read_csv(path)

    # Initialize scores dictionary
    scores = {}

    # Group by username, submission_id, and model_arch
    for (username, submission_id, model_arch), group in df.groupby(["username", "submission_id", "model_arch"]):
        # Initialize nested dictionaries if they don't exist
        if username not in scores:
            scores[username] = {}
        if submission_id not in scores[username]:
            scores[username][submission_id] = {}

        # Get the submission score (same for all rows in this group)
        submission_score = group["score"].iloc[0]

        # Initialize model arch data
        scores[username][submission_id][model_arch] = {"score": submission_score, "details": []}

        # Add details for each item
        for _, row in group.iterrows():
            # Extract score categories and values
            score_categories = [cat.lower() for cat in ScoreCategory]
            item_scores = {}
            for category in score_categories:
                if category in row and not pd.isna(row[category]):
                    item_scores[category] = float(row[category])

            # Add item details
            detail = {"item_id": int(row["item_id"]), "scores": item_scores}
            scores[username][submission_id][model_arch]["details"].append(detail)

    return scores


def process_scores(
    responses: Dict[int, str],
    followup_responses: Dict[int, str],
    username: str,
    submission_id: str,
    model_arch: str,
    scores: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[int, Dict[str, float]], float, int]:
    """Process model responses and extract scores.

    Extracts scores from model responses and updates the scores dictionary.
    The scores are stored with lowercase category names from the ScoreCategory enum.

    Args:
        responses: Dictionary mapping item IDs to model responses
        followup_responses: Dictionary mapping item IDs to follow-up responses
        username: Username of the submission
        submission_id: ID of the submission
        model_arch: Model architecture name
        scores: Dictionary to update with extracted scores

    Returns:
        Tuple of (updated scores dict, item_scores dict, total_score, processed_count)
    """
    item_scores = {}
    total_score = 0
    processed_count = 0

    # Initialize details list if needed
    if "details" not in scores[username][submission_id][model_arch]:
        scores[username][submission_id][model_arch]["details"] = []

    # Process each response
    for idx, response in responses.items():
        # Get follow-up response if it exists
        followup_response = followup_responses.get(idx)

        # Try to extract scores from initial response
        extracted_scores, success = extract_scores(response)

        # If unsuccessful and a follow-up response exists, try that one
        if not success and followup_response:
            extracted_scores, success = extract_scores(followup_response)

        # If we have scores, normalize them and store
        if success or extracted_scores:
            # Normalize scores to be between 1 and 10
            for key, value in list(extracted_scores.items()):
                extracted_scores[key] = min(10, max(1, value))

            # Store scores
            item_scores[idx] = extracted_scores

            # Update total score (using the ScoreCategory.OVERALL category)
            overall_category = ScoreCategory.OVERALL.lower()
            total_score += extracted_scores.get(overall_category, 0)
            processed_count += 1

            # Add to details
            details_item = {"item_id": idx, "scores": extracted_scores}
            scores[username][submission_id][model_arch]["details"].append(details_item)

    # Update average score if we processed any items
    if processed_count > 0:
        avg_score = total_score / processed_count
        scores[username][submission_id][model_arch]["score"] = avg_score

    return scores, item_scores, total_score, processed_count


def generate_model_dataframe(
    username: str,
    submission_id: str,
    model_name: str,
    prompts: List[str],
    completions: List[str],
    item_scores: Dict[int, Dict[str, float]],
) -> pd.DataFrame:
    """Generate a per-model DataFrame for a submission.

    Args:
        username: Username of the submission
        submission_id: ID of the submission
        model_name: Name of the model
        prompts: List of prompts
        completions: List of completions
        item_scores: Dictionary mapping item IDs to score dictionaries

    Returns:
        DataFrame containing the scores
    """
    data = []
    for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
        if idx in item_scores:
            row = {
                "item_id": idx,
                "prompt": prompt,
                "completion": completion,
            }
            # Add all scores
            row.update(item_scores[idx])
            data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)
    return df


def save_model_csv(
    username: str,
    submission_id: str,
    model_name: str,
    prompts: List[str],
    completions: List[str],
    item_scores: Dict[int, Dict[str, float]],
    base_dir: str = "submissions",
    overwrite: bool = False,
) -> Path:
    """Save per-model CSV for a submission.

    Args:
        username: Username of the submission
        submission_id: ID of the submission
        model_name: Name of the model
        prompts: List of prompts
        completions: List of completions
        item_scores: Dictionary mapping item IDs to score dictionaries
        base_dir: Base directory for submissions
        overwrite: Whether to overwrite existing CSV files

    Returns:
        Path to the saved CSV file
    """
    # Create the directory structure
    submission_dir = Path(base_dir) / username / submission_id
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Define the CSV file path
    csv_path = submission_dir / f"{model_name}.csv"

    # Check if the CSV already exists and we're not overwriting
    if csv_path.exists() and not overwrite:
        console.print(f"[yellow]Per-model CSV already exists: {csv_path}[/yellow]")
        return csv_path

    # Generate the DataFrame
    df = generate_model_dataframe(
        username=username,
        submission_id=submission_id,
        model_name=model_name,
        prompts=prompts,
        completions=completions,
        item_scores=item_scores,
    )

    # Save the DataFrame to CSV
    df.to_csv(csv_path, index=False)
    action = "Updated" if csv_path.exists() and overwrite else "Saved"
    console.print(f"[green]{action} per-model CSV to: {csv_path}[/green]")

    return csv_path


def find_model_csvs(username: str, submission_id: str, base_dir: str = "submissions") -> List[Path]:
    """Find all model CSV files for a submission.

    Args:
        username: Username of the submission
        submission_id: ID of the submission
        base_dir: Base directory for submissions

    Returns:
        List of model CSV paths
    """
    submission_dir = Path(base_dir) / username / submission_id
    if not submission_dir.exists():
        return []

    # Find all CSV files except summary files
    model_csvs = [file for file in submission_dir.glob("*.csv") if file.stem != "submission_summary"]
    return model_csvs


def calculate_submission_average(username: str, submission_id: str, base_dir: str = "submissions") -> Optional[pd.DataFrame]:
    """Calculate average scores across all models for a submission.

    Args:
        username: Username of the submission
        submission_id: ID of the submission
        base_dir: Base directory for submissions

    Returns:
        DataFrame with average scores or None if no data found
    """
    # Find all model CSV files
    model_csvs = find_model_csvs(username, submission_id, base_dir)
    if not model_csvs:
        console.print(f"[yellow]No model CSV files found for {username}/{submission_id}[/yellow]")
        return None

    # Prepare to collect per-model data
    all_model_data = []
    overall_categories = [cat.lower() for cat in ScoreCategory]

    # Process each model file
    for model_csv in model_csvs:
        try:
            model_df = pd.read_csv(model_csv)
            model_name = model_csv.stem  # Use filename as model name

            # Calculate average for each category for this model
            model_avgs = {}
            for category in overall_categories:
                if category in model_df.columns:
                    model_avgs[category] = model_df[category].mean()

            # Create a row for this model
            model_row = pd.DataFrame([model_avgs])
            model_row["model_name"] = model_name
            model_row["item_id"] = -1  # Use -1 to indicate this is a summary
            model_row["username"] = username
            model_row["submission_id"] = submission_id

            all_model_data.append(model_row)
        except Exception as e:
            console.print(f"[yellow]Error processing {model_csv}: {e}[/yellow]")

    if not all_model_data:
        console.print(f"[yellow]No valid model data found for {username}/{submission_id}[/yellow]")
        return None

    # Combine all model rows into a single DataFrame
    summary_df = pd.concat(all_model_data, ignore_index=True)

    # Count the number of models
    num_models = len(all_model_data)

    # Create a row for the overall average across all models
    overall_avg = {}
    for category in overall_categories:
        if category in summary_df.columns:
            overall_avg[category] = summary_df[category].mean()

    overall_row = pd.DataFrame([overall_avg])
    overall_row["model_name"] = "average"
    overall_row["item_id"] = -1  # Use -1 to indicate this is a summary
    overall_row["username"] = username
    overall_row["submission_id"] = submission_id
    overall_row["num_models"] = num_models  # Add the number of models to the average row

    # Add the overall average row at the top
    summary_df = pd.concat([overall_row, summary_df], ignore_index=True)

    # Add num_models column to all model rows
    summary_df["num_models"] = summary_df.apply(lambda x: num_models if x["model_name"] == "average" else 1, axis=1)

    return summary_df


def save_submission_summary(
    username: str, submission_id: str, base_dir: str = "submissions", overwrite: bool = False, repo_id: str = None
) -> Optional[Path]:
    """Generate and save a submission summary CSV.

    This creates a summary of scores for each model for a single submission.
    The summary includes one row per model with all categories, plus an "average" row.

    Args:
        username: Username of the submission
        submission_id: ID of the submission
        base_dir: Base directory for submissions
        overwrite: Whether to overwrite existing summary file
        repo_id: Optional repository ID to download existing summary from

    Returns:
        Path to the summary CSV if successful, None otherwise
    """
    # Calculate submission average
    summary_df = calculate_submission_average(username, submission_id, base_dir)

    if summary_df is None:
        return None

    # Create the directory structure
    submission_dir = Path(base_dir) / username / submission_id
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Define the summary CSV path
    summary_path = submission_dir / "submission_summary.csv"

    # Only handle existing summary if not overwriting
    if summary_path.exists() and not overwrite:
        console.print(f"[yellow]Summary file already exists and overwrite=False: {summary_path}[/yellow]")
        return summary_path

    # Save the summary DataFrame
    summary_df.to_csv(summary_path, index=False)
    action = "Updated" if summary_path.exists() else "Saved"
    console.print(f"[green]{action} submission summary to: {summary_path}[/green]")

    return summary_path


def update_user_score_history(
    username: str,
    submission_id: str,
    summary_df: pd.DataFrame,
    base_dir: str = "submissions",
    overwrite: bool = True,  # Default to overwrite for history
) -> Path:
    """Update the user's score history with this submission.

    This maintains a record of all submissions by a user with their overall average scores.
    Model-specific scores are NOT included in the history.

    Args:
        username: Username of the submission
        submission_id: ID of the submission
        summary_df: DataFrame with submission summary
        base_dir: Base directory for submissions
        overwrite: Whether to overwrite existing entries for this submission_id

    Returns:
        Path to the updated history CSV
    """
    # Create the user directory if it doesn't exist
    user_dir = Path(base_dir) / username
    user_dir.mkdir(parents=True, exist_ok=True)

    history_path = user_dir / "score_history.csv"

    # Use only the "average" row for the history
    if "model_name" in summary_df.columns:
        history_entry = summary_df[summary_df["model_name"] == "average"].copy()
        # Remove model_name column from the history
        if not history_entry.empty:
            history_entry = history_entry.drop(columns=["model_name"])
    else:
        # For backwards compatibility
        history_entry = summary_df.copy()
        # Remove any model-specific columns
        cols_to_drop = [col for col in history_entry.columns if col.endswith("_overall")]
        if cols_to_drop:
            history_entry = history_entry.drop(columns=cols_to_drop)

    # Add timestamp using submission_id instead of current time
    history_entry["timestamp"] = extract_submission_datetime(submission_id)

    # Check if history file exists
    if history_path.exists():
        # Load existing history
        history_df = pd.read_csv(history_path)

        # Check if this submission is already in the history
        if submission_id in history_df["submission_id"].values and overwrite:
            # Update the existing entry
            history_df = history_df[history_df["submission_id"] != submission_id]
        elif submission_id in history_df["submission_id"].values and not overwrite:
            console.print(f"[yellow]Submission {submission_id} already in history and overwrite=False[/yellow]")
            return history_path

        # Append the new entry
        history_df = pd.concat([history_df, history_entry], ignore_index=True)
    else:
        # Create new history
        history_df = history_entry

    # Save the updated history
    history_df.to_csv(history_path, index=False)
    console.print(f"[green]User score history updated: {history_path}[/green]")

    return history_path


def read_user_metadata(username: str, base_dir: str = "submissions") -> str:
    """Read user metadata from metadata.json file.

    Args:
        username: Username of the user
        base_dir: Base directory for submissions

    Returns:
        Weight class as a string: 'small', 'medium', or 'large'
    """
    # First check in downloaded_submissions directory since we know the files are there
    downloads_dir = Path("downloaded_submissions")
    downloads_metadata_path = downloads_dir / username / "metadata.json"

    if downloads_metadata_path.exists():
        metadata_path = downloads_metadata_path
        console.print(f"[blue]Found metadata.json in downloads directory for {username}[/blue]")
    else:
        # If not in downloads, check in the submissions directory
        user_dir = Path(base_dir) / username
        metadata_path = user_dir / "metadata.json"

        # If not found in scores directory, check test submissions directory if needed
        if not metadata_path.exists():
            # Detect if we're using test or production mode based on base_dir
            is_test_mode = "test" in base_dir.lower()

            # Determine the correct downloaded submissions directory for test mode
            if is_test_mode:
                test_downloads_dir = Path("downloaded_submissions_test")
                test_downloads_metadata_path = test_downloads_dir / username / "metadata.json"

                if test_downloads_metadata_path.exists():
                    metadata_path = test_downloads_metadata_path
                    console.print(f"[blue]Found metadata.json in test downloads directory for {username}[/blue]")

    # Default to small if metadata doesn't exist
    if not metadata_path.exists():
        console.print(f"[yellow]No metadata.json found for {username}, defaulting to 'small' weight class[/yellow]")
        return "small"

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Get weight class or default to small
        weight_class = metadata.get("weight_class", "small")

        # Validate weight class
        if weight_class not in ["small", "medium", "large"]:
            console.print(f"[yellow]Invalid weight class: {weight_class} for user {username}, defaulting to 'small'[/yellow]")
            return "small"

        console.print(f"[green]Using weight class '{weight_class}' for user {username}[/green]")
        return weight_class
    except Exception as e:
        console.print(f"[red]Error reading metadata for {username}: {e}[/red]")
        return "small"


def find_all_user_histories(base_dir: str = "submissions") -> List[Path]:
    """Find all user score history files.

    Args:
        base_dir: Base directory for submissions

    Returns:
        List of paths to score_history.csv files
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    # Find all score_history.csv files
    history_files = list(base_path.glob("*/score_history.csv"))
    return history_files


def generate_global_leaderboard(base_dir: str = "submissions") -> pd.DataFrame:
    """Generate a global leaderboard across all users.

    Args:
        base_dir: Base directory for submissions

    Returns:
        DataFrame containing the global leaderboard
    """
    # Find all user history files
    history_files = find_all_user_histories(base_dir)

    if not history_files:
        console.print(f"[yellow]No user history files found in {base_dir}[/yellow]")
        return pd.DataFrame()

    # Load and combine all histories
    user_data = []

    for history_path in history_files:
        try:
            # Get username from directory path
            username = history_path.parent.name

            # Read history file
            history = pd.read_csv(history_path)

            if history.empty:
                continue

            # For each user, find their best submission based on ScoreCategory.OVERALL score
            overall_score_category = ScoreCategory.OVERALL.lower()
            if overall_score_category in history.columns:
                best_submission = history.loc[history[overall_score_category].idxmax()]
            else:
                # Fall back to first submission if no overall score
                best_submission = history.iloc[0]

            # Get individual category scores
            category_scores = {}
            for category in ScoreCategory:
                category_lower = category.lower()
                if category_lower in best_submission:
                    category_scores[category_lower] = best_submission[category_lower]
                else:
                    category_scores[category_lower] = 0.0

            # Get number of models used for scoring
            num_models = best_submission.get("num_models", 0)

            # Calculate total number of submissions for this user
            submission_count = len(history)

            # Get submission_id
            submission_id = best_submission["submission_id"] if "submission_id" in best_submission else ""

            # Get submission timestamp - use the extracted timestamp from submission_id when possible
            if "timestamp" in best_submission and best_submission["timestamp"]:
                timestamp = best_submission["timestamp"]
            else:
                # Extract timestamp from submission_id if possible
                timestamp = extract_submission_datetime(submission_id)

            # Get weight class
            weight_class = read_user_metadata(username, base_dir)

            # Create user entry
            user_entry = {
                "username": username,
                "submission_id": submission_id,
                "submission_count": submission_count,
                "submission_date": timestamp,
                "weight_class": weight_class,
                "num_models": num_models,  # Add number of models to the leaderboard
                **category_scores,  # Add all category scores individually
            }

            user_data.append(user_entry)

        except Exception as e:
            console.print(f"[red]Error processing history for {history_path.parent.name}: {e}[/red]")
            continue

    if not user_data:
        return pd.DataFrame()

    # Create leaderboard DataFrame
    leaderboard = pd.DataFrame(user_data)

    # Sort by overall score descending
    leaderboard = leaderboard.sort_values(by=ScoreCategory.OVERALL.lower(), ascending=False).reset_index(drop=True)

    # Add rank column
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))

    return leaderboard


def save_global_leaderboard(leaderboard: pd.DataFrame, base_dir: str = "leaderboards") -> Dict[str, Path]:
    """Save the global leaderboard to CSV and Markdown.

    Args:
        leaderboard: DataFrame containing the global leaderboard
        base_dir: Base directory to save the leaderboard files

    Returns:
        Dictionary with paths to the saved files
    """
    base_path = Path(base_dir)

    if leaderboard.empty:
        console.print("[yellow]Empty leaderboard, not saving[/yellow]")
        return {}

    saved_files = {}

    # Save CSV
    csv_path = base_path / "leaderboard_global.csv"
    leaderboard.to_csv(csv_path, index=False)
    saved_files["csv"] = csv_path
    console.print(f"[green]Global leaderboard saved to {csv_path}[/green]")

    return saved_files


def generate_weight_class_leaderboards(leaderboard: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Generate separate leaderboards for each weight class.

    Args:
        leaderboard: The global leaderboard DataFrame

    Returns:
        Dictionary mapping weight class to its leaderboard DataFrame
    """
    if leaderboard.empty:
        return {}

    # Ensure weight_class column exists
    if "weight_class" not in leaderboard.columns:
        console.print("[yellow]Leaderboard missing weight_class column, cannot generate weight class leaderboards[/yellow]")
        return {}

    # Group by weight class
    weight_class_leaderboards = {}

    for weight_class in ["small", "medium", "large"]:
        # Filter the leaderboard by weight class
        class_leaderboard = leaderboard[leaderboard["weight_class"] == weight_class].copy()

        if class_leaderboard.empty:
            console.print(f"[yellow]No entries for weight class '{weight_class}'[/yellow]")
            continue

        # Rerank within this weight class
        class_leaderboard = class_leaderboard.sort_values(by=ScoreCategory.OVERALL.lower(), ascending=False).reset_index(drop=True)
        class_leaderboard["rank"] = range(1, len(class_leaderboard) + 1)

        weight_class_leaderboards[weight_class] = class_leaderboard

    return weight_class_leaderboards


def save_weight_class_leaderboards(weight_class_leaderboards: Dict[str, pd.DataFrame], base_dir: str = "leaderboards") -> Dict[str, Path]:
    """Save weight class leaderboards to CSV.

    Args:
        weight_class_leaderboards: Dictionary of weight class leaderboards
        base_dir: Base directory to save the leaderboard files

    Returns:
        Dictionary with paths to the saved files
    """
    base_path = Path(base_dir)
    saved_files = {}

    if not weight_class_leaderboards:
        console.print("[yellow]No weight class leaderboards to save[/yellow]")
        return saved_files

    # Save individual weight class CSV files
    for weight_class, leaderboard in weight_class_leaderboards.items():
        if leaderboard.empty:
            continue

        # Save CSV
        csv_path = base_path / f"leaderboard_{weight_class}.csv"
        leaderboard.to_csv(csv_path, index=False)
        saved_files[f"{weight_class}_csv"] = csv_path
        console.print(f"[green]{weight_class.capitalize()} weight class leaderboard saved to {csv_path}[/green]")

    return saved_files


def generate_and_save_all_leaderboards(base_dir: str = "submissions", output_dir: str = "leaderboards") -> pd.DataFrame:
    """Generate and save all leaderboards in one function.

    This is a convenience function that combines all leaderboard generation
    and saving into a single call.

    Args:
        base_dir: Base directory for submissions
        output_dir: Directory to save leaderboard files

    Returns:
        Dictionary with paths to all saved files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate global leaderboard
    global_leaderboard = generate_global_leaderboard(base_dir)

    if global_leaderboard.empty:
        console.print("[yellow]No data for leaderboards[/yellow]")
        return {}

    # Save global leaderboard
    saved_files = save_global_leaderboard(global_leaderboard, output_dir)

    # Generate weight class leaderboards
    weight_class_leaderboards = generate_weight_class_leaderboards(global_leaderboard)

    # Save weight class leaderboards
    weight_class_files = save_weight_class_leaderboards(weight_class_leaderboards, output_dir)

    # Combine all saved files
    saved_files.update(weight_class_files)

    return global_leaderboard


def upload_user_files(
    username: str,
    submission_id: Optional[str] = None,
    base_dir: str = "submissions",
    repo_id: str = "cluster-of-stars/TinyStoriesHackathon_Scores",
) -> Dict[str, int]:
    """Upload files for a specific user and optionally a specific submission.

    Args:
        username: Username to upload files for
        submission_id: Optional specific submission ID to upload
        base_dir: Base directory for submissions
        repo_id: Hugging Face repository ID

    Returns:
        Dictionary with counts of uploaded files
    """
    user_dir = Path(base_dir) / username
    if not user_dir.exists():
        console.print(f"[yellow]User directory not found: {user_dir}[/yellow]")
        return {"uploaded": 0, "skipped": 0, "error": 0}

    # Get authenticated API
    try:
        api = get_authenticated_api()
    except Exception as e:
        console.print(f"[red]Authentication error: {e}[/red]")
        return {"uploaded": 0, "skipped": 0, "error": 1}

    # Create upload tracker
    is_test_mode = "Test" in repo_id
    tracker = UploadTracker(is_test_mode=is_test_mode)
    results = {"uploaded": 0, "skipped": 0, "error": 0}

    # Upload user metadata if it exists
    metadata_path = user_dir / "metadata.json"
    if metadata_path.exists() and tracker.should_upload_file(metadata_path):
        try:
            remote_path = f"submissions/{username}/metadata.json"
            api.upload_file(path_or_fileobj=str(metadata_path), path_in_repo=remote_path, repo_id=repo_id, repo_type="dataset")
            tracker.mark_file_uploaded(metadata_path)
            results["uploaded"] += 1
            console.print(f"[green]Uploaded {metadata_path} to {remote_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error uploading {metadata_path}: {e}[/red]")
            results["error"] += 1
    elif metadata_path.exists():
        results["skipped"] += 1

    # Upload score history if it exists
    history_path = user_dir / "score_history.csv"
    if history_path.exists() and tracker.should_upload_file(history_path):
        try:
            remote_path = f"submissions/{username}/score_history.csv"
            api.upload_file(path_or_fileobj=str(history_path), path_in_repo=remote_path, repo_id=repo_id, repo_type="dataset")
            tracker.mark_file_uploaded(history_path)
            results["uploaded"] += 1
            console.print(f"[green]Uploaded {history_path} to {remote_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error uploading {history_path}: {e}[/red]")
            results["error"] += 1
    elif history_path.exists():
        results["skipped"] += 1

    # Handle submission files
    if submission_id:
        # Upload specific submission
        submission_dir = user_dir / submission_id
        if submission_dir.exists():
            results = _upload_submission_folder(api, submission_dir, username, submission_id, tracker, repo_id, results)
    else:
        # Upload all submissions
        for submission_dir in user_dir.glob("*/"):
            if submission_dir.is_dir() and submission_dir.name != "__pycache__":
                submission_id = submission_dir.name
                results = _upload_submission_folder(api, submission_dir, username, submission_id, tracker, repo_id, results)

    return results


def _upload_submission_folder(
    api: HfApi, submission_dir: Path, username: str, submission_id: str, tracker: UploadTracker, repo_id: str, results: Dict[str, int]
) -> Dict[str, int]:
    """Helper function to upload a submission folder at once using upload_folder.

    Args:
        api: Authenticated Hugging Face API client
        submission_dir: Directory containing submission files
        username: Username of the submission owner
        submission_id: ID of the submission
        tracker: Upload tracker to avoid redundant uploads
        repo_id: Hugging Face repository ID
        results: Current upload metrics to update

    Returns:
        Updated upload metrics dictionary
    """
    # Count files to be uploaded for metrics
    csv_files = list(submission_dir.glob("*.csv"))

    # Check if any files need uploading using tracker
    files_to_upload = [f for f in csv_files if tracker.should_upload_file(f)]

    if not files_to_upload:
        # Update skipped count
        results["skipped"] += len(csv_files)
        return results

    try:
        # Upload the entire submission folder at once
        remote_path = f"submissions/{username}/{submission_id}"
        response = api.upload_folder(folder_path=str(submission_dir), path_in_repo=remote_path, repo_id=repo_id, repo_type="dataset")

        # Mark all files as uploaded
        for file_path in csv_files:
            tracker.mark_file_uploaded(file_path)

        # Count newly uploaded files for metrics
        results["uploaded"] += len(files_to_upload)
        console.print(f"[green]Uploaded submission folder {submission_dir} to {remote_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error uploading submission folder {submission_dir}: {e}[/red]")
        results["error"] += 1

        # Try individual files as fallback
        console.print(f"[yellow]Attempting individual file uploads as fallback...[/yellow]")
        for file_path in files_to_upload:
            try:
                remote_file_path = f"submissions/{username}/{submission_id}/{file_path.name}"
                api.upload_file(path_or_fileobj=str(file_path), path_in_repo=remote_file_path, repo_id=repo_id, repo_type="dataset")
                tracker.mark_file_uploaded(file_path)
                results["uploaded"] += 1
                console.print(f"[green]Uploaded {file_path} to {remote_file_path}[/green]")
            except Exception as e2:
                console.print(f"[red]Error uploading {file_path}: {e2}[/red]")
                results["error"] += 1

    return results


def upload_all_user_files(base_dir: str = "submissions", repo_id: str = "cluster-of-stars/TinyStoriesHackathon_Scores") -> Dict[str, int]:
    """Upload files for all users in the base directory.

    Args:
        base_dir: Base directory for submissions
        repo_id: Hugging Face repository ID

    Returns:
        Dictionary with counts of uploaded files
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        console.print(f"[yellow]Base directory not found: {base_path}[/yellow]")
        return {"uploaded": 0, "skipped": 0, "error": 0}

    total_results = {"uploaded": 0, "skipped": 0, "error": 0}

    # Process each user directory
    for user_dir in base_path.glob("*/"):
        if user_dir.is_dir() and user_dir.name != "__pycache__":
            username = user_dir.name
            console.print(f"[blue]Processing user: {username}[/blue]")

            # Upload user files
            results = upload_user_files(username, base_dir=base_dir, repo_id=repo_id)

            # Aggregate results
            for key in total_results:
                total_results[key] += results.get(key, 0)

    console.print(
        f"[blue]Upload summary: {total_results['uploaded']} files uploaded, "
        f"{total_results['skipped']} files skipped, "
        f"{total_results['error']} errors[/blue]"
    )

    return total_results


def upload_leaderboards(
    leaderboard_dir: str = "leaderboards", repo_id: str = "cluster-of-stars/TinyStoriesHackathon_Scores"
) -> Dict[str, int]:
    """Upload leaderboard files to the repository.

    Args:
        leaderboard_dir: Directory containing leaderboard files
        repo_id: Hugging Face repository ID

    Returns:
        Dictionary with counts of uploaded files
    """
    leaderboard_path = Path(leaderboard_dir)
    if not leaderboard_path.exists():
        console.print(f"[yellow]Leaderboard directory not found: {leaderboard_path}[/yellow]")
        return {"uploaded": 0, "skipped": 0, "error": 0}

    # Get authenticated API
    try:
        api = get_authenticated_api()
    except Exception as e:
        console.print(f"[red]Authentication error: {e}[/red]")
        return {"uploaded": 0, "skipped": 0, "error": 1}

    # Create upload tracker
    is_test_mode = "Test" in repo_id
    tracker = UploadTracker(is_test_mode=is_test_mode)
    results = {"uploaded": 0, "skipped": 0, "error": 0}

    # Find all leaderboard CSV files
    leaderboard_patterns = [
        "leaderboard_global.csv",
        "leaderboard_small.csv",
        "leaderboard_medium.csv",
        "leaderboard_large.csv",
    ]

    # Gather leaderboard files to check if they need updating
    leaderboard_files = []
    for pattern in leaderboard_patterns:
        found_files = list(leaderboard_path.glob(pattern))
        leaderboard_files.extend(found_files)

    # Check if there are any files to upload
    files_to_upload = [f for f in leaderboard_files if tracker.should_upload_file(f)]

    if not files_to_upload:
        results["skipped"] = len(leaderboard_files)
        console.print(f"[yellow]All {len(leaderboard_files)} leaderboard files already up to date[/yellow]")
        return results

    try:
        # Upload the entire leaderboard directory directly
        api.upload_folder(
            folder_path=str(leaderboard_path),
            path_in_repo="",  # Upload to root
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Mark all files as uploaded
        for file_path in files_to_upload:
            tracker.mark_file_uploaded(file_path)

        # Count uploaded files
        results["uploaded"] = len(files_to_upload)
        console.print(f"[green]Uploaded {len(files_to_upload)} leaderboard files[/green]")

    except Exception as e:
        console.print(f"[red]Error uploading leaderboard files using upload_folder: {e}[/red]")
        console.print(f"[yellow]Falling back to individual file uploads...[/yellow]")

        # Fall back to individual uploads
        for file_path in files_to_upload:
            if tracker.should_upload_file(file_path):
                try:
                    remote_path = file_path.name
                    api.upload_file(path_or_fileobj=str(file_path), path_in_repo=remote_path, repo_id=repo_id, repo_type="dataset")
                    tracker.mark_file_uploaded(file_path)
                    results["uploaded"] += 1
                    console.print(f"[green]Uploaded {file_path} to {remote_path}[/green]")
                except Exception as e:
                    console.print(f"[red]Error uploading {file_path}: {e}[/red]")
                    results["error"] += 1
            else:
                results["skipped"] += 1

    console.print(
        f"[blue]Leaderboard upload summary: {results['uploaded']} files uploaded, "
        f"{results['skipped']} files skipped, "
        f"{results['error']} errors[/blue]"
    )

    return results


def generate_readme_leaderboard_section(leaderboard_dir: str = "leaderboards") -> str:
    """Generate a markdown section for the README with leaderboard highlights.

    Args:
        leaderboard_dir: Directory containing leaderboard files

    Returns:
        Markdown content for README
    """
    # Look for global leaderboard CSV file
    global_csv_path = Path(leaderboard_dir) / "leaderboard_global.csv"
    if not global_csv_path.exists():
        console.print(f"[yellow]Global leaderboard CSV not found: {global_csv_path}[/yellow]")
        return "<!-- No leaderboard data available -->"

    # Read the global leaderboard CSV
    try:
        global_df = pd.read_csv(global_csv_path)
        if global_df.empty:
            return "<!-- No leaderboard data available -->"
    except Exception as e:
        console.print(f"[yellow]Error reading global leaderboard CSV: {e}[/yellow]")
        return "<!-- No leaderboard data available -->"

    # Generate timestamp
    # Use AOE (Anywhere on Earth, UTC-12) timezone for the leaderboard timestamp
    aoe_timezone = datetime.datetime.timezone(datetime.timedelta(hours=-12))
    timestamp = datetime.datetime.now(aoe_timezone).strftime("%Y-%m-%d %H:%M:%S AOE")

    # Define column order with overall first, then other categories
    score_categories = [ScoreCategory.OVERALL.lower()]
    for category in ScoreCategory:
        if category.lower() != ScoreCategory.OVERALL.lower():
            score_categories.append(category.lower())

    # Create a new DataFrame with columns in the desired order, including num_models
    display_cols = ["rank", "username"] + score_categories + ["num_models", "submission_count", "submission_date"]

    # Filter to only include columns that exist in the leaderboard
    display_cols = [col for col in display_cols if col in global_df.columns]
    global_display_df = global_df[display_cols].copy()

    # Format floating point columns
    for col in score_categories:
        if col in global_display_df.columns:
            global_display_df[col] = global_display_df[col].map(lambda x: f"{float(x):.2f}")

    # Generate global leaderboard markdown
    global_md = global_display_df.to_markdown(index=False, tablefmt="pipe")

    # Build markdown content
    md_content = [
        "<!-- LEADERBOARD_START -->",
        f"## Leaderboard (Updated: {timestamp})",
        "",
        "### Global Standings",
        "",
        global_md,
        "",
        "### Weight Class Standings",
        "",
    ]

    # Add weight class tables by reading directly from CSV files
    for weight_class in ["small", "medium", "large"]:
        csv_path = Path(leaderboard_dir) / f"leaderboard_{weight_class}.csv"
        if csv_path.exists():
            try:
                # Read the CSV file
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue

                # Create a new DataFrame with columns in the desired order
                display_cols = ["rank", "username"] + score_categories + ["num_models", "submission_count", "submission_date"]

                # Filter to only include columns that exist in the leaderboard
                display_cols = [col for col in display_cols if col in df.columns]
                display_df = df[display_cols].copy()

                # Format floating point columns
                for col in score_categories:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map(lambda x: f"{float(x):.2f}")

                # Generate the markdown table
                table_md = display_df.to_markdown(index=False, tablefmt="pipe")

                # Add weight class header and table
                md_content.extend(
                    [
                        f"#### {weight_class.capitalize()} Weight Class",
                        "",
                        table_md,
                        "",
                    ]
                )

                console.print(f"[green]Added {weight_class} weight class table to README[/green]")
            except Exception as e:
                console.print(f"[yellow]Error generating {weight_class} weight class table: {e}[/yellow]")
        else:
            console.print(f"[yellow]No leaderboard CSV found for {weight_class} weight class[/yellow]")

    md_content.extend(["<!-- LEADERBOARD_END -->"])

    return "\n".join(md_content)


def update_repository_readme(
    leaderboard_dir: str = "leaderboards", repo_id: str = "cluster-of-stars/TinyStoriesHackathon_Scores", force_update: bool = False
) -> bool:
    """Update the repository README with leaderboard information.

    Args:
        leaderboard_dir: Directory containing leaderboard files
        repo_id: Hugging Face repository ID
        force_update: If True, update README regardless of when it was last updated

    Returns:
        Boolean indicating success
    """
    # Get authenticated API
    try:
        api = get_authenticated_api()
    except Exception as e:
        console.print(f"[red]Authentication error: {e}[/red]")
        return False

    # Create tracker and check if update is needed (unless forced)
    is_test_mode = "Test" in repo_id
    tracker = UploadTracker(is_test_mode=is_test_mode)
    if not force_update and not tracker.should_update_readme():
        console.print(f"[yellow]README was updated recently. Skipping update.[/yellow]")
        return True

    # Create temporary directory for README work
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Try to download existing README
        readme_path = temp_dir_path / "README.md"
        try:
            api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="README.md", local_dir=temp_dir)
        except Exception as e:
            console.print(f"[yellow]Could not download existing README: {e}. Creating new README.[/yellow]")
            readme_path.write_text("# TinyStories Hackathon Submissions\n\n")

        # Read existing README
        readme_content = readme_path.read_text() if readme_path.exists() else ""

        # Generate new leaderboard section
        leaderboard_section = generate_readme_leaderboard_section(leaderboard_dir)

        # Replace or append leaderboard section
        leaderboard_start = "<!-- LEADERBOARD_START -->"
        leaderboard_end = "<!-- LEADERBOARD_END -->"

        if leaderboard_start in readme_content and leaderboard_end in readme_content:
            # Replace existing section
            start_pos = readme_content.find(leaderboard_start)
            end_pos = readme_content.find(leaderboard_end) + len(leaderboard_end)
            new_readme = readme_content[:start_pos] + leaderboard_section + readme_content[end_pos:]
        else:
            # Append to README
            new_readme = readme_content.rstrip() + "\n\n" + leaderboard_section

        # Add YAML metadata at the top if it doesn't exist already
        yaml_pattern = r"---\nlanguage:.*?\n---\n"
        if not re.search(yaml_pattern, new_readme, re.DOTALL):
            # Create YAML metadata block
            yaml_metadata = """---
language: en
---

"""
            # Prepend YAML metadata to README content
            new_readme = yaml_metadata + new_readme

        # Write updated README
        readme_path.write_text(new_readme)

        # Upload README back to repository
        try:
            api.upload_file(path_or_fileobj=str(readme_path), path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
            tracker.mark_readme_updated()
            console.print(f"[green]Updated README.md with leaderboard information and YAML metadata[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Error uploading README: {e}[/red]")
            return False


def display_leaderboard_preview(leaderboard_df: pd.DataFrame):
    """Show a terminal preview of the leaderboard using Rich Table."""
    # Create a table with proper styling
    table = Table(title="Leaderboard Preview")

    # Add columns with styles
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Username", style="magenta")
    table.add_column("Overall", justify="right", style="green")
    table.add_column("Grammar", justify="right")

    # Check if coherence exists in the data and add the column if it does
    has_coherence = "coherence" in leaderboard_df.columns
    if has_coherence:
        table.add_column("Coherence", justify="right", style="yellow")

    table.add_column("Creativity", justify="right")
    table.add_column("Consistency", justify="right")
    table.add_column("Plot", justify="right")
    table.add_column("Models", justify="right", style="blue")
    table.add_column("Weight", style="blue")

    # Add top rows (limited to avoid cluttering terminal)
    for _, row in leaderboard_df.head(10).iterrows():
        # Build a list of values for the row
        row_values = [
            str(row["rank"]),
            row["username"],
            f"{row['overall']:.2f}",
            f"{row['grammar']:.2f}",
        ]

        # Add coherence if it exists
        if has_coherence:
            row_values.append(f"{row['coherence']:.2f}")

        # Add the rest of the values
        row_values.extend(
            [
                f"{row['creativity']:.2f}",
                f"{row['consistency']:.2f}",
                f"{row['plot']:.2f}",
                str(int(row["num_models"])) if "num_models" in row else "?",
                row["weight_class"],
            ]
        )

        # Add the row to the table
        table.add_row(*row_values)

    # Display the table in the terminal
    console.print(table)


# Add Typer CLI
app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)


@app.command()
def run_scoring(
    model_dir: Annotated[Optional[Path], typer.Option(help="Directory containing the model files (deprecated, use model_dirs)")] = None,
    model_dirs: Annotated[Optional[List[Path]], typer.Option(help="List of directories containing the model files to use as judges")] = None,
    submissions_dir: Annotated[Optional[Path], typer.Option(help="Directory containing submissions")] = None,
    scores_dir: Annotated[Optional[Path], typer.Option(help="Directory to save scoring results")] = None,
    max_new_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate")] = 1024,
    temperature: Annotated[float, typer.Option(help="Temperature for generation")] = 1.0,
    top_p: Annotated[float, typer.Option(help="Top-p sampling value")] = 1.0,
    cache_size: Annotated[int, typer.Option(help="Cache size in tokens")] = 1024 * 50,
    log_prompts: Annotated[bool, typer.Option(help="Log prompts and responses")] = False,
    upload: Annotated[bool, typer.Option(help="Upload results to HF repo")] = False,
    prompt_file: Annotated[str, typer.Option(help="Path to prompt file")] = "prompts/simple_prompt.yaml",
    reasoning_template: Annotated[Optional[str], typer.Option(help="Path to reasoning template YAML file for deep reasoning models")] = None,
    mode: Annotated[bool, typer.Option("--submit/--test", help="Submit to production (--submit) or test environment (--test). Defaults to test mode.")] = False,
    sub_test: Annotated[Optional[bool], typer.Option("--sub-test/--sub-prod", help="Override submission repository selection (test or prod). If not specified, follows the main mode.")] = None,
    score_test: Annotated[Optional[bool], typer.Option("--score-test/--score-prod", help="Override score repository selection (test or prod). If not specified, follows the main mode.")] = None,
    test_samples: Annotated[Optional[int], typer.Option(help="Number of test samples to score")] = None,
    draft_model_dir: Annotated[Optional[Path], typer.Option(help="Directory for draft model (speculative decoding)")] = None,
    draft_cache_size: Annotated[Optional[Path], typer.Option(help="Cache size for draft model")] = None,
    config: Annotated[Optional[Path], typer.Option(callback=conf_callback, is_eager=True, help="Relative path to YAML config file for setting options. Passing CLI options will supersede config options.", case_sensitive=False)] = None,
):  # fmt: skip
    """Run the scoring process, downloading new submissions and evaluating them.

    Parameters:
        model_dir: Directory containing model files (deprecated, use model_dirs)
        model_dirs: List of directories containing model files to use as judges
        submissions_dir: Directory containing submissions
        scores_dir: Directory to save scoring results
        max_new_tokens: Maximum tokens to generate
        temperature: Temperature for text generation
        top_p: Top-p sampling value
        cache_size: Cache size in tokens for model
        log_prompts: Whether to log prompts and responses
        upload: Whether to upload results to Hugging Face
        prompt_file: Path to file with evaluation prompts
        reasoning_template: Path to reasoning template YAML file for deep reasoning models
        mode: Submit to production or test environment
        sub_test/sub_prod: Override submission repository selection
        score_test/score_prod: Override score repository selection
        test_samples: Number of samples to test per submission
        draft_model_dir: Directory for draft model (speculative decoding)
        draft_cache_size: Cache size for draft model
    """

    # Handle backward compatibility for model_dir vs model_dirs
    if model_dirs is None:
        if model_dir is None:
            console.print("[red]Error: At least one model directory must be provided via --model-dir or --model-dirs[/red]")
            return
        model_dirs = [model_dir]
    elif model_dir is not None:
        # Both were provided, prioritize model_dirs but warn
        console.print("[yellow]Warning: Both --model-dir and --model-dirs provided. Using --model-dirs and ignoring --model-dir.[/yellow]")

    # Validate that model directories exist
    for directory in model_dirs:
        if not os.path.isdir(directory):
            console.print(f"[red]Error: Model directory not found: {directory}[/red]")
            return

    # Validate that prompt file exists
    if not os.path.exists(prompt_file):
        console.print(f"[red]Error: Prompt file not found: {prompt_file}[/red]")
        return

    # Handle deep reasoning mode
    if reasoning_template is not None:
        # Check that the reasoning template exists
        if not os.path.exists(reasoning_template):
            console.print(f"[red]Error: Reasoning template file {reasoning_template} not found[/red]")
            return
        console.print(f"[green]Using reasoning template: {reasoning_template}[/green]")

    # Extract model names (use directory name)
    model_names = [os.path.basename(os.path.normpath(d)) for d in model_dirs]
    console.print(f"[blue]Using {len(model_names)} judging models: {', '.join(model_names)}[/blue]")

    # Resolve which mode to use for inputs and outputs
    # If specific overrides aren't provided, follow the main mode
    use_test_sub = sub_test if sub_test is not None else not mode
    use_test_score = score_test if score_test is not None else not mode

    # Provide feedback about the mode
    if not mode:
        console.print("[yellow]TEST MODE - scoring will use test repositories and directories[/yellow]")
        if use_test_sub != use_test_score:
            console.print(f"[yellow]MIXED MODE - Submissions: {'TEST' if use_test_sub else 'PRODUCTION'}, Scores: {'TEST' if use_test_score else 'PRODUCTION'}[/yellow]")  # fmt: skip
    else:
        console.print("[green]PRODUCTION MODE - scoring will use production repositories and directories[/green]")
        if use_test_sub != use_test_score:
            console.print(f"[yellow]MIXED MODE - Submissions: {'TEST' if use_test_sub else 'PRODUCTION'}, Scores: {'TEST' if use_test_score else 'PRODUCTION'}[/yellow]")  # fmt: skip

    # Set repository IDs based on mode
    if use_test_sub:
        submission_repo_id = "cluster-of-stars/TinyStoriesHackathon_Submissions_Test"
        submission_dir = submissions_dir or "downloaded_submissions_test"
    else:
        submission_repo_id = "cluster-of-stars/TinyStoriesHackathon_Submissions"
        submission_dir = submissions_dir or "downloaded_submissions"

    if use_test_score:
        output_repo_id = "cluster-of-stars/TinyStoriesHackathon_Scores_Test"
        scores_dir_path = scores_dir or "scores_test"
    else:
        output_repo_id = "cluster-of-stars/TinyStoriesHackathon_Scores"
        scores_dir_path = scores_dir or "scores"

    # Print repository information
    console.print(f"[blue]Submission Repository: {submission_repo_id}[/blue]")
    console.print(f"[blue]Local Submissions Directory: {submission_dir}[/blue]")
    console.print(f"[blue]Score Repository: {output_repo_id}[/blue]")
    console.print(f"[blue]Local Scores Directory: {scores_dir_path}[/blue]")

    # Download new submissions
    if upload and not mode:
        console.print("[yellow]In test mode with upload enabled - will wait 3 seconds before proceeding...[/yellow]")
        for i in range(3, 0, -1):
            console.print(f"[yellow]    {i} seconds remaining...[/yellow]")
            time.sleep(1)

    # Download new submissions
    console.print(f"[yellow]Downloading new submissions from {submission_repo_id}...[/yellow]")
    new_submissions = download_new_submissions(dataset_id=submission_repo_id, output_dir=submission_dir, is_test_mode=use_test_sub)
    console.print(f"[green]Found {len(new_submissions)} new or updated submissions[/green]")

    # Initialize scoring tracker and sync with repository if uploading
    scoring_tracker = ScoringTracker(is_test_mode=use_test_score)
    if upload:
        try:
            api = get_authenticated_api()
            scoring_tracker.sync_with_repository(output_repo_id, api)
        except Exception as e:
            console.print(f"[yellow]Could not authenticate to sync scoring state: {e}[/yellow]")

    # Determine which submissions need scoring for each model
    unscored_files_by_model = {model_name: [] for model_name in model_names}
    Path(submission_dir).mkdir(exist_ok=True, parents=True)
    Path(scores_dir_path).mkdir(exist_ok=True, parents=True)

    # Use a simple message instead of an indeterminate progress bar
    console.print("[yellow]Scanning for unscored submissions...[/yellow]")

    # Get list of all CSV files
    for user_dir in Path(submission_dir).glob("*"):
        if not user_dir.is_dir():
            continue

        for csv_file in user_dir.glob("*.csv"):
            # Skip summary files
            if csv_file.name == "submission_summary.csv" or csv_file.name == "score_history.csv":
                continue

            username = user_dir.name
            submission_id = csv_file.stem

            # Check which models need to score this submission
            for model_name in model_names:
                # Check if this model has already scored this submission
                if not scoring_tracker.is_submission_scored(username, submission_id, model_name):
                    unscored_files_by_model[model_name].append(
                        {
                            "username": username,
                            "submission_id": submission_id,
                            "csv_path": csv_file,
                        }
                    )

    console.print("[green]Completed scanning for unscored submissions[/green]")

    # Print a summary of what needs to be scored
    for model_name, unscored_files in unscored_files_by_model.items():
        console.print(f"[green]Model {model_name}: Found {len(unscored_files)} submissions needing scoring[/green]")

    # Check if there's anything to score
    total_unscored = sum(len(files) for files in unscored_files_by_model.values())
    if total_unscored == 0:
        console.print("[yellow]No submissions to score. Exiting.[/yellow]")
        return

    # Track successfully scored submissions
    successfully_scored_submissions = set()

    # Process each model separately
    for model_idx, model_name in enumerate(model_names):
        model_dir = model_dirs[model_idx]
        unscored_files = unscored_files_by_model[model_name]

        if not unscored_files:
            console.print(f"[yellow]No unscored submissions for model {model_name}. Skipping.[/yellow]")
            continue

        console.print(f"[yellow]Processing {len(unscored_files)} submissions with model {model_name}...[/yellow]")

        # Get CSV paths for this model's unscored submissions
        all_csv_paths = [item["csv_path"] for item in unscored_files]

        if test_samples:
            console.print(f"[yellow]Limiting to {test_samples} samples per submission for testing[/yellow]")

        # Score submissions with this model
        try:
            all_scores = llm_scoring.score_submission(
                submission_file=all_csv_paths,
                model_dir=model_dir,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                cache_size=cache_size,
                log_prompts=log_prompts,
                prompt_file=prompt_file,
                sample=test_samples,  # Pass the test_samples parameter
                draft_model_dir=draft_model_dir,
                draft_cache_size=draft_cache_size,
                reasoning_template=reasoning_template,  # Pass the reasoning template
            )

            # Validate scoring results before proceeding
            if not validate_scoring_results(all_scores):
                console.print(f"[red]Error: Scoring results for model {model_name} are invalid or empty. Skipping this model.[/red]")
                continue

        except Exception as e:
            console.print(f"[red]Error during scoring with model {model_name}: {e}[/red]")
            console.print("[red]Skipping this model and continuing with others.[/red]")
            continue

        # Process and save results for this model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[green]Saving scoring results for model {model_name}...", total=len(unscored_files))

            for item in unscored_files:
                username = item["username"]
                submission_id = item["submission_id"]
                csv_path = item["csv_path"]

                if username in all_scores and submission_id in all_scores[username]:
                    # Read original submission to get prompts and completions
                    try:
                        df = pd.read_csv(csv_path)
                        if "prompt" not in df.columns or "completion" not in df.columns:
                            console.print(f"[red]Error: Missing required columns in {csv_path}. Skipping.[/red]")
                            progress.update(task, advance=1)
                            continue

                        prompts = df["prompt"].tolist()
                        completions = df["completion"].tolist()

                        # Extract scores for this submission
                        submission_scores = all_scores[username][submission_id]

                        # Get this model's results
                        current_model_name = list(submission_scores.keys())[0]  # We'll rename this to our model_name
                        item_scores = submission_scores[current_model_name]["details"]

                        # Validate item scores
                        if not item_scores:
                            console.print(
                                f"[red]Error: No valid scores for {username}/{submission_id} with model {model_name}. Skipping.[/red]"
                            )
                            progress.update(task, advance=1)
                            continue

                        # Create a dictionary of item_id to scores
                        item_scores_dict = {entry["item_id"]: entry["scores"] for entry in item_scores}

                        # Verify scores contain actual values
                        if not any(scores for scores in item_scores_dict.values()):
                            console.print(
                                f"[red]Error: Empty scores for {username}/{submission_id} with model {model_name}. Skipping.[/red]"
                            )
                            progress.update(task, advance=1)
                            continue

                        # Save model CSV with our defined model_name
                        saved_path = save_model_csv(
                            username=username,
                            submission_id=submission_id,
                            model_name=model_name,  # Use our defined model name
                            prompts=prompts,
                            completions=completions,
                            item_scores=item_scores_dict,
                            base_dir=scores_dir_path,
                            overwrite=True,
                        )

                        # Verify the CSV was written with actual data
                        if saved_path.exists():
                            try:
                                saved_df = pd.read_csv(saved_path)
                                if not saved_df.empty and any(col.lower() in saved_df.columns for col in ScoreCategory):
                                    # Mark this model as having scored this submission
                                    scoring_tracker.mark_submission_scored(username, submission_id, model_name)
                                    # Add to successfully scored submissions
                                    successfully_scored_submissions.add((username, submission_id))
                                else:
                                    console.print(f"[red]Error: Saved CSV {saved_path} is empty or missing score columns. Skipping.[/red]")
                            except Exception as e:
                                console.print(f"[red]Error validating saved CSV {saved_path}: {e}. Skipping.[/red]")
                        else:
                            console.print(
                                f"[red]Error: Failed to save CSV for {username}/{submission_id} with model {model_name}. Skipping.[/red]"
                            )

                    except Exception as e:
                        console.print(f"[red]Error processing {username}/{submission_id} with model {model_name}: {e}. Skipping.[/red]")

                progress.update(task, advance=1)

        console.print(f"[green]Completed scoring with model {model_name}[/green]")

    # Check if we have any successfully scored submissions
    if not successfully_scored_submissions:
        console.print("[red]No submissions were successfully scored. Skipping summary generation and uploads.[/red]")
        return

    console.print(f"[green]Successfully scored {len(successfully_scored_submissions)} submissions[/green]")

    # After all models have scored, generate/update submission summaries
    console.print("[yellow]Generating submission summaries...[/yellow]")

    # Generate summaries only for successfully scored submissions
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Generating submission summaries...", total=len(successfully_scored_submissions))

        for username, submission_id in successfully_scored_submissions:
            try:
                # Save submission summary
                summary_path = save_submission_summary(
                    username=username,
                    submission_id=submission_id,
                    base_dir=scores_dir_path,
                    overwrite=True,
                    repo_id=output_repo_id if upload else None,
                )

                # Update user score history if summary was created and contains valid data
                if summary_path and summary_path.exists():
                    try:
                        summary_df = pd.read_csv(summary_path)
                        if not summary_df.empty and any(col.lower() in summary_df.columns for col in ScoreCategory):
                            update_user_score_history(
                                username=username,
                                submission_id=submission_id,
                                summary_df=summary_df,
                                base_dir=scores_dir_path,
                                overwrite=True,
                            )
                        else:
                            console.print(
                                f"[red]Error: Summary file {summary_path} is empty or missing score columns. Skipping history update.[/red]"
                            )
                    except Exception as e:
                        console.print(f"[red]Error processing summary for {username}/{submission_id}: {e}. Skipping history update.[/red]")
            except Exception as e:
                console.print(f"[red]Error generating summary for {username}/{submission_id}: {e}. Skipping.[/red]")

            progress.update(task, advance=1)

    # Generate leaderboards
    console.print("[yellow]Generating leaderboards...[/yellow]")
    try:
        leaderboard_files = generate_and_save_all_leaderboards(base_dir=scores_dir_path)
        if not leaderboard_files.empty:
            console.print("[green]Leaderboards generated successfully[/green]")
        else:
            console.print("[red]Error: Generated leaderboards are empty. Skipping upload.[/red]")
            return
    except Exception as e:
        console.print(f"[red]Error generating leaderboards: {e}. Skipping upload.[/red]")
        return

    # Upload results if requested
    if upload:
        console.print(f"[yellow]Uploading results to {output_repo_id}...[/yellow]")

        # Upload user files
        user_upload_results = upload_all_user_files(base_dir=scores_dir_path, repo_id=output_repo_id)
        console.print(f"[green]Uploaded {user_upload_results.get('uploaded', 0)} user files[/green]")

        # Upload leaderboards
        leaderboard_upload_results = upload_leaderboards(repo_id=output_repo_id)
        console.print(f"[green]Uploaded {leaderboard_upload_results.get('uploaded', 0)} leaderboard files[/green]")

        # Update repository README - force update if any submissions were scored
        readme_updated = update_repository_readme(repo_id=output_repo_id, force_update=len(successfully_scored_submissions) > 0)
        if readme_updated:
            console.print("[green]Updated repository README[/green]")
        else:
            console.print("[yellow]README update failed[/yellow]")

        # Upload scoring state again after all processing is complete (in case more scores were added)
        try:
            scoring_tracker.upload_to_repository(output_repo_id, api)
        except Exception as e:
            console.print(f"[yellow]Could not upload final scoring state: {e}[/yellow]")

    console.print("[green]Scoring complete![/green]")

    # Display leaderboard preview
    display_leaderboard_preview(leaderboard_files)


# Add a new function to validate scoring results
def validate_scoring_results(scores: Dict[str, Dict[str, Dict[str, Any]]]) -> bool:
    """Validate that scoring results contain actual data.

    Args:
        scores: The scoring dictionary to validate

    Returns:
        True if results are valid, False otherwise
    """
    if not scores:
        return False

    # Check that we have user entries
    for username, user_data in scores.items():
        if not user_data:
            return False

        # Check each submission
        for submission_id, submission_data in user_data.items():
            if not submission_data:
                return False

            # Check each model
            for model_name, model_data in submission_data.items():
                # Verify we have details and they contain actual scores
                if not model_data or "details" not in model_data or not model_data["details"]:
                    return False

                # Check each detail item has scores
                for detail in model_data["details"]:
                    if "item_id" not in detail or "scores" not in detail or not detail["scores"]:
                        return False

                    # Verify scores contain numerical values
                    for category, score in detail["scores"].items():
                        if not isinstance(score, (int, float)) or score <= 0:
                            return False

    return True


if __name__ == "__main__":
    app()
