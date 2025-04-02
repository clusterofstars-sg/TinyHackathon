import json
import re
import shutil
import tempfile
import time  # Import the time module
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union, Set

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
                console.print("[yellow]Warning: Could not parse upload state file. Creating new state.[/yellow]")
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
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        return mtime > last_upload_time

    def mark_file_uploaded(self, file_path: Path):
        """Mark a file as uploaded with current timestamp."""
        file_path_str = str(file_path)
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        self.state["last_upload"][file_path_str] = mtime
        self.save_state()

    def mark_readme_updated(self):
        """Mark the README as updated with current timestamp."""
        self.state["last_readme_update"] = datetime.now().isoformat()
        self.save_state()

    def should_update_readme(self, min_interval_hours=12) -> bool:
        """Check if README should be updated based on time interval."""
        last_update = self.state.get("last_readme_update")
        if last_update is None:
            return True

        last_update_time = datetime.fromisoformat(last_update)
        hours_since_update = (datetime.now() - last_update_time).total_seconds() / 3600
        return hours_since_update >= min_interval_hours


# File-based functions to replace ScoringTracker
def is_submission_scored(username: str, submission_id: str, model_name: Optional[str] = None, base_dir: str = "submissions") -> bool:
    """Check if a submission has been scored by looking for CSV files.

    Args:
    username: The username of the submission owner
        submission_id: The ID of the submission
        model_name: Optional model name to check if this specific model has scored the submission
    base_dir: Base directory for submissions

    Returns:
        True if the submission has been scored (by the specified model if model_name is provided)
    """
    submission_dir = Path(base_dir) / username / submission_id

    # If the directory doesn't exist, the submission hasn't been scored
    if not submission_dir.exists():
        return False

    # If model_name is specified, check for that specific model's CSV file
    if model_name is not None:
        model_csv = submission_dir / f"{model_name}.csv"
        return model_csv.exists()

    # Otherwise, check if any model has scored this submission (look for any CSV file except summary files)
    model_csvs = list(submission_dir.glob("*.csv"))
    return any(csv_file.name != "submission_summary.csv" and csv_file.name != "score_history.csv" for csv_file in model_csvs)


def get_scored_models(username: str, submission_id: str, base_dir: str = "submissions") -> List[str]:
    """Get list of models that have scored a submission by checking CSV files.

    Args:
    username: The username of the submission owner
        submission_id: The ID of the submission
    base_dir: Base directory for submissions

    Returns:
        List of model names that have scored this submission
    """
    submission_dir = Path(base_dir) / username / submission_id

    if not submission_dir.exists():
        return []

    # Get all CSV files except submission_summary.csv and score_history.csv
    model_csvs = [
        csv_file for csv_file in submission_dir.glob("*.csv") if csv_file.name not in ("submission_summary.csv", "score_history.csv")
    ]

    # Extract model names from filenames (without .csv extension)
    return [csv_file.stem for csv_file in model_csvs]


def get_all_scored_submissions(base_dir: str = "submissions") -> Dict[str, Dict[str, Dict[str, Union[List[str], str]]]]:
    """Get all scored submissions by scanning the filesystem.

    Args:
        base_dir: Base directory for submissions

    Returns:
        Dictionary mapping usernames to dictionaries mapping submission_ids to information about scored submissions
    """
    result = {"scored_submissions": {}}
    base_path = Path(base_dir)

    # Scan all user directories
    for user_dir in base_path.glob("*"):
        if not user_dir.is_dir():
            continue

        username = user_dir.name
        result["scored_submissions"][username] = {}

        # Scan all submission directories for this user
        for submission_dir in user_dir.glob("*"):
            if not submission_dir.is_dir():
                continue

            submission_id = submission_dir.name

            # Get scored models for this submission
            scored_models = get_scored_models(username, submission_id, base_dir)

            if scored_models:
                timestamp = (
                    extract_submission_datetime(submission_id) if "extract_submission_datetime" in globals() else datetime.now().isoformat()
                )
                result["scored_submissions"][username][submission_id] = {"models": scored_models, "timestamp": timestamp}

    return result


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
            ts_datetime = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            # Format as YYYY-MM-DD HH:MM
            return ts_datetime.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            # If parsing fails, return current time in simplified format
            return datetime.now().strftime("%Y-%m-%d %H:%M")
    else:
        # If no timestamp found, return current time in simplified format
        return datetime.now().strftime("%Y-%m-%d %H:%M")


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
    include_texts: bool = False,
) -> pd.DataFrame:
    """Generate a per-model DataFrame for a submission.

    Args:
        username: Username of the submission
        submission_id: ID of the submission
        model_name: Name of the model
        prompts: List of prompts
        completions: List of completions
        item_scores: Dictionary mapping item IDs to score dictionaries
        include_texts: Whether to include prompt and completion texts in the output

    Returns:
        DataFrame containing the scores
    """
    data = []
    for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
        if idx in item_scores:
            row = {
                "item_id": idx,
            }
            # Only include text data if requested
            if include_texts:
                row["prompt"] = prompt
                row["completion"] = completion

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

    # Generate the DataFrame without prompt and completion columns
    df = generate_model_dataframe(
        username=username,
        submission_id=submission_id,
        model_name=model_name,
        prompts=prompts,
        completions=completions,
        item_scores=item_scores,
        include_texts=False,  # Don't include prompt and completion texts
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

            # If num_models is 0 or 1, try to calculate it directly
            if num_models <= 1:
                # Get the submission_id
                submission_id = best_submission["submission_id"] if "submission_id" in best_submission else ""

                if submission_id:
                    # Find model CSV files to get actual model count
                    model_csvs = find_model_csvs(username, submission_id, base_dir)
                    if model_csvs:
                        num_models = len(model_csvs)

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
        pd.DataFrame: The global leaderboard DataFrame.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate global leaderboard
    global_leaderboard = generate_global_leaderboard(base_dir=base_dir)

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
    if metadata_path.exists():
        remote_path = f"submissions/{username}"
        results = upload_with_folder_fallback(
            api=api, local_path=metadata_path, remote_path=remote_path, repo_id=repo_id, tracker=tracker, results=results
        )

    # Upload score history if it exists
    history_path = user_dir / "score_history.csv"
    if history_path.exists():
        remote_path = f"submissions/{username}"
        results = upload_with_folder_fallback(
            api=api, local_path=history_path, remote_path=remote_path, repo_id=repo_id, tracker=tracker, results=results
        )

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
    # Set the remote path for this submission
    remote_path = f"submissions/{username}/{submission_id}"

    # Use the consolidated upload helper with CSV file pattern
    return upload_with_folder_fallback(
        api=api, local_path=submission_dir, remote_path=remote_path, repo_id=repo_id, tracker=tracker, file_pattern="*.csv", results=results
    )


def upload_with_folder_fallback(
    api: HfApi,
    local_path: Path,
    remote_path: str,
    repo_id: str,
    tracker: UploadTracker,
    file_pattern: str = "*",
    results: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Helper function to upload files with folder-level upload and individual file fallback.

    This consolidates the common upload pattern used throughout the codebase.

    Args:
        api: Authenticated Hugging Face API client
        local_path: Path to local directory or file to upload
        remote_path: Path in the repository where files should be uploaded
        repo_id: Hugging Face repository ID
        tracker: Upload tracker to avoid redundant uploads
        file_pattern: Glob pattern to filter files (default: "*" for all files)
        results: Optional existing results dictionary to update

    Returns:
        Dictionary with counts of uploaded files, skipped files, and errors
    """
    if results is None:
        results = {"uploaded": 0, "skipped": 0, "error": 0}

    # Handle directory vs file
    if local_path.is_dir():
        # Get all files matching the pattern
        all_files = list(local_path.glob(file_pattern))

        # Filter to files that need uploading
        files_to_upload = [f for f in all_files if tracker.should_upload_file(f)]

        if not files_to_upload:
            # Nothing needs uploading
            results["skipped"] += len(all_files)
            return results

        try:
            # Try bulk folder upload first
            _ = api.upload_folder(folder_path=str(local_path), path_in_repo=remote_path, repo_id=repo_id, repo_type="dataset")

            # Mark all files as uploaded
            for file_path in all_files:
                tracker.mark_file_uploaded(file_path)

            # Update metrics
            results["uploaded"] += len(files_to_upload)
            console.print(f"[green]Uploaded folder {local_path} to {remote_path}[/green]")

        except Exception as e:
            console.print(f"[red]Error uploading folder {local_path}: {e}[/red]")
            results["error"] += 1

            # Try individual files as fallback
            console.print("[yellow]Attempting individual file uploads as fallback...[/yellow]")
            for file_path in files_to_upload:
                try:
                    # Determine relative path for the file
                    rel_path = file_path.relative_to(local_path)
                    file_remote_path = f"{remote_path}/{rel_path}" if remote_path else str(rel_path)

                    api.upload_file(path_or_fileobj=str(file_path), path_in_repo=file_remote_path, repo_id=repo_id, repo_type="dataset")
                    tracker.mark_file_uploaded(file_path)
                    results["uploaded"] += 1
                    console.print(f"[green]Uploaded {file_path} to {file_remote_path}[/green]")
                except Exception as e2:
                    console.print(f"[red]Error uploading {file_path}: {e2}[/red]")
                    results["error"] += 1
    else:
        # Handle single file upload
        if tracker.should_upload_file(local_path):
            try:
                # For a single file, determine the remote path
                file_remote_path = f"{remote_path}/{local_path.name}" if remote_path else local_path.name

                api.upload_file(path_or_fileobj=str(local_path), path_in_repo=file_remote_path, repo_id=repo_id, repo_type="dataset")
                tracker.mark_file_uploaded(local_path)
                results["uploaded"] += 1
                console.print(f"[green]Uploaded {local_path} to {file_remote_path}[/green]")
            except Exception as e:
                console.print(f"[red]Error uploading {local_path}: {e}[/red]")
                results["error"] += 1
        else:
            results["skipped"] += 1

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

    # Define leaderboard file patterns to look for
    file_pattern = "leaderboard_*.csv"

    # Use the consolidated upload helper
    results = upload_with_folder_fallback(
        api=api,
        local_path=leaderboard_path,
        remote_path="",  # Upload to root
        repo_id=repo_id,
        tracker=tracker,
        file_pattern=file_pattern,
    )

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
    aoe_timezone = timezone(timedelta(hours=-12))
    timestamp = datetime.now().replace(tzinfo=timezone.utc).astimezone(aoe_timezone).strftime("%Y-%m-%d %H:%M:%S AOE")

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

    # Format integer columns
    if "num_models" in global_display_df.columns:
        global_display_df["num_models"] = global_display_df["num_models"].map(lambda x: int(x))

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

                # Format integer columns
                if "num_models" in display_df.columns:
                    display_df["num_models"] = display_df["num_models"].map(lambda x: int(x))

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
        console.print("[yellow]README was updated recently. Skipping update.[/yellow]")
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
            console.print("[green]Updated README.md with leaderboard information and YAML metadata[/green]")
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


def download_existing_scores(
    repo_id: str = "cluster-of-stars/TinyStoriesHackathon_Scores",
    base_dir: str = "submissions",
    is_test_mode: bool = False,
) -> Dict[str, int]:
    """Download all previously scored submissions from the HF repository.

    This ensures we have the latest scores, including those scored by others.

    Args:
        repo_id: Hugging Face repo ID
        base_dir: Base directory to save scores
        is_test_mode: Whether to use test mode

    Returns:
        Dictionary with counts of downloaded files
    """
    console.print(f"[yellow]Downloading existing scores from {repo_id}...[/yellow]")

    # Get HF API
    try:
        _, api = get_hf_user()
    except Exception as e:
        console.print(f"[red]Authentication error: {e}[/red]")
        return {"downloaded": 0, "skipped": 0, "error": 1}

    # Get all files in the repository
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        console.print(f"[red]Error listing repository files: {e}[/red]")
        return {"downloaded": 0, "skipped": 0, "error": 1}

    # Track download statistics
    results = {"downloaded": 0, "skipped": 0, "error": 0}

    # Find and download submission files
    # First scan for summary files to determine which submissions have been scored
    scored_submissions = {}
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

                # Add to scored submissions
                if username not in scored_submissions:
                    scored_submissions[username] = {}

                if submission_id not in scored_submissions[username]:
                    scored_submissions[username][submission_id] = {
                        "timestamp": extract_submission_datetime(submission_id),
                        "models": model_names,
                    }
                else:
                    # Update model list
                    existing_models = scored_submissions[username][submission_id].get("models", [])
                    updated_models = list(set(existing_models + model_names))
                    scored_submissions[username][submission_id]["models"] = updated_models
                    # Update timestamp
                    scored_submissions[username][submission_id]["timestamp"] = extract_submission_datetime(submission_id)

    # Download all summary files first
    console.print("[yellow]Downloading submission summaries and model scores...[/yellow]")
    for file_path in files:
        # Only process files in the submissions directory
        if not file_path.startswith("submissions/"):
            continue

        # Extract parts
        parts = file_path.split("/")
        if len(parts) < 4:
            continue

        username = parts[1]
        submission_id = parts[2]
        filename = parts[3]

        # Create directory structure
        submission_dir = Path(base_dir) / username / submission_id
        submission_dir.mkdir(parents=True, exist_ok=True)

        # Download the file
        local_path = submission_dir / filename
        try:
            if not local_path.exists():
                _ = api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=file_path, local_dir=Path(base_dir))
                results["downloaded"] += 1
            else:
                results["skipped"] += 1
        except Exception as e:
            console.print(f"[red]Error downloading {file_path}: {e}[/red]")
            results["error"] += 1

    # Download user histories
    for username in scored_submissions:
        history_path = f"submissions/{username}/score_history.csv"
        local_history_path = Path(base_dir) / username / "score_history.csv"

        if history_path in files:
            if not local_history_path.exists() or is_test_mode:  # Always download in test mode for testing
                try:
                    _ = api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=history_path, local_dir=Path(base_dir))
                    results["downloaded"] += 1
                    console.print(f"[green]Downloaded user history for {username}[/green]")
                except Exception as e:
                    console.print(f"[red]Error downloading history for {username}: {e}[/red]")
                    results["error"] += 1
            else:
                console.print(f"[blue]Skipped existing user history for {username}[/blue]")
                results["skipped"] += 1

    # Download leaderboard files
    leaderboard_files = [f for f in files if f.startswith("leaderboards/")]
    for file_path in leaderboard_files:
        local_path = Path(file_path)  # Relative path from workspace root
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not local_path.exists() or is_test_mode:  # Always download in test mode
                _ = api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=file_path)
                results["downloaded"] += 1
            else:
                results["skipped"] += 1
        except Exception as e:
            console.print(f"[red]Error downloading {file_path}: {e}[/red]")
            results["error"] += 1

    console.print(
        f"[green]Downloaded {results['downloaded']} files, skipped {results['skipped']} existing files, encountered {results['error']} errors[/green]"
    )
    return results


class EnvConfig:
    """Configuration object for environment settings.

    This class consolidates various environment flags into a single object
    to simplify environment mode handling throughout the codebase.
    """

    def __init__(self, is_test_mode: bool, sub_test_override: Optional[bool] = None, score_test_override: Optional[bool] = None):
        """Initialize environment configuration.

        Args:
            is_test_mode: Base test mode setting (True for test mode, False for production)
            sub_test_override: Optional override for submissions repository test mode
            score_test_override: Optional override for scores repository test mode
        """
        # Base test mode setting
        self.is_test_mode = is_test_mode

        # Apply overrides if specified, otherwise use the base setting
        self.use_test_sub = sub_test_override if sub_test_override is not None else is_test_mode
        self.use_test_score = score_test_override if score_test_override is not None else is_test_mode

        # Set repository IDs based on test mode settings
        self.submissions_repo_id = (
            "cluster-of-stars/TinyStoriesHackathon_Submissions_Test"
            if self.use_test_sub
            else "cluster-of-stars/TinyStoriesHackathon_Submissions"
        )

        self.scores_repo_id = (
            "cluster-of-stars/TinyStoriesHackathon_Scores_Test" if self.use_test_score else "cluster-of-stars/TinyStoriesHackathon_Scores"
        )

        # Set download directories
        self.submission_dir = "downloaded_submissions_test" if self.use_test_sub else "downloaded_submissions"
        self.scores_dir = "scores_test" if self.use_test_score else "scores"

    def __str__(self) -> str:
        """Return string representation of the configuration."""
        mode = "TEST" if self.is_test_mode else "PRODUCTION"
        sub_mode = "TEST" if self.use_test_sub else "PRODUCTION"
        score_mode = "TEST" if self.use_test_score else "PRODUCTION"

        return (
            f"Environment: {mode} mode\n"
            f"  Submissions: {sub_mode} mode - {self.submissions_repo_id}\n"
            f"  Scores: {score_mode} mode - {self.scores_repo_id}\n"
            f"  Directories: {self.submission_dir}, {self.scores_dir}"
        )


# Add Typer CLI
app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)


def setup_environment(
    mode: bool, sub_test: Optional[bool], score_test: Optional[bool], model_dir: Optional[Path], model_dirs: Optional[List[Path]]
) -> Tuple[EnvConfig, List[Path], List[str]]:
    """Set up the environment configuration and model directories.

    This function handles the environment configuration setup, model directory
    validation, and model name extraction.

    Args:
        mode: True for production mode (--submit), False for test mode (--test)
        sub_test: Override for submission repository test mode
        score_test: Override for scores repository test mode
        model_dir: Single model directory (deprecated)
        model_dirs: List of model directories to use

    Returns:
        Tuple containing:
        - EnvConfig: The environment configuration
        - List[Path]: List of model directories to use
        - List[str]: List of model names (derived from directory names)
    """
    # Create environment configuration
    is_test_mode = not mode  # False = test, True = submit
    env_config = EnvConfig(is_test_mode=is_test_mode, sub_test_override=sub_test, score_test_override=score_test)

    # Print environment status
    console.print(f"[bold {'yellow' if is_test_mode else 'green'}]{env_config}[/bold {'yellow' if is_test_mode else 'green'}]")

    # Handle backward compatibility for model_dir vs model_dirs
    if model_dirs is None:
        if model_dir is None:
            console.print("[red]Error: At least one model directory must be provided via --model-dir or --model-dirs[/red]")
            raise ValueError("No model directory provided")
        model_dirs = [model_dir]
    elif model_dir is not None:
        # Both were provided, prioritize model_dirs but warn
        console.print("[yellow]Warning: Both --model-dir and --model-dirs provided. Using --model-dirs and ignoring --model-dir.[/yellow]")

    # Extract model names (use directory basename)
    model_names = [Path(d).name for d in model_dirs]
    console.print(f"[blue]Using {len(model_names)} judging models: {', '.join(model_names)}[/blue]")

    return env_config, model_dirs, model_names


def setup_api_client(upload: bool) -> Optional[HfApi]:
    """Set up the Hugging Face API client for uploads if requested.

    This function handles authentication with HuggingFace and returns
    an authenticated API client if successful.

    Args:
        upload: Whether uploads are enabled (if False, returns None)

    Returns:
        Optional[HfApi]: Authenticated API client if upload is True and
        authentication succeeds, None otherwise
    """
    if not upload:
        console.print("[yellow]Upload is disabled, skipping API authentication[/yellow]")
        return None

    try:
        _, api = get_hf_user()
        console.print("[green]Successfully authenticated with Hugging Face[/green]")
        return api
    except Exception as e:
        console.print(f"[red]Error authenticating with Hugging Face: {e}[/red]")
        console.print("[red]Continuing without upload capability[/red]")
        return None


def download_submissions_and_scores(
    env_config: EnvConfig, submission_dir: str, scores_dir: str, api_client: Optional[HfApi] = None
) -> Dict[str, Any]:
    """Download new submissions and existing scores.

    This function handles downloading new submissions from the submissions repository
    and existing scores from the scores repository.

    Args:
        env_config: Environment configuration
        submission_dir: Directory to save downloaded submissions
        scores_dir: Directory to save scores
        api_client: Optional authenticated API client (required for downloading scores)

    Returns:
        Dictionary containing information about downloaded submissions and scores
    """
    result = {"new_submissions": [], "scores_downloaded": False, "submission_dir": submission_dir, "scores_dir": scores_dir}

    # Create necessary directories
    Path(submission_dir).mkdir(exist_ok=True, parents=True)
    Path(scores_dir).mkdir(exist_ok=True, parents=True)

    # Download new submissions
    console.print(f"[yellow]Downloading new submissions from {env_config.submissions_repo_id}...[/yellow]")
    try:
        new_submissions = download_new_submissions(
            dataset_id=env_config.submissions_repo_id, output_dir=submission_dir, is_test_mode=env_config.use_test_sub
        )

        if new_submissions:
            console.print(f"[green]Downloaded {len(new_submissions)} new submissions[/green]")
            result["new_submissions"] = new_submissions
        else:
            console.print("[yellow]No new submissions to download[/yellow]")
    except Exception as e:
        console.print(f"[red]Error downloading new submissions: {e}[/red]")
        # Continue with existing submissions if download fails

    # Download existing scores if API client is available
    if api_client is not None:
        try:
            # Initial download of existing scores to:
            # 1. Avoid duplicate work by having local copies of already scored submissions
            # 2. Ensure we have the most current state before determining which submissions need scoring
            # 3. This is especially important in collaborative environments where multiple scorers may be working
            download_stats = download_existing_scores(
                repo_id=env_config.scores_repo_id, base_dir=scores_dir, is_test_mode=env_config.use_test_score
            )

            result["scores_downloaded"] = True
            result["download_stats"] = download_stats
        except Exception as e:
            console.print(f"[yellow]Could not download existing scores: {e}[/yellow]")
    else:
        console.print("[yellow]No API client available, skipping score download[/yellow]")

    return result


def find_unscored_submissions(
    submission_dir: Union[str, Path], scores_dir: Union[str, Path], model_names: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Find submissions that need to be scored by each model.

    This function scans the submissions directory to identify which
    submissions have not yet been scored by which models.

    Args:
        submission_dir: Directory containing submissions
        scores_dir: Directory containing scores
        model_names: List of model names to check

    Returns:
        Dictionary mapping model names to lists of submissions that need scoring
    """
    # Convert paths to Path objects if they aren't already
    submission_dir = Path(submission_dir)
    scores_dir = Path(scores_dir)

    # Initialize a dictionary to store unscored submissions for each model
    unscored_files_by_model = {model_name: [] for model_name in model_names}

    # Use a simple message instead of an indeterminate progress bar
    console.print("[yellow]Scanning for unscored submissions...[/yellow]")

    submission_count = 0
    unscored_count = 0

    # Get list of all CSV files in the submissions directory
    for user_dir in submission_dir.glob("*"):
        if not user_dir.is_dir():
            continue

        for csv_file in user_dir.glob("*.csv"):
            # Skip summary files
            if csv_file.name == "submission_summary.csv" or csv_file.name == "score_history.csv":
                continue

            username = user_dir.name
            submission_id = csv_file.stem
            submission_count += 1

            # Check which models need to score this submission
            for model_name in model_names:
                # Check if this model has already scored this submission
                if not is_submission_scored(username, submission_id, model_name, base_dir=scores_dir):
                    unscored_files_by_model[model_name].append(
                        {
                            "username": username,
                            "submission_id": submission_id,
                            "csv_path": csv_file,
                        }
                    )
                    unscored_count += 1

    console.print("[green]Completed scanning for unscored submissions[/green]")

    # Print a summary of what needs to be scored
    for model_name, unscored_files in unscored_files_by_model.items():
        console.print(f"[green]Model {model_name}: Found {len(unscored_files)} submissions needing scoring[/green]")

    console.print(f"[blue]Total submissions: {submission_count}, Total unscored items: {unscored_count}[/blue]\n")

    return unscored_files_by_model


def score_submissions(
    unscored_files_by_model: Dict[str, List[Dict[str, Any]]],
    model_dirs: List[Path],
    model_names: List[str],
    scores_dir: Path,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    cache_size: int = 1024 * 50,
    prompt_file: str = "prompts/simple_prompt.yaml",
    reasoning_template: Optional[str] = None,
    log_prompts: bool = False,
    test_samples: Optional[int] = None,
    draft_model_dir: Optional[Path] = None,
    draft_cache_size: Optional[int] = None,
) -> Set[Tuple[str, str]]:
    """Score submissions with the specified models.

    This function handles running models on submissions that need scoring,
    and saving the results to CSV files.

    Args:
        unscored_files_by_model: Dictionary mapping model names to lists of submissions needing scoring
        model_dirs: List of model directories
        model_names: List of model names
        scores_dir: Directory to save score results
        max_new_tokens: Maximum tokens to generate
        temperature: Temperature for text generation
        top_p: Top-p sampling value
        cache_size: Cache size in tokens for model
        prompt_file: Path to file with evaluation prompts
        reasoning_template: Path to reasoning template YAML file for deep reasoning models
        log_prompts: Whether to log prompts and responses
        test_samples: Number of samples to test per submission (if specified)
        draft_model_dir: Directory for draft model (speculative decoding)
        draft_cache_size: Cache size for draft model

    Returns:
        Set of tuples (username, submission_id) for successfully scored submissions
    """
    # Initialize set to track successfully scored submissions
    successfully_scored_submissions = set()

    # Process each model separately
    for model_idx, model_name in enumerate(model_names):
        model_dir = model_dirs[model_idx]
        unscored_files = unscored_files_by_model[model_name]

        if not unscored_files:
            console.print(f"[yellow]No unscored submissions for model {model_name}. Skipping.[/yellow]")
            continue

        console.print(f"[yellow]Processing {len(unscored_files)} submissions with model {model_name}...[/yellow]")

        if test_samples:
            console.print(f"[yellow]Limiting to {test_samples} samples per submission for testing[/yellow]")

        all_scores = {}

        try:
            # Score all submissions at once
            all_scores = llm_scoring.score_submission(
                submission_file=[item["csv_path"] for item in unscored_files],
                model_dir=model_dir,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                cache_size=cache_size,
                log_prompts=log_prompts,
                prompt_file=prompt_file,
                sample=test_samples,
                draft_model_dir=draft_model_dir,
                draft_cache_size=draft_cache_size,
                reasoning_template=reasoning_template,
            )
        except Exception as e:
            console.print(f"[red]Error scoring submissions with {model_name}: {e}[/red]")
            continue

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
                        console.print(f"[red]Error: No valid scores for {username}/{submission_id} with model {model_name}. Skipping.[/red]")  # fmt: skip
                        continue

                    # Create a dictionary of item_id to scores
                    item_scores_dict = {entry["item_id"]: entry["scores"] for entry in item_scores}

                    # Verify scores contain actual values
                    if not any(scores for scores in item_scores_dict.values()):
                        console.print(f"[red]Error: Empty scores for {username}/{submission_id} with model {model_name}. Skipping.[/red]")
                        continue

                    # Save model CSV with our defined model_name
                    saved_path = save_model_csv(
                        username=username,
                        submission_id=submission_id,
                        model_name=model_name,  # Use our defined model name
                        prompts=prompts,
                        completions=completions,
                        item_scores=item_scores_dict,
                        base_dir=scores_dir,
                        overwrite=True,
                    )

                    # Verify the CSV was written with actual data
                    if saved_path.exists():
                        try:
                            saved_df = pd.read_csv(saved_path)
                            if not saved_df.empty and any(col.lower() in saved_df.columns for col in ScoreCategory):
                                # Add to successfully scored submissions
                                successfully_scored_submissions.add((username, submission_id))
                            else:
                                console.print(f"[red]Error: Saved CSV {saved_path} is empty or missing score columns. Skipping.[/red]")  # fmt: skip
                        except Exception as e:
                            console.print(f"[red]Error validating saved CSV {saved_path}: {e}. Skipping.[/red]")
                    else:
                        console.print(f"[red]Error: Failed to save CSV for {username}/{submission_id} with model {model_name}. Skipping.[/red]")  # fmt: skip

                except Exception as e:
                    console.print(f"[red]Error processing {username}/{submission_id} with model {model_name}: {e}. Skipping.[/red]")

        console.print(f"[green]Completed scoring with model {model_name}[/green]")

    # Check if we have any successfully scored submissions
    if not successfully_scored_submissions:
        console.print("[yellow]No submissions were successfully scored.[/yellow]")
    else:
        console.print(f"[green]Successfully scored {len(successfully_scored_submissions)} submissions[/green]")

    return successfully_scored_submissions


def generate_summaries_and_history(
    scored_submissions: Set[Tuple[str, str]], scores_dir: Union[Path, str], overwrite: bool = False, repo_id: Optional[str] = None
) -> Dict[str, List[Path]]:
    """Generate submission summaries and update score history for scored submissions.

    This function generates summary files for each scored submission and updates
    the user's score history with the new results.

    Args:
        scored_submissions: Set of (username, submission_id) tuples for successfully scored submissions
        scores_dir: Directory containing the scored results
        overwrite: Whether to overwrite existing summary files
        repo_id: Repository ID for including in paths (optional)

    Returns:
        Dictionary containing lists of generated summary and history files
    """
    scores_dir = Path(scores_dir)

    # Initialize result tracking
    results = {"summary_files": [], "history_files": []}

    if not scored_submissions:
        console.print("[yellow]No submissions to generate summaries for.[/yellow]")
        return results

    console.print(f"[yellow]Generating summaries and updating history for {len(scored_submissions)} submissions...[/yellow]")

    for username, submission_id in scored_submissions:
        try:
            # Generate and save submission summary
            summary_path = save_submission_summary(
                username=username, submission_id=submission_id, base_dir=scores_dir, overwrite=overwrite, repo_id=repo_id
            )

            if summary_path:
                results["summary_files"].append(summary_path)

                # Check if the summary CSV was successfully created
                try:
                    summary_df = pd.read_csv(summary_path)

                    # Update user score history
                    history_path = update_user_score_history(
                        username=username,
                        submission_id=submission_id,
                        summary_df=summary_df,
                        base_dir=scores_dir,
                        overwrite=True,  # Always overwrite history to ensure latest
                    )

                    if history_path:
                        results["history_files"].append(history_path)
                    else:
                        console.print(f"[red]Failed to update history for {username}/{submission_id}[/red]")

                except Exception as e:
                    console.print(f"[red]Error reading summary file for history update ({username}/{submission_id}): {e}[/red]")

            else:
                console.print(f"[red]Failed to generate summary for {username}/{submission_id}[/red]")

        except Exception as e:
            console.print(f"[red]Error processing summary/history for {username}/{submission_id}: {e}[/red]")

    # Report results
    console.print(f"[green]Generated {len(results['summary_files'])} summary files[/green]")
    console.print(f"[green]Updated {len(results['history_files'])} history files[/green]")

    return results


def generate_and_upload_leaderboards(
    scores_dir: Union[Path, str],
    leaderboard_dir: Union[Path, str] = "leaderboards",
    upload: bool = False,
    api_client: Optional[HfApi] = None,
    repo_id: Optional[str] = None,
    update_readme: bool = True,
    force_readme_update: bool = False,
) -> Dict[str, Any]:
    """Generate leaderboards and optionally upload them to Hugging Face.

    This function generates global and weight class leaderboards, saves them to files,
    and optionally uploads them to Hugging Face. It can also update the README file.

    Args:
        scores_dir: Directory containing score files
        leaderboard_dir: Directory to save leaderboard files
        upload: Whether to upload leaderboards
        api_client: Authenticated Hugging Face API client (required if upload=True)
        repo_id: Repository ID for uploads (required if upload=True)
        update_readme: Whether to update the repository README
        force_readme_update: Whether to force README update regardless of time interval

    Returns:
        Dictionary with results including leaderboard data and upload statistics
    """
    # Convert paths to Path objects
    scores_dir = Path(scores_dir)
    leaderboard_dir = Path(leaderboard_dir)

    # Create leaderboard directory if it doesn't exist
    leaderboard_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results dictionary
    results = {
        "leaderboard": None,
        "weight_class_leaderboards": {},
        "saved_files": {},
        "upload_stats": {"files": 0, "errors": 0},
        "readme_updated": False,
    }

    console.print("[yellow]Generating leaderboards...[/yellow]")

    try:
        # Step 1: Generate global leaderboard
        global_leaderboard = generate_global_leaderboard(base_dir=scores_dir)
        results["leaderboard"] = global_leaderboard

        if global_leaderboard.empty:
            console.print("[yellow]Global leaderboard is empty. No score data found.[/yellow]")
            return results

        # Display a preview of the leaderboard
        display_leaderboard_preview(global_leaderboard)

        # Step 2: Generate weight class leaderboards
        weight_class_leaderboards = generate_weight_class_leaderboards(global_leaderboard)
        results["weight_class_leaderboards"] = weight_class_leaderboards

        # Step 3: Save global leaderboard
        global_leaderboard_files = save_global_leaderboard(global_leaderboard, base_dir=leaderboard_dir)
        results["saved_files"]["global"] = global_leaderboard_files

        # Step 4: Save weight class leaderboards
        weight_class_files = save_weight_class_leaderboards(weight_class_leaderboards, base_dir=leaderboard_dir)
        results["saved_files"]["weight_classes"] = weight_class_files

        # Step 5: Upload leaderboards if requested
        if upload:
            if api_client is None or repo_id is None:
                console.print("[red]Cannot upload leaderboards: Missing API client or repository ID[/red]")
            else:
                console.print(f"[yellow]Uploading leaderboards to {repo_id}...[/yellow]")
                tracker = UploadTracker(is_test_mode=repo_id.endswith("Test"))

                upload_stats = upload_leaderboards(leaderboard_dir=leaderboard_dir, repo_id=repo_id)
                results["upload_stats"] = upload_stats

                # Update README if requested and uploads were successful
                if update_readme and upload_stats["uploaded"] > 0:
                    readme_updated = update_repository_readme(
                        leaderboard_dir=leaderboard_dir, repo_id=repo_id, force_update=force_readme_update
                    )
                    results["readme_updated"] = readme_updated

                    if readme_updated:
                        console.print("[green]Repository README updated with latest leaderboard[/green]")
                    else:
                        console.print("[yellow]Repository README not updated (not needed or failed)[/yellow]")

    except Exception as e:
        console.print(f"[red]Error generating or uploading leaderboards: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())

    # Final status report
    file_count = 0
    for files_dict in results["saved_files"].values():
        file_count += len(files_dict)

    console.print(f"[green]Generated and saved {file_count} leaderboard files[/green]")

    if upload and results.get("upload_stats") and isinstance(results["upload_stats"], dict):
        stats = results["upload_stats"]
        console.print(f"[green]Uploaded {stats.get('uploaded', 0)} files to repository[/green]")
        if stats.get("error", 0) > 0:
            console.print(f"[red]Encountered {stats.get('error', 0)} errors during upload[/red]")

    return results


def upload_results(
    scored_submissions: Set[Tuple[str, str]], scores_dir: Union[Path, str], repo_id: str, api_client: Optional[HfApi] = None
) -> Dict[str, int]:
    """Upload scored submission files to Hugging Face.

    This function handles uploading user score files for all successfully
    scored submissions to the specified repository.

    Args:
        scored_submissions: Set of (username, submission_id) tuples to upload
        scores_dir: Directory containing score files
        repo_id: Repository ID to upload to
        api_client: Authenticated HF API client

    Returns:
        Dictionary with upload statistics
    """
    scores_dir = Path(scores_dir)
    results = {"files": 0, "errors": 0, "skipped": 0}

    if not api_client or not scored_submissions:
        console.print("[yellow]Skipping uploads: No API client or no submissions to upload[/yellow]")
        return results

    console.print(f"[yellow]Uploading user score files for {len(scored_submissions)} submissions...[/yellow]")

    for username, submission_id in scored_submissions:
        try:
            upload_stats = upload_user_files(username=username, submission_id=submission_id, base_dir=scores_dir, repo_id=repo_id)

            # Update results
            for key in ["uploaded", "skipped", "error"]:
                if key in upload_stats:
                    results_key = "files" if key == "uploaded" else key if key == "skipped" else "errors"
                    results[results_key] += upload_stats[key]

        except Exception as e:
            console.print(f"[red]Error uploading files for {username}/{submission_id}: {e}[/red]")
            results["errors"] += 1

    # Summary
    console.print(
        f"[green]Upload results: {results['files']} files uploaded, {results['skipped']} skipped, {results['errors']} errors[/green]"
    )

    return results


@app.command()
def score(
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
    """Run the scoring process using a refactored approach with better separation of concerns.

    This is a refactored version of the original run_scoring function, split into
    smaller, more focused functions for improved maintainability.

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
        config: Path to configuration file for setting options
    """
    try:
        # Step 1: Set up environment
        env_config, model_dirs, model_names = setup_environment(
            mode=mode, sub_test=sub_test, score_test=score_test, model_dir=model_dir, model_dirs=model_dirs
        )

        # Step 2: Set up API client for uploads
        api_client = setup_api_client(upload=upload)

        # Set final upload flag based on API client availability
        can_upload = upload and api_client is not None

        # Set up input and output directories
        submission_dir = submissions_dir or env_config.submission_dir
        scores_dir_path = scores_dir or env_config.scores_dir

        # Step 3: Download submissions and scores
        download_result = download_submissions_and_scores(
            env_config=env_config, submission_dir=submission_dir, scores_dir=scores_dir_path, api_client=api_client if can_upload else None
        )

        # Step 4: Find unscored submissions
        unscored_files_by_model = find_unscored_submissions(
            submission_dir=submission_dir, scores_dir=scores_dir_path, model_names=model_names
        )

        # Check if there's anything to score
        total_unscored = sum(len(files) for files in unscored_files_by_model.values())
        if total_unscored == 0:
            console.print("[yellow]No submissions to score. Exiting.[/yellow]")
            return

        # Step 5: Score submissions
        successfully_scored_submissions = score_submissions(
            unscored_files_by_model=unscored_files_by_model,
            model_dirs=model_dirs,
            model_names=model_names,
            scores_dir=scores_dir_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            cache_size=cache_size,
            prompt_file=prompt_file,
            reasoning_template=reasoning_template,
            log_prompts=log_prompts,
            test_samples=test_samples,
            draft_model_dir=draft_model_dir,
            draft_cache_size=draft_cache_size,
        )

        # Stop here if no submissions were successfully scored
        if not successfully_scored_submissions:
            console.print("[red]No submissions were successfully scored. Skipping summary generation and uploads.[/red]")
            return

        # Step 6: Generate summaries and update history for scored submissions
        summary_results = generate_summaries_and_history(
            scored_submissions=successfully_scored_submissions,
            scores_dir=scores_dir_path,
            overwrite=True,
            repo_id=env_config.scores_repo_id if can_upload else None,
        )

        # Step 7: Generate and optionally upload leaderboards
        leaderboard_results = generate_and_upload_leaderboards(
            scores_dir=scores_dir_path,
            leaderboard_dir="leaderboards",
            upload=can_upload,
            api_client=api_client,
            repo_id=env_config.scores_repo_id,
            update_readme=can_upload,
            force_readme_update=False,
        )

        # Step 8: Upload user score files if requested
        upload_stats = {}
        if can_upload and successfully_scored_submissions:
            upload_stats = upload_results(
                scored_submissions=successfully_scored_submissions,
                scores_dir=scores_dir_path,
                repo_id=env_config.scores_repo_id,
                api_client=api_client,
            )

        # Print summary of results
        console.print("\n[bold blue]===== Scoring Process Complete =====[/bold blue]")
        console.print(f"[blue]Environment: {'Production' if mode else 'Test'}[/blue]")
        console.print(f"[blue]Submissions Repository: {env_config.submissions_repo_id}[/blue]")
        console.print(f"[blue]Scores Repository: {env_config.scores_repo_id}[/blue]")
        console.print(f"[blue]Models used: {', '.join(model_names)}[/blue]")
        console.print(f"[blue]Total submissions scored: {len(successfully_scored_submissions)}[/blue]")
        console.print(f"[blue]Generated {len(summary_results['summary_files'])} summary files[/blue]")
        console.print(f"[blue]Updated {len(summary_results['history_files'])} history files[/blue]")

        if leaderboard_results.get("leaderboard") is not None:
            weight_class_count = len(leaderboard_results.get("weight_class_leaderboards", {}))
            console.print(f"[blue]Generated global leaderboard and {weight_class_count} weight class leaderboards[/blue]")

            if can_upload:
                upload_stats_lb = leaderboard_results.get("upload_stats", {})
                console.print(f"[blue]Uploaded {upload_stats_lb.get('files', 0)} leaderboard files[/blue]")
                if leaderboard_results.get("readme_updated", False):
                    console.print("[blue]Updated repository README with latest leaderboards[/blue]")

        if can_upload and upload_stats:
            console.print(f"[blue]Uploaded {upload_stats.get('files', 0)} user score files[/blue]")

        console.print("[green]Scoring process completed successfully[/green]")

    except Exception as e:
        console.print(f"[red]Error in refactored scoring process: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())


@app.command()
def cleanup(
    dir: Annotated[Optional[Path],typer.Option(help="Directory containing the directories that need to be deleted [state,downloaded_submissions,logs,etc]")] = Path("."),
    force: Annotated[bool, typer.Option("--force", "-f", help="Force deletion without countdown.")] = False,
):  # fmt: skip
    "Remove directories used for testing and evaluation"
    dirs_to_remove = ["state", "downloaded_submissions_test", "leaderboards", "downloaded_submissions", "logs", "scores_test", "scores"]

    if not force:
        console.print("[bold yellow]WARNING:[/bold yellow] This will permanently delete the following directories:")
        for dir_name in dirs_to_remove:
            if (Path(dir) / dir_name).exists():
                console.print(f"  - {Path(dir) / dir_name}")
        console.print("[yellow]Starting deletion in 5 seconds. Press Ctrl+C to cancel.[/yellow]")
        for i in range(5, 0, -1):
            console.print(f"[yellow]Deleting in {i}...[/yellow]", end="\r")
            time.sleep(1)

    console.print(f"[blue]Cleaning up directories in: {dir.resolve()}[/blue]")
    for dir_name in dirs_to_remove:
        path = Path(dir) / dir_name  # Construct path relative to the optional 'dir' argument
        if path.exists():
            if path.is_dir():
                try:
                    shutil.rmtree(path)
                    console.print(f"[green]Removed {path}[/green]")
                except OSError as e:
                    console.print(f"[red]Error removing directory {path}: {e}[/red]")
            else:
                console.print(f"[red]Not a directory - {path}[/red]")
        else:
            console.print(f"[yellow]Skipping {path} - not found[/yellow]")


if __name__ == "__main__":
    app()
