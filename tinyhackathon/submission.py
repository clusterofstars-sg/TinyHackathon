# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.2.4",
#     "rich>=13.9.4",
#     "typer>=0.15.2",
#     "huggingface-hub>=0.29.3",
#     "datasets>=3.4.1",
#     "pandas>=2.2.3",
# ]
# ///

import json
import os
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, Tuple, Union

import pandas as pd
import typer
from huggingface_hub import HfApi, login
from rich.console import Console

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)
console = Console()


class WeightClass(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


def hf_login(token: Optional[str] = None) -> HfApi:
    "Login to Hugging Face using token from env or param."
    if token is None:
        token = os.environ.get("HF_TOKEN")
    assert token, "Please provide a Hugging Face token via param or HF_TOKEN env variable"
    login(token=token)
    return HfApi()


def read_submission(file_path: Union[str, Path]) -> pd.DataFrame:
    "Read a submission file and return as a DataFrame."
    file_path = Path(file_path)
    if file_path.suffix == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix == ".txt":
        # Create DataFrame with empty prompts and text file lines as completions
        lines = open(file_path).read().splitlines()
        return pd.DataFrame({"prompt": [""] * len(lines), "completion": lines})
    elif file_path.suffix in [".parquet", ".pq"]:
        return pd.read_parquet(file_path)
    else:
        console.print(f"[red]Error: Unsupported file format: {file_path.suffix}[/red]")
        raise typer.Exit(code=1)


def get_hf_user() -> Tuple[str, HfApi]:
    "Get HF username from the API."
    api = HfApi()
    user_info = api.whoami()
    return user_info, api


def upload_submission(
    file_path: Path,
    submission_name: Optional[str] = None,
    weight_class: Optional[WeightClass] = None,
    hf_repo: str = "cluster-of-stars/TinyStoriesHackathon_Submissions",
) -> Dict[str, Any]:
    "Upload a submission to the HF dataset using environment credentials."
    info, api = get_hf_user()
    username = info["name"]

    # Check for daily submission limit using AOE (UTC-12) timezone
    aoe_timezone = timezone(timedelta(hours=-12))
    today_aoe = datetime.now(aoe_timezone).date()

    # Check if submission already exists for the user
    try:
        repo_files = api.list_repo_files(repo_id=hf_repo, repo_type="dataset")
        user_submissions = [f for f in repo_files if f.startswith(f"submissions/{username}/")]

        if user_submissions:
            # Parse timestamps from existing submission filenames
            for submission in user_submissions:
                # Extract timestamp using regex - match YYYY(mm)dd_HHMMSS pattern
                timestamp_match = re.search(r"(\d{8}_\d{6})", submission)
                if timestamp_match:
                    ts_str = timestamp_match.group(1)
                    try:
                        # Parse timestamp in format YYYYmmdd_HHMMSS
                        ts_datetime = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                        # Convert to AOE timezone
                        ts_datetime = ts_datetime.replace(tzinfo=timezone.utc).astimezone(aoe_timezone)
                        # Check if submission was made today in AOE
                        if ts_datetime.date() == today_aoe:
                            # Calculate time until next submission is allowed (midnight AOE)
                            now_aoe = datetime.now(aoe_timezone)
                            tomorrow_aoe = datetime(now_aoe.year, now_aoe.month, now_aoe.day, tzinfo=aoe_timezone) + timedelta(days=1)
                            time_until_next = tomorrow_aoe - now_aoe
                            total_hours = time_until_next.seconds // 3600
                            minutes = (time_until_next.seconds % 3600) // 60

                            console.print("\n[red]Error: You have already submitted today (AOE timezone). Only one submission is allowed per day.[/red]")  # fmt: skip
                            console.print(f"\n[yellow]Your previous submission: {submission}[/yellow]")
                            console.print(f"[yellow]You can submit again in {total_hours} hours and {minutes} minutes (at midnight AOE).[/yellow]\n")  # fmt: skip
                            raise typer.Exit()
                    except ValueError:
                        # Skip files with invalid timestamp format
                        continue
    except typer.Exit:
        # Re-raise typer.Exit exceptions to ensure they propagate properly
        raise
    except Exception as e:
        # Don't prevent submission if we can't check existing submissions
        console.print(f"\n[yellow]Warning: Could not check for existing submissions: {str(e)}[/yellow]")

    # Read submission
    df = read_submission(file_path)

    # Validate columns - must have exactly prompt and completion columns
    expected_columns = set(["prompt", "completion"])
    if set(df.columns) != expected_columns:
        console.print(f"\n[red]Error: Submission must contain exactly two columns: 'prompt' and 'completion'. Found: {', '.join(df.columns)}[/red]")  # fmt: skip
        raise typer.Exit(code=1)

    # Validate that completions are unique
    if df["completion"].duplicated().any():
        duplicate_count = df["completion"].duplicated().sum()
        console.print(f"\n[red]Error: All completions must be unique. Found {duplicate_count} duplicate completions.[/red]")
        raise typer.Exit(code=1)

    # Check if weight_class is provided, and if not, try to get it from existing metadata or prompt
    metadata_remote_path = f"submissions/{username}/metadata.json"
    existing_metadata = None

    if weight_class is None:
        # Try to get existing weight_class from metadata.json
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                api.hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=metadata_remote_path, local_dir=tmp_dir)
                existing_metadata_path = Path(tmp_dir) / metadata_remote_path.split("/")[-1]
                if existing_metadata_path.exists():
                    with open(existing_metadata_path, "r") as f:
                        existing_metadata = json.load(f)
                        if "weight_class" in existing_metadata and existing_metadata["weight_class"]:
                            existing_weight_class = existing_metadata["weight_class"]
                            console.print(f"[yellow]Using existing weight class: {existing_weight_class}[/yellow]")
                            weight_class = WeightClass(existing_weight_class)
        except:
            # If we can't get the metadata, we'll prompt the user
            pass

        # If still no weight_class from metadata, prompt the user
        if weight_class is None:
            weight_class = typer.prompt(
                "\nSelect model weight class size (small: up to 30M, medium: up to 60M, or large: up to 120M) [small|medium|large]",
                type=WeightClass,
                show_choices=True,
            )

    # Generate timestamp for the submission
    timestamp = datetime.now()
    timestamp = timestamp.replace(tzinfo=timezone.utc).astimezone(aoe_timezone).strftime("%Y%m%d_%H%M%S")

    # Use temporary directory instead of local directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create filename
        file_name = f"{submission_name or 'submission'}_{timestamp}.csv"
        save_path = Path(temp_dir) / file_name
        # Save as CSV
        df.to_csv(save_path, index=False)

        # Upload to HF
        remote_path = f"submissions/{username}/{file_name}"
        api.upload_file(path_or_fileobj=str(save_path), path_in_repo=remote_path, repo_id=hf_repo, repo_type="dataset")

        # Handle metadata.json - get existing if we don't have it yet
        if existing_metadata is None:
            try:
                # Try to download existing metadata.json only if we haven't already
                tmp_meta_path = Path(temp_dir) / "tmp_metadata.json"
                api.hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=metadata_remote_path, local_path=str(tmp_meta_path))
                if tmp_meta_path.exists():
                    with open(tmp_meta_path, "r") as f:
                        existing_metadata = json.load(f)
            except:
                # If metadata.json doesn't exist, create new
                existing_metadata = {}

        # Determine if we need to update metadata
        metadata_exists = bool(existing_metadata)
        current_weight_class = existing_metadata.get("weight_class", None) if existing_metadata else None
        need_metadata_update = not metadata_exists or (weight_class and weight_class.value != current_weight_class)

        # Update metadata with new weight_class if needed
        metadata = existing_metadata.copy() if existing_metadata else {}
        if weight_class and need_metadata_update:
            metadata["weight_class"] = weight_class.value

        # Only upload if metadata changed or didn't exist
        if need_metadata_update:
            # Write and upload metadata.json
            metadata_path = Path(temp_dir) / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            console.print(f"\n[yellow]Updating metadata with weight class: {metadata.get('weight_class')}[/yellow]")
            api.upload_file(path_or_fileobj=str(metadata_path), path_in_repo=metadata_remote_path, repo_id=hf_repo, repo_type="dataset")

    return dict(
        participant=username,
        timestamp=timestamp,
        n_rows=len(df),
        remote_path=remote_path,
        metadata=metadata,
    )


@app.command()
def submit(
    submission_path: Annotated[Path, typer.Option(help="Path to the submission file", show_default=False)],
    submission_name: Annotated[Optional[str], typer.Option(help="Optional user friendly name of the submission ")] = None,
    weight_class: Annotated[Optional[WeightClass], typer.Option(help="Model weight class size (small: up to 30M, medium: up to 60M, or large: up to 120M). If provided this will update the current model weight class.")] = None,
    submit: Annotated[bool, typer.Option("--submit/--test",help="Upload submission (--submit) or test submission (--test). Default's to test submission.")] = False,
):  # fmt: skip
    "Submit a file to the TinyStories hackathon."
    try:
        if not submission_path.exists():
            console.print(f"\n[red]Error: File {submission_path} not found[/red]")
            return

        if not submit:
            console.print("[yellow]Test mode - submission will be uploaded to test dataset in 5 seconds...[/yellow]")
            for i in range(5, 0, -1):
                console.print(f"[yellow]    {i} seconds remaining...[/yellow]")
                time.sleep(1)

        if submit:
            hf_repo = "cluster-of-stars/TinyStoriesHackathon_Submissions"
        else:
            hf_repo = "cluster-of-stars/TinyStoriesHackathon_Submissions_Test"

        console.print(f"\n[yellow]Uploading {'test' if not submit else ''} submission from {submission_path}...[/yellow]")
        result = upload_submission(submission_path, submission_name, weight_class, hf_repo)
        console.print(f"\n[green]{'Test submission' if not submit else 'Submission'} successful![/green]")
        console.print(f"\nUsername: [blue]{result['participant']}[/blue]")
        console.print(f"Timestamp: [blue]{result['timestamp']}[/blue]")
        console.print(f"Rows submitted: [blue]{result['n_rows']}[/blue]")
        console.print(f"Stored at: [blue]{result['remote_path']}[/blue]")
        console.print(f"Weight class: [blue]{result['metadata']['weight_class']}[/blue]\n")

    except typer.Exit:
        # Just re-raise typer.Exit without additional error message
        raise
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")


@app.command()
def whoami():
    "Check your Hugging Face identity."
    try:
        info, _ = get_hf_user()
        role = info.get("auth", {}).get("accessToken", {}).get("role", None)
        if role is None:
            console.print(f"[red]Logged in as [blue]{info['name']}[/blue] without read or write access. Please login with write access to submit.[/red]")  # fmt: skip
        elif role == "read":
            console.print(f"[red]Logged in as [blue]{info['name']}[/blue] with read-only access. Need to login with write access to submit.[/red]")  # fmt: skip
        elif role == "write":
            console.print(f"[green]Logged in as [blue]{info['name']}[/blue] with write access.[/green]")
        else:
            raise ValueError(f"Unknown Hugging Face role for user: {info['name']}")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@app.command()
def download_eval():  # fmt: skip
    "Download evaluation prompts for Tiny Stories hackathon"
    api = HfApi()
    try:
        api.hf_hub_download(
            repo_id="cluster-of-stars/tiny_stories_evaluation_prompts",
            filename="evaluation_prompts.csv",
            repo_type="dataset",
            local_dir=".",
        )
        console.print("\n[green]Successfully downloaded and saved to evaluation_prompts.csv[/green]\n")
    except Exception as e:
        console.print(f"\n[red]Error: Failed to download: {str(e)}[/red]\n")


if __name__ == "__main__":
    app()
