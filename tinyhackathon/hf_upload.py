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

import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, Tuple, Union

import pandas as pd
import requests
import typer
from huggingface_hub import HfApi, login
from rich.console import Console

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)
console = Console()


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
    hf_repo: str = "cluster-of-stars/TinyStoriesHackathon_Submissions",
) -> Dict[str, Any]:
    "Upload a submission to the HF dataset using environment credentials."
    info, api = get_hf_user()
    username = info["name"]

    # Check if submission already exists for the user
    try:
        repo_files = api.list_repo_files(repo_id=hf_repo, repo_type="dataset")
        user_submissions = [f for f in repo_files if f.startswith(f"submissions/{username}/")]

        if user_submissions:
            # Check for daily submission limit using AOE (UTC-12) timezone
            aoe_timezone = timezone(timedelta(hours=-12))
            today_aoe = datetime.now(aoe_timezone).date()

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

                            console.print("[red]Error: You have already submitted today (AOE timezone). Only one submission is allowed per day.[/red]")  # fmt: skip
                            console.print(f"[yellow]Your previous submission: {submission}[/yellow]")
                            console.print(f"[yellow]You can submit again in {total_hours} hours and {minutes} minutes (at midnight AOE).[/yellow]")  # fmt: skip
                            raise typer.Exit(code=1)
                    except ValueError:
                        # Skip files with invalid timestamp format
                        continue
    except Exception as e:
        # Don't prevent submission if we can't check existing submissions
        console.print(f"[yellow]Warning: Could not check for existing submissions: {str(e)}[/yellow]")

    # Read submission
    df = read_submission(file_path)

    # Validate columns - must have exactly prompt and completion columns
    expected_columns = set(["prompt", "completion"])
    if set(df.columns) != expected_columns:
        console.print(f"[red]Error: Submission must contain exactly two columns: 'prompt' and 'completion'. Found: {', '.join(df.columns)}[/red]")  # fmt: skip
        raise typer.Exit(code=1)

    # Validate that completions are unique
    if df["completion"].duplicated().any():
        duplicate_count = df["completion"].duplicated().sum()
        console.print(f"[red]Error: All completions must be unique. Found {duplicate_count} duplicate completions.[/red]")
        raise typer.Exit(code=1)

    # Generate timestamp for the submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use temporary directory instead of local directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create filename
        file_name = f"{submission_name or 'submission'}_{timestamp}.csv"
        save_path = Path(temp_dir) / file_name
        # Save as CSV
        df.to_csv(save_path, index=False)
        # Upload to HF
        remote_path = f"submissions/{username}/{timestamp}.csv"
        api.upload_file(path_or_fileobj=str(save_path), path_in_repo=remote_path, repo_id=hf_repo, repo_type="dataset")

    return dict(participant=username, timestamp=timestamp, n_rows=len(df), remote_path=remote_path)


@app.command()
def submit(
    file_path: Annotated[Path, typer.Argument(help="Path to the submission file")],
    submission_name: Annotated[Optional[str], typer.Option(help="Name of the submission")] = None,
):
    "Submit a file to the TinyStories hackathon."
    try:
        if not file_path.exists():
            console.print(f"[red]Error: File {file_path} not found[/red]")
            return

        console.print(f"[yellow]Uploading submission from {file_path}...[/yellow]")
        result = upload_submission(file_path, submission_name)
        console.print("[green]Submission successful![/green]")
        console.print(f"Username: [blue]{result['participant']}[/blue]")
        console.print(f"Timestamp: [blue]{result['timestamp']}[/blue]")
        console.print(f"Rows submitted: [blue]{result['n_rows']}[/blue]")
        console.print(f"Stored at: [blue]{result['remote_path']}[/blue]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


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
            raise ValueError(f"Unknown Hugging Face role: {info['name']}")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@app.command()
def download_eval(
    output_file: Annotated[Path, typer.Option(help="Evaluation prompts file")] = "evaluation_prompts.csv",
    url: Annotated[str, typer.Option(help="Url for eval data")] = "https://huggingface.co/datasets/cluster-of-stars/tiny_stories_evaluation_prompts/resolve/main/evaluation_prompts.csv",
):  # fmt: skip
    "Download evaluation prompts for Tiny Stories dackathon"
    response = requests.get(url)
    if response.status_code != 200:
        console.print(f"[red]Error: Failed to download {response.status_code}[/red]")
    else:
        with open(output_file, "wb") as file:
            file.write(response.content)
            console.print(f"Successfully downloaded and saved to {output_file}")


if __name__ == "__main__":
    app()
