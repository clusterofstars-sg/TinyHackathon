import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from huggingface_hub import HfApi, login
from rich.console import Console
from rich.table import Table

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)
console = Console()


def hf_login(token=None):
    "Login to Hugging Face using token from env or param"
    if token is None:
        token = os.environ.get("HF_TOKEN")
    assert token, "Please provide a Hugging Face token via param or HF_TOKEN env variable"
    login(token=token)
    return HfApi()


def read_submission(file_path):
    "Read a submission file and return as a DataFrame"
    file_path = Path(file_path)
    if file_path.suffix == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix == ".txt":
        return pd.DataFrame({"completion": open(file_path).read().splitlines()})
    elif file_path.suffix in [".parquet", ".pq"]:
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def get_hf_user():
    "Get HF username from the API"
    token = os.environ.get("HF_TOKEN")
    assert token, "Please set the HF_TOKEN environment variable"
    login(token=token)
    api = HfApi()
    user_info = api.whoami()
    return user_info["name"], api


def upload_submission(file_path, dataset_id="cluster-of-stars/TinyStoriesHackathon"):
    "Upload a submission to the HF dataset using environment credentials"
    username, api = get_hf_user()
    # Read submission and convert to parquet
    df = read_submission(file_path)
    # Generate timestamp for the submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create participant directory if it doesn't exist
    participant_dir = Path(f"submissions/{username}")
    participant_dir.mkdir(exist_ok=True, parents=True)
    # Save as parquet with timestamp
    parquet_path = participant_dir / f"{timestamp}.parquet"
    df.to_parquet(parquet_path, index=False)
    # Upload to HF
    remote_path = f"submissions/{username}/{timestamp}.parquet"
    api.upload_file(path_or_fileobj=str(parquet_path), path_in_repo=remote_path, repo_id=dataset_id, repo_type="dataset")
    return dict(participant=username, timestamp=timestamp, n_rows=len(df), remote_path=remote_path)


@app.command()
def submit(file_path: Annotated[str, typer.Argument(help="Path to the submission file")]):
    "Submit a file to the TinyStories hackathon"
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            console.print(f"[red]Error: File {file_path} not found[/red]")
            return

        console.print(f"[yellow]Uploading submission from {file_path}...[/yellow]")
        result = upload_submission(file_path)
        console.print("[green]Submission successful![/green]")
        console.print(f"Username: [blue]{result['participant']}[/blue]")
        console.print(f"Timestamp: [blue]{result['timestamp']}[/blue]")
        console.print(f"Rows submitted: [blue]{result['n_rows']}[/blue]")
        console.print(f"Stored at: [blue]{result['remote_path']}[/blue]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@app.command()
def whoami():
    "Check your Hugging Face identity"
    try:
        username, _ = get_hf_user()
        console.print(f"Logged in as: [green]{username}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


def download_new_submissions(dataset_id="cluster-of-stars/TinyStoriesHackathon", output_dir="downloaded_submissions"):
    "Download all new submissions from HF dataset"
    # Get HF API
    _, api = get_hf_user()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create tracking file for processed submissions
    processed_file = output_dir / "processed.json"
    if processed_file.exists():
        processed = set(json.loads(processed_file.read_text()))
    else:
        processed = set()

    # Get all files in the dataset that match our pattern
    files = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")
    submission_files = [f for f in files if f.startswith("submissions/") and f.endswith(".parquet")]

    # Download new submissions
    new_files = []
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


@app.command()
def download_submissions(
    output_dir: Annotated[str, typer.Option(help="Directory to store downloaded submissions")] = "downloaded_submissions",
):
    "Download all new submissions from the TinyStories hackathon"
    try:
        console.print("[yellow]Downloading new submissions...[/yellow]")
        new_files = download_new_submissions(output_dir=output_dir)

        if not new_files:
            console.print("[green]No new submissions found.[/green]")
            return

        console.print(f"[green]Downloaded {len(new_files)} new submissions:[/green]")

        table = Table(show_header=True)
        table.add_column("Username")
        table.add_column("Filename")
        table.add_column("Local Path")

        for file in new_files:
            table.add_row(file["username"], file["filename"], str(file["local_path"]))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


if __name__ == "__main__":
    app()
