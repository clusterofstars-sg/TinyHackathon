import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, Optional, Tuple, Any, Union

import pandas as pd
import typer
from huggingface_hub import HfApi, login
from rich.console import Console
from datasets import load_dataset

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
        return pd.DataFrame({"completion": open(file_path).read().splitlines()})
    elif file_path.suffix in [".parquet", ".pq"]:
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def get_hf_user() -> Tuple[str, HfApi]:
    "Get HF username from the API."
    token = os.environ.get("HF_TOKEN")
    assert token, "Please set the HF_TOKEN environment variable"
    login(token=token)
    api = HfApi()
    user_info = api.whoami()
    return user_info["name"], api


def upload_submission(file_path: Union[str, Path], dataset_id: str = "cluster-of-stars/TinyStoriesHackathon") -> Dict[str, Any]:
    "Upload a submission to the HF dataset using environment credentials."
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
    "Submit a file to the TinyStories hackathon."
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
    "Check your Hugging Face identity."
    try:
        username, _ = get_hf_user()
        console.print(f"Logged in as: [green]{username}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

def download_tinystories_dataset(split: str ="validation", output_dir: str ="tinystories_data"):
    "Download Tiny Stories dataset using hugginface datasets"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir/f"{split}.parquet"
    if output_file.exists():
        console.print(f"[green]Dataset already downloaded to {output_file}[/green]")
        return output_file
    console.print(f"[yellow]Downloading TinyStories {split} split...[/yellow]")
    dataset = load_dataset("roneneldan/TinyStories", split=split)
    df = dataset.to_pandas()
    df.to_parquet(output_file, index=False)
    console.print(f"[green]Dataset saved to {output_file}[/green]")
    return output_file

@app.command()
def download_dataset(split: Annotated[str, typer.Option(help="Split to download, such as train or validation")] = "validation",
                     output_dir: Annotated[str, typer.Option(help="Directory to output data")] = "tinystories_data"):
    "Download Tiny Stories dataset from huggingface"
    try:
        output_file = download_tinystories_dataset(split, output_dir)
        console.print(f"[green]Successfully downloaded TinyStories {split} split to {output_file}[/green]")
        df = pd.read_parquet(output_file)
        console.print(f"[blue]Dataset has {len(df)} rows and columns: {', '.join(df.columns)}[/blue]")

        console.print("[yellow]Sample story:[/yellow]")
        sample = df.sample(1).iloc[0]
        console.print(sample["text"])
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

if __name__ == "__main__":
    app()
