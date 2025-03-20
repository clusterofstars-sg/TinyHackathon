import pandas as pd, numpy as np
from huggingface_hub import HfApi, login
from pathlib import Path
from fastcore.utils import *
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import typer
from rich.console import Console
from rich.table import Table
import os
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
import json
import re
import traceback

app = typer.Typer()
console = Console()
def hf_login(token=None):
    "Login to Hugging Face using token from env or param"
    if token is None: token = os.environ.get('HF_TOKEN')
    assert token, "Please provide a Hugging Face token via param or HF_TOKEN env variable"
    login(token=token)
    return HfApi()
def read_submission(file_path):
    "Read a submission file and return as a DataFrame"
    file_path = Path(file_path)
    if file_path.suffix == '.csv': return pd.read_csv(file_path)
    elif file_path.suffix == '.txt': return pd.DataFrame({"completion": open(file_path).read().splitlines()})
    elif file_path.suffix in ['.parquet', '.pq']: return pd.read_parquet(file_path)
    else: raise ValueError(f"Unsupported file format: {file_path.suffix}")
def get_hf_user():
    "Get HF username from the API"
    token = os.environ.get('HF_TOKEN')
    assert token, "Please set the HF_TOKEN environment variable"
    login(token=token)
    api = HfApi()
    user_info = api.whoami()
    return user_info['name'], api
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
    parquet_path = participant_dir/f"{timestamp}.parquet"
    df.to_parquet(parquet_path, index=False)
    # Upload to HF
    remote_path = f"submissions/{username}/{timestamp}.parquet"
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo=remote_path,
        repo_id=dataset_id,
        repo_type="dataset"
    )
    return dict(participant=username, timestamp=timestamp, n_rows=len(df), remote_path=remote_path)
@app.command()
def submit(file_path: str):
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
    processed_file = output_dir/"processed.json"
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
        if remote_path in processed: continue
        
        # Extract username and filename
        parts = remote_path.split("/")
        if len(parts) != 3: continue  # Skip unexpected formats
        
        username = parts[1]
        filename = parts[2]
        
        # Create user directory
        user_dir = output_dir/username
        user_dir.mkdir(exist_ok=True)
        
        # Download file
        local_path = user_dir/filename
        console.print(f"Downloading [yellow]{remote_path}[/yellow] to [blue]{local_path}[/blue]")
        
        # Download to the correct path
        downloaded_path = api.hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=remote_path,
            local_dir=output_dir
        )
        
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
def download_submissions(output_dir: str = "downloaded_submissions"):
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
def load_submissions(submissions_dir):
    submissions_dir = Path(submissions_dir)
    if not submissions_dir.exists():
        raise ValueError(f"Submissions directory {submissions_dir} does not exist")
    files = list(submissions_dir.glob("**/*.parquet"))
    console.print(f"[green]Found {len(files)} submissions to evaluate[/green]")
    return files

def process_submission(pq_file,generator,scores,output_file):
    file_path = str(pq_file)
    username = pq_file.parent.name
    submission_id = pq_file.stem
        
    if username in scores and submission_id in scores[username]:
        console.print(f"[blue]Skipping already evaluated submission: {username}/{submission_id}[/blue]")
        return scores
    
    console.print(f"[yellow]Evaluating submission: {username}/{submission_id}[/yellow]")
    
    df = pd.read_parquet(pq_file)
    completions = df["completion"].tolist()
        
    if username not in scores: scores[username] = {}
    scores[username][submission_id] = {"score": 0, "details": []}
    scores = eval_completions(completions,generator,username,submission_id,scores,output_file)
    return scores

def evaluate_submissions(model_dir, output_file="scores.json", submissions_dir="downloaded_submissions"):
    "Evaluate submissions using ExLlama2"
    parquet_files = load_submissions(submissions_dir)
    console.print(f"[yellow]Loading model from {model_dir}...[/yellow]")
    config = ExLlamaV2Config(model_dir)
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len=2048, lazy=True)
    model.load_autosplit(cache)
    tokenizer = ExLlamaV2Tokenizer(config)    
    generator = ExLlamaV2DynamicGenerator(
        model=model,
        cache=cache,
        tokenizer=tokenizer
    )
    output_file = Path(output_file)
    scores = {}
    if output_file.exists(): scores = json.loads(output_file.read_text())
    
    for pq_file in parquet_files: scores=process_submission(pq_file,generator,scores,output_file)
    
    leaderboard = []
    for username, user_submissions in scores.items():
        best_score = max(sub["score"] for sub in user_submissions.values())
        leaderboard.append({"username": username, "score": best_score})
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    return scores, leaderboard

# Evaluate each completion
def eval_completions(completions,generator,username,submission_id,scores,output_file):
    for i, completion in enumerate(completions):
        prompt = f"Rate the quality of this story completion on a scale of 1-10: {completion}"        
        response = generator.generate(prompt,temperature=0.1, top_p=0.9,max_new_tokens=20)
        item_score = 5
        try:
            score_match = re.search(r'(\d+)', response)
            if score_match: item_score = min(10, max(1, int(score_match.group(1))))
        except: pass
            
        scores[username][submission_id]["details"].append({"item_id": i, "score": item_score})
        avg_score = sum(d["score"] for d in scores[username][submission_id]["details"]) / len(scores[username][submission_id]["details"])
        scores[username][submission_id]["score"] = avg_score
        output_file.write_text(json.dumps(scores, indent=2))
        if (i+1) % 10 == 0:
            console.print(f"Progress: {i+1}/{len(completions)} items evaluated. Current avg score: {avg_score:.2f}")
    return scores

@app.command()
def evaluate(
    model_dir: str, 
    output_file: str = "scores.json", 
    submissions_dir: str = "downloaded_submissions",
):
    "Evaluate submissions using ExLlama2"
    try:
        scores, leaderboard = evaluate_submissions(model_dir, output_file, submissions_dir)
        
        # Display leaderboard
        console.print("[green]Evaluation complete! Leaderboard:[/green]")
        
        table = Table(show_header=True)
        table.add_column("Rank")
        table.add_column("Username")
        table.add_column("Score")
        
        for i, entry in enumerate(leaderboard[:10]):
            table.add_row(str(i+1), entry["username"], f"{entry['score']:.2f}")
        
        console.print(table)
        console.print(f"[blue]Full results saved to {output_file}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        console.print(traceback.format_exc())


if __name__ == "__main__":
    app()
