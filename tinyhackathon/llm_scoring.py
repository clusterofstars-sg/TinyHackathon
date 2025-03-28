import datetime
import json
import re
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Union

import pandas as pd
import torch
import transformers
import typer
import yaml
from datasets import Dataset, load_dataset
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
from hf_upload import get_hf_user
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)
console = Console()


class ScoreCategory(str, Enum):
    GRAMMAR = "GRAMMAR"
    CREATIVITY = "CREATIVITY"
    CONSISTENCY = "CONSISTENCY"
    PLOT = "PLOT"
    OVERALL = "OVERALL"


def load_submissions(submissions_dir: Union[str, Path]) -> List[Path]:
    "Load submission files from the specified directory."
    submissions_dir = Path(submissions_dir)
    if not submissions_dir.exists():
        raise ValueError(f"Submissions directory {submissions_dir} does not exist")
    files = list(submissions_dir.glob("**/*.csv"))
    console.print(f"[green]Found {len(files)} submissions to evaluate[/green]")
    return files


def create_evaluation_prompt(story_start: str, completion: str, prompt_file: Union[str, Path] = "prompts/simple_prompt.yaml"):
    "Create a structured prompt for evaluating story completions from a YAML file"
    prompt_file = Path(prompt_file) if not isinstance(prompt_file, Path) else prompt_file

    if not prompt_file.exists():
        console.print(f"[red]Prompt file {prompt_file} not found, using default prompts[/red]")
        typer.Exit(1)
    else:
        # Load prompts from YAML file
        try:
            with open(prompt_file, "r") as f:
                prompts = yaml.safe_load(f)

            system_prompt = prompts.get("system_prompt", "")
            user_prompt = prompts.get("user_prompt", "")
        except Exception as e:
            console.print(f"[red]Error loading prompts from {prompt_file}: {str(e)}[/red]")
            raise

    # Format the user prompt with provided values
    user_prompt = user_prompt.format(story_start=story_start, completion=completion)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def process_submission(
    submission_file: Path,
    generator: ExLlamaV2DynamicGenerator,
    scores: Dict[str, Any],
    output_file: Path,
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_new_tokens: int = 20,
    sample: Optional[int] = None,
    log_prompts: bool = False,
    prompt_file: Union[str, Path] = "prompts.yaml",
):
    "Process a single submission file and evaluate its completions."
    username = submission_file.parent.name
    submission_id = submission_file.stem

    if username in scores and submission_id in scores[username]:
        console.print(f"[blue]Skipping already evaluated submission: {username}/{submission_id}[/blue]")
        return scores

    console.print(f"[yellow]Evaluating submission: {username}/{submission_id}[/yellow]")

    try:
        df = pd.read_csv(submission_file)
        prompts = df["prompt"].tolist()
        completions = df["completion"].tolist()

        if username not in scores:
            scores[username] = {}

        if submission_id not in scores[username]:
            scores[username][submission_id] = {"score": 0, "details": []}

        # Create log file path if logging is enabled
        log_file = None
        if log_prompts:
            log_dir = Path("logs") / username
            log_dir.mkdir(exist_ok=True, parents=True)
            log_file = log_dir / f"{submission_id}.json"
            console.print(f"[yellow]Will log prompts and responses to {log_file}[/yellow]")

        scores = eval_completions(prompts, completions, generator, username, submission_id, scores, output_file, temperature, top_p, max_new_tokens, sample=sample, log_file=log_file, prompt_file=prompt_file)  # fmt: skip
        console.print(f"[green]Completed evaluation for {username}/{submission_id} with score: {scores[username][submission_id]['score']:.2f}[/green]")  # fmt: skip

    except Exception as e:
        console.print(f"[red]Error processing {username}/{submission_id}: {str(e)}[/red]")
        traceback.print_exc()

    return scores


def evaluate_submissions(
    model_dir: str,
    output_file: str = "scores.json",
    submissions_dir: str = "downloaded_submissions",
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_new_tokens: int = 20,
    batch_size: int = 128,
    cache_size: int = 1024 * 50,
    sample: Optional[int] = None,
    log_prompts: bool = False,
    prompt_file: Union[str, Path] = "prompts/simple_prompt.yaml",
):
    "Evaluate all submissions using ExLlama2."
    submission_files = load_submissions(submissions_dir)
    console.print(f"[yellow]Loading model from {model_dir}...[/yellow]")
    config = ExLlamaV2Config(model_dir)
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len=cache_size, lazy=True)
    model.load_autosplit(cache)
    tokenizer = ExLlamaV2Tokenizer(config)
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    tokenizer.apply_chat_template = hf_tokenizer.apply_chat_template
    generator = ExLlamaV2DynamicGenerator(model=model, cache=cache, tokenizer=tokenizer, max_batch_size=batch_size)
    generator.warmup()
    output_file = Path(output_file)
    scores: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if output_file.exists():
        scores = json.loads(output_file.read_text())

    for submission_file in submission_files:
        scores = process_submission(
            submission_file,
            generator,
            scores,
            output_file,
            temperature,
            top_p,
            max_new_tokens,
            sample=sample,
            log_prompts=log_prompts,
            prompt_file=prompt_file,
        )

    leaderboard = []
    for username, user_submissions in scores.items():
        best_score = max(sub["score"] for sub in user_submissions.values())
        leaderboard.append({"username": username, "score": best_score})
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    return scores, leaderboard


def extract_scores(response):
    "Extract scores from the model's response"
    scores = {}
    # Extract individual category scores
    for category in ScoreCategory:
        pattern = f"{category.value}: (\\d+(?:\\.\\d+)?)"
        # Find all matches and use the last one if multiple exist
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            scores[category.lower()] = float(matches[-1])
    return scores


def eval_completions(
    prompts: List[str],
    completions: List[str],
    generator: ExLlamaV2DynamicGenerator,
    username: str,
    submission_id: str,
    scores: Dict[str, Dict[str, Dict[str, Any]]],
    output_file: Path,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_new_tokens: int = 20,
    sample: Optional[int] = None,
    log_file: Optional[Path] = None,
    prompt_file: Union[str, Path] = "prompts/simple_prompt.yaml",
):
    "Evaluate each completion in a submission using batch processing."
    try:
        # Create sampler settings
        gen_settings = ExLlamaV2Sampler.Settings(temperature=temperature, top_p=top_p, token_repetition_penalty=1.0, top_k=0)
        # Queue all jobs
        console.print(f"[yellow]Queueing {len(completions)} evaluation jobs...[/yellow]")

        # Reset generator queue if needed
        if generator.num_remaining_jobs() > 0:
            console.print("[yellow]Clearing existing generator queue...[/yellow]")
            generator.clear_queue()

        # Queue all evaluation jobs
        responses = {}
        metadata = {}
        prompts_log = {}

        for i, (beginning, completion) in enumerate(zip(prompts, completions)):
            raw_prompt = create_evaluation_prompt(story_start=beginning, completion=completion, prompt_file=prompt_file)
            templated_prompt = generator.tokenizer.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=False)

            # Store prompt for logging
            prompts_log[i] = templated_prompt

            prompt_ids = generator.tokenizer.encode(templated_prompt, encode_special_tokens=True)
            job = ExLlamaV2DynamicJob(
                input_ids=prompt_ids,
                gen_settings=gen_settings,
                max_new_tokens=max_new_tokens,
                stop_conditions=[generator.tokenizer.eos_token_id],
                identifier=i,
            )
            generator.enqueue(job)
            responses[i] = ""  # Initialize empty response

        # Process jobs and collect results
        console.print("[yellow]Processing evaluation jobs...[/yellow]")

        time_begin = time.time()
        num_completions = 0
        num_tokens = 0
        total_jobs = generator.num_remaining_jobs()

        # Generate all completions with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # Create the main progress task
            task = progress.add_task(f"[green]Evaluating {username}/{submission_id}", total=total_jobs)

            while generator.num_remaining_jobs():
                try:
                    results = generator.iterate()

                    # Track tokens processed
                    bsz = len(set([r["identifier"] for r in results]))
                    num_tokens += bsz

                    for result in results:
                        idx = result["identifier"]

                        if not result["eos"]:
                            continue

                        # For EOS results, get the full completion
                        responses[idx] = result["full_completion"]
                        metadata[idx] = result
                        num_completions += 1

                        # Update progress
                        progress.update(task, completed=num_completions)

                except Exception as e:
                    console.print(f"[red]Error in generator iteration: {str(e)}[/red]")
                    traceback.print_exc()
                    continue

        # Output statistics
        elapsed_time = time.time() - time_begin
        rpm = num_completions / (elapsed_time / 60) if elapsed_time > 0 else 0
        tps = num_tokens / elapsed_time if elapsed_time > 0 else 0
        console.print()
        console.print(f"[blue]Avg. completions/minute: {rpm:.2f}[/blue]")
        console.print(f"[blue]Avg. output tokens/second: {tps:.2f}[/blue]")
        console.print()

        # Process all responses and update scores
        console.print("[yellow]Processing responses and calculating scores...[/yellow]")

        # Initialize details list if needed
        if "details" not in scores[username][submission_id]:
            scores[username][submission_id]["details"] = []

        total_score = 0
        processed_count = 0

        # Log prompts and responses if log_file is provided
        if log_file:
            log_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "username": username,
                "submission_id": submission_id,
                "evaluations": [],
            }

        # Process each response
        for idx, response in responses.items():
            # Extract score
            try:
                item_scores = extract_scores(response)
                for key, value in list(item_scores.items()):
                    item_scores[key] = min(10, max(1, value))
            except Exception as e:
                console.print(f"[red]Error extracting score from '{response}': {str(e)}[/red]")

            # Store the score
            details_item = {"item_id": idx, "scores": item_scores}
            scores[username][submission_id]["details"].append(details_item)
            total_score += item_scores.get("overall", 0)
            processed_count += 1

            # Add to log if enabled
            if log_file:
                log_data["evaluations"].append(
                    {
                        "item_id": idx,
                        "prompt": prompts_log.get(idx, ""),
                        "response": response,
                        "scores": item_scores,
                        "metadata": {
                            k: v.item() if isinstance(v, torch.Tensor) else v
                            for k, v in metadata.get(idx, {}).items()
                            if k not in ["full_completion", "job", "held"]
                        },
                    }
                )

        # Save logs if enabled
        if log_file and processed_count > 0:
            # Ensure log directory exists
            log_file.parent.mkdir(exist_ok=True, parents=True)

            # Append to existing log if it exists
            existing_logs = []
            if log_file.exists():
                try:
                    existing_logs = json.loads(log_file.read_text())
                    if not isinstance(existing_logs, list):
                        existing_logs = [existing_logs]
                except:
                    existing_logs = []

            existing_logs.append(log_data)
            log_file.write_text(json.dumps(existing_logs, indent=2))
            console.print(f"[green]Saved prompt and response logs to {log_file}[/green]")

        # Final update
        if processed_count > 0:
            avg_score = total_score / processed_count
            scores[username][submission_id]["score"] = avg_score
            output_file.write_text(json.dumps(scores, indent=2))
            console.print(f"[green]Completed processing {processed_count} responses with final avg score: {avg_score:.2f}[/green]")

    except Exception as e:
        console.print(f"[red]Error in eval_completions: {str(e)}[/red]")
        traceback.print_exc()

    return scores


@app.command()
def evaluate(
    model_dir: Annotated[str, typer.Argument(help="Directory containing the ExLlama2 model files")],
    output_file: Annotated[str, typer.Option(help="Path to save scores JSON")] = "scores.json",
    submissions_dir: Annotated[str, typer.Option(help="Directory containing submission files")] = "downloaded_submissions",
    temperature: Annotated[float, typer.Option(help="Temperature for generation sampling")] = 1.0,
    top_p: Annotated[float, typer.Option(help="Top-p (nucleus) sampling value")] = 0.9,
    max_new_tokens: Annotated[int, typer.Option(help="Maximum number of tokens to generate")] = 20,
    batch_size: Annotated[int, typer.Option(help="Maximum batch size for inference")] = 128,
    cache_size: Annotated[int, typer.Option(help="Cache size in tokens (multiply by 4 for bytes)")] = 2048,
    sample: Annotated[int, typer.Option(help="Sample the first N completions and test data")] = None,
    log_prompts: Annotated[bool, typer.Option(help="Enable prompt and response logging")] = False,
    prompt_file: Annotated[str, typer.Option(help="Path to YAML file with evaluation prompts")] = "prompts/simple_prompt.yaml",
):
    "Evaluate submissions using ExLlama2 and display a leaderboard."
    try:
        console.print(f"[yellow]Starting evaluation with batch_size={batch_size}, cache_size={cache_size}[/yellow]")
        scores, leaderboard = evaluate_submissions(
            model_dir=model_dir,
            output_file=output_file,
            submissions_dir=submissions_dir,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            cache_size=cache_size,
            sample=sample,
            log_prompts=log_prompts,
            prompt_file=prompt_file,
        )

        # Display leaderboard
        console.print("[green]Evaluation complete! Leaderboard:[/green]")

        table = Table(show_header=True)
        table.add_column("Rank")
        table.add_column("Username")
        table.add_column("Score")

        for i, entry in enumerate(leaderboard[:10]):
            table.add_row(str(i + 1), entry["username"], f"{entry['score']:.2f}")

        console.print(table)
        console.print(f"[blue]Full results saved to {output_file}[/blue]")
        if log_prompts:
            console.print("[blue]Prompt and response logs saved to logs/[username]/[submission_id].json[/blue]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        console.print(traceback.format_exc())


def download_new_submissions(
    dataset_id: str = "cluster-of-stars/TinyStoriesHackathon_Submissions", output_dir: Union[str, Path] = "downloaded_submissions"
) -> List[Dict[str, Any]]:
    "Download all new submissions from HF dataset."
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
    submission_files = [f for f in files if f.startswith("submissions/") and f.endswith(".csv")]

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
    "Download all new submissions from the TinyStories hackathon."
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
