import json
import os
import re
import traceback
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

import pandas as pd
import typer
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


def load_submissions(submissions_dir):
    submissions_dir = Path(submissions_dir)
    if not submissions_dir.exists():
        raise ValueError(f"Submissions directory {submissions_dir} does not exist")
    files = list(submissions_dir.glob("**/*.parquet"))
    console.print(f"[green]Found {len(files)} submissions to evaluate[/green]")
    return files


def process_submission(pq_file, generator, scores, output_file):
    file_path = str(pq_file)
    username = pq_file.parent.name
    submission_id = pq_file.stem

    if username in scores and submission_id in scores[username]:
        console.print(f"[blue]Skipping already evaluated submission: {username}/{submission_id}[/blue]")
        return scores

    console.print(f"[yellow]Evaluating submission: {username}/{submission_id}[/yellow]")

    df = pd.read_parquet(pq_file)
    completions = df["completion"].tolist()

    if username not in scores:
        scores[username] = {}
    scores[username][submission_id] = {"score": 0, "details": []}
    scores = eval_completions(completions, generator, username, submission_id, scores, output_file)
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
    generator = ExLlamaV2DynamicGenerator(model=model, cache=cache, tokenizer=tokenizer)
    output_file = Path(output_file)
    scores = {}
    if output_file.exists():
        scores = json.loads(output_file.read_text())

    for pq_file in parquet_files:
        scores = process_submission(pq_file, generator, scores, output_file)

    leaderboard = []
    for username, user_submissions in scores.items():
        best_score = max(sub["score"] for sub in user_submissions.values())
        leaderboard.append({"username": username, "score": best_score})
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    return scores, leaderboard


# Evaluate each completion
def eval_completions(completions, generator, username, submission_id, scores, output_file):
    for i, completion in enumerate(completions):
        prompt = f"Rate the quality of this story completion on a scale of 1-10: {completion}"
        response = generator.generate(prompt, temperature=0.1, top_p=0.9, max_new_tokens=20)
        item_score = 5
        try:
            score_match = re.search(r"(\d+)", response)
            if score_match:
                item_score = min(10, max(1, int(score_match.group(1))))
        except:
            pass

        scores[username][submission_id]["details"].append({"item_id": i, "score": item_score})
        avg_score = sum(d["score"] for d in scores[username][submission_id]["details"]) / len(scores[username][submission_id]["details"])
        scores[username][submission_id]["score"] = avg_score
        output_file.write_text(json.dumps(scores, indent=2))
        if (i + 1) % 10 == 0:
            console.print(f"Progress: {i + 1}/{len(completions)} items evaluated. Current avg score: {avg_score:.2f}")
    return scores


@app.command()
def evaluate(
    model_dir: Annotated[str, typer.Argument(help="Directory containing the ExLlama2 model files")],
    output_file: Annotated[str, typer.Option(help="Path to save scores JSON")] = "scores.json",
    submissions_dir: Annotated[str, typer.Option(help="Directory containing submission files")] = "downloaded_submissions",
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
            table.add_row(str(i + 1), entry["username"], f"{entry['score']:.2f}")

        console.print(table)
        console.print(f"[blue]Full results saved to {output_file}[/blue]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        console.print(traceback.format_exc())


if __name__ == "__main__":
    app()
