from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from typing import Annotated, Optional
import itertools

# Initialize Typer app and console
app = typer.Typer()
console = Console()


def create_sample_submission(output_file: str = "sample_submission.csv", dataset_file: str = "valid.parquet", num_samples: int = 1000):
    "Create a sample submission from TinyStories validation data"
    dataset_file = dataset_file

    # Load dataset
    df = read_file(dataset_file)

    if len(df) < num_samples:
        num_samples = len(df)

    # Sample num_samples from the dataset
    df = df.head(num_samples)

    # Add a small bit of text
    completions = []
    text_cycle = itertools.cycle(
        [
            " cat jumped up.",
            " bird flew high.",
            "frog jumped into water.",
            " fish swam.",
            " mouse squeeked.",
            "rabbit hopped.",
            "turtle turtled",
            " dog woofed",
        ]
    )  # so each is unique
    for text in df["prompt"]:
        small_text = next(text_cycle)
        completions.append(text + small_text)

    # Create submission dataframe
    submission_df = pd.DataFrame({"prompt": df["prompt"], "completion": completions})

    # Save as CSV
    submission_df.to_csv(output_file, index=False)
    console.print(f"[green]Sample submission created with {len(submission_df)} entries at {output_file}[/green]")
    return submission_df


def read_file(path: str):
    if path[-4:] == ".csv":
        return pd.read_csv(path)
    else:
        df = pd.read_parquet(dataset_file)
        df["completion"] = None
        df.columns = ["prompt", "completion"]
        return df


@app.command()
def create_submission_sample(
    output_file: Annotated[str, typer.Option(help="Please to put test submission")] = "sample_submission.csv",
    dataset_file: Annotated[str, typer.Option(help="Location of test data")] = "valid.parquet",
):
    "Create a sample submission file from TinyStories test data"
    try:
        df = create_sample_submission(output_file, dataset_file)
        console.print(f"[green]Created sample submission with {len(df)} entries[/green]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


if __name__ == "__main__":
    app()
