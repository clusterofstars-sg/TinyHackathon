import pandas as pd
import typer
from rich.console import Console
from typing import Annotated

# Initialize Typer app and console
app = typer.Typer()
console = Console()
def create_sample_submission(output_file: str ="sample_submission.csv", dataset_file: str ="valid.parquet"):
    "Create a sample submission from TinyStories validation data"
    dataset_file = dataset_file
    
    # Load dataset
    df = pd.read_parquet(dataset_file)
    
    # Extract last 50% of each story as "completion"
    completions = []
    for text in df["text"]:
        completions.append(text[:-len(text)//10]) #remove last 10% so not perfect
    
    # Create submission dataframe
    submission_df = pd.DataFrame({"completion": completions})
    
    # Save as CSV
    submission_df.to_csv(output_file, index=False)
    console.print(f"[green]Sample submission created with {len(submission_df)} entries at {output_file}[/green]")
    return submission_df
@app.command()
def create_submission_sample(output_file: Annotated[str, typer.Option(help="Please to put test submission")] = "sample_submission.csv",
                             dataset_file: Annotated[str, typer.Option(help="Location of test data")] = 'valid.parquet'):
    "Create a sample submission file from TinyStories test data"
    try:
        df = create_sample_submission(output_file, dataset_file)
        console.print(f"[green]Created sample submission with {len(df)} entries[/green]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

if __name__ == "__main__":
    app()
