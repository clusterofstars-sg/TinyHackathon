from typing import Any, Set, Dict, List, Optional, Tuple, Union
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from datasets import load_dataset, Dataset, DatasetDict
import re
import string
import pandas as pd
from functools import partial

# Import from scoring only what we need
from tinyhackathon.scoring import extract_scores, extract_submission_datetime, process_scores

console = Console()


def remove_leading_symbol(text: str):
    # Remove leading whitespace and symbols, these tend to be ill formatted
    return re.sub(
        r"^[-:\s]*[^-:\s]", lambda o: o[0][-1], text
    )  # Remove difference between regular and instruct for comparison(removes first - or :)


def standard_2_new_lines(text: str):
    # Normalize new lines, due to instruct dataset missing newlines 1 v 2
    return re.sub(r"\n+", "\n\n", text)


def maps(*fs, col="text"):
    # Stacks multiple functions into a single function for use in datasets map.
    def maps_f(e):
        e = e[col]
        for f in fs:
            e = map(f, e)
        return {col: list(e)}

    return maps_f


def combine_stories(dataset, sep="\n\n"):
    # Combine rows into full stories based on the start and end markers
    full_stories = []
    current_story = None

    full_instructs = []
    current_instructs = []

    for row in dataset:
        # If the row starts a story, start collecting sentences
        if row.startswith("Story:"):
            current_story = []  # Reset for the next story
        elif row == "<|endoftext|>":
            # If we encounter the end marker, finish the current story.
            if current_story:
                full_stories.append(sep.join(current_story))
                full_instructs += ["\n".join(current_instructs)]
            current_instructs = []
            current_story = None  # We are in instructs area next
        else:
            # Otherwise, continue collecting sentences/instructions
            if current_story is not None:
                current_story.append(row)
            else:
                current_instructs += [row]

    # In case the last story doesn't have a proper end marker
    if current_story:
        full_stories.append(sep.join(current_story))
        full_instructs += ["\n".join(current_instructs)]

    return full_stories, full_instructs


def fix_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    if row["text"].find("Summary:") == -1:
        return {}
    story, summary = row["text"].split("Summary:")
    story.rstrip()
    summary = "\nSummary:" + summary
    return {"text": story, "instructions": row["instructions"] + summary}


def determine_overlap(e: str, overlap: Set[str]) -> bool:
    e = standard_2_new_lines(remove_leading_symbol(e)).rstrip()
    return e in overlap


def calculate_overlap(combined_dict, tiny_stories):
    final_instruct_stories = {"train": combined_dict["train"]["text"]}
    final_instruct_stories["validation"] = combined_dict["validation"]["text"]
    full_stories_instruct = final_instruct_stories["train"]
    full_stories_instruct += final_instruct_stories["validation"]
    # Leading symbols differ between pretraining and instruct ds
    full_stories_instruct = [remove_leading_symbol(o) for o in full_stories_instruct]
    for i, instruction in enumerate(full_stories_instruct):
        instruction = remove_leading_symbol(instruction)
        instruction = standard_2_new_lines(instruction)
        instruction = instruction.rstrip()
        full_stories_instruct[i] = instruction
    tiny_stories_set = set(tiny_stories["train"]["text"] + tiny_stories["validation"]["text"])
    tiny_stories_instruct_set = set(full_stories_instruct)
    return tiny_stories_set.intersection(tiny_stories_instruct_set)


def to_datasets(full_train, full_validation):
    full_train_stories, full_train_instructs = full_train
    full_validation_stories, full_validation_instructs = full_validation
    train_df = pd.DataFrame({"text": full_train_stories, "instructions": full_train_instructs})
    validation_df = pd.DataFrame({"text": full_validation_stories, "instructions": full_validation_instructs})
    train_ds = Dataset.from_pandas(train_df)
    validation_ds = Dataset.from_pandas(validation_df)
    return DatasetDict({"train": train_ds, "validation": validation_ds})


def generate_instructs():
    console.print(f"[green]Downlaoding datasets...[/green]")
    tiny_stories = load_dataset("roneneldan/TinyStories")
    tiny_stories_instruct = load_dataset("roneneldan/TinyStoriesInstruct")
    console.print(f"[green]Format Pretraining Dataset for comparison...[/green]")
    tiny_stories = tiny_stories.map(maps(lambda t: t.rstrip(), standard_2_new_lines, remove_leading_symbol), batched=True)
    console.print(f"[green]Parsing training data...[/green]")
    full_train = combine_stories(tiny_stories_instruct["train"]["text"])
    console.print(f"[green]Parsing validation data...[/green]")
    full_validation = combine_stories(tiny_stories_instruct["validation"]["text"])
    console.print(f"[green]Formatting Datasets...[/green]")
    combined_dict = to_datasets(full_train, full_validation)
    console.print(f"[green]Cleaning Instruct Dataset...[/green]")
    combined_dict = combined_dict.map(maps(lambda t: t.rstrip(), remove_leading_symbol), batched=True)
    combined_dict = combined_dict.map(fix_summary)
    console.print(f"[green]Calculating overlap...[/green]")
    overlap = calculate_overlap(combined_dict, tiny_stories)
    f_overlap = partial(determine_overlap, overlap=overlap)
    combined_dict = combined_dict.map(lambda xs: {"overlaps": [f_overlap(o) for o in xs["text"]]}, batched=True)
    console.print(f"[green]Filtering duplicates...[/green]")
    combined_dict = combined_dict.filter(lambda x: x["text"] not in ["", "The end."])
    return combined_dict
