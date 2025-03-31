import csv
import datetime
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from rich.console import Console

console = Console()


class ScoreCategory(str, Enum):
    GRAMMAR = "GRAMMAR"
    CREATIVITY = "CREATIVITY"
    CONSISTENCY = "CONSISTENCY"
    PLOT = "PLOT"
    OVERALL = "OVERALL"


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

    # Check if all required categories have scores
    success = all(category.lower() in scores for category in ScoreCategory)
    return scores, success


def calc_scores(user_submissions):
    "Calculate scores for single user"
    idv_headers = ["grammar", "creativity", "consistency", "plot"]
    avg_scores = []
    avg_idv_scores = []
    avg_consistency_scores = []
    for submission in user_submissions.values():
        model_scores = [model_arch["score"] for model_arch in submission.values()]
        avg_scores += [sum(model_scores) / len(model_scores)]
        item_scores = [model_arch["details"] for model_arch in submission.values()]
        item_scores = sum(item_scores, [])  # concat arrays for all models

        def filter_by_header(scores: Dict[str, float]):
            return [scores[header] for header in idv_headers if header in scores]

        def mean(scores: Dict[str, float]):
            return sum(filter_by_header(scores)) / len(filter_by_header(scores))

        idv_avg_scores = [mean(item["scores"]) for item in item_scores]
        avg_idv_scores += [sum(idv_avg_scores) / len(idv_avg_scores)]
        consistency_scores = [model_arch["details"] for model_arch in submission.values()]
        consistency_scores = sum(consistency_scores, [])  # concat arrays for all models
        consistency_scores = [item["scores"]["consistency"] for item in consistency_scores]
        avg_consistency_scores += [sum(consistency_scores) / len(consistency_scores)]
    max_score = max(avg_scores)
    max_index = avg_scores.index(max_score)
    max_idv_score = avg_idv_scores[max_index]
    max_consistency_score = avg_consistency_scores[max_index]
    return (max_score, max_idv_score, max_consistency_score)


def write_csv(path: Path, scores: Dict[str, Dict[str, Dict[str, Any]]]):
    """Write scores to a CSV file using pandas.

    Args:
        path: Path to output CSV file
        scores: Dictionary containing score data
    """
    # Convert nested dictionary to a list of records for DataFrame
    records = []
    for username in scores:
        for submission_id in scores[username]:
            for model_arch in scores[username][submission_id]:
                model_data = scores[username][submission_id][model_arch]
                score = model_data["score"]

                # Add a record for each item detail
                if "details" in model_data:
                    for item in model_data["details"]:
                        item_id = item["item_id"]
                        # Create a record with all the basic info
                        record = {
                            "username": username,
                            "submission_id": submission_id,
                            "model_arch": model_arch,
                            "score": score,
                            "item_id": item_id,
                        }
                        # Add individual scores
                        for category, value in item["scores"].items():
                            record[category] = value
                        records.append(record)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    console.print(f"[green]Scores written to {path}[/green]")


def read_csv(path: str):
    """Read scores from a CSV file using pandas.

    Args:
        path: Path to input CSV file

    Returns:
        Dictionary containing score data
    """
    # Read CSV into DataFrame
    df = pd.read_csv(path)

    # Initialize scores dictionary
    scores = {}

    # Group by username, submission_id, and model_arch
    for (username, submission_id, model_arch), group in df.groupby(["username", "submission_id", "model_arch"]):
        # Initialize nested dictionaries if they don't exist
        if username not in scores:
            scores[username] = {}
        if submission_id not in scores[username]:
            scores[username][submission_id] = {}

        # Get the submission score (same for all rows in this group)
        submission_score = group["score"].iloc[0]

        # Initialize model arch data
        scores[username][submission_id][model_arch] = {"score": submission_score, "details": []}

        # Add details for each item
        for _, row in group.iterrows():
            # Extract score categories and values
            score_categories = ["grammar", "creativity", "consistency", "plot", "overall"]
            item_scores = {}
            for category in score_categories:
                if category in row and not pd.isna(row[category]):
                    item_scores[category] = float(row[category])

            # Add item details
            detail = {"item_id": int(row["item_id"]), "scores": item_scores}
            scores[username][submission_id][model_arch]["details"].append(detail)

    return scores


def process_scores(
    responses: Dict[int, str],
    followup_responses: Dict[int, str],
    username: str,
    submission_id: str,
    model_arch: str,
    scores: Dict[str, Dict[str, Dict[str, Any]]],
):
    """Process model responses and extract scores.

    Args:
        responses: Dictionary mapping item IDs to model responses
        followup_responses: Dictionary mapping item IDs to follow-up responses
        username: Username of the submission
        submission_id: ID of the submission
        model_arch: Model architecture name
        scores: Dictionary to update with extracted scores

    Returns:
        Updated scores dictionary, item_scores dictionary, total_score, and processed_count
    """
    item_scores = {}
    total_score = 0
    processed_count = 0

    # Initialize details list if needed
    if "details" not in scores[username][submission_id][model_arch]:
        scores[username][submission_id][model_arch]["details"] = []

    # Process each response
    for idx, response in responses.items():
        # Get follow-up response if it exists
        followup_response = followup_responses.get(idx)

        # Try to extract scores from initial response
        extracted_scores, success = extract_scores(response)

        # If unsuccessful and a follow-up response exists, try that one
        if not success and followup_response:
            extracted_scores, success = extract_scores(followup_response)

        # If we have scores, normalize them and store
        if success or extracted_scores:
            # Normalize scores to be between 1 and 10
            for key, value in list(extracted_scores.items()):
                extracted_scores[key] = min(10, max(1, value))

            # Store scores
            item_scores[idx] = extracted_scores

            # Update total score
            total_score += extracted_scores.get("overall", 0)
            processed_count += 1

            # Add to details
            details_item = {"item_id": idx, "scores": extracted_scores}
            scores[username][submission_id][model_arch]["details"].append(details_item)

    # Update average score if we processed any items
    if processed_count > 0:
        avg_score = total_score / processed_count
        scores[username][submission_id][model_arch]["score"] = avg_score

    return scores, item_scores, total_score, processed_count


def generate_model_dataframe(
    username: str,
    submission_id: str,
    model_name: str,
    prompts: List[str],
    completions: List[str],
    item_scores: Dict[int, Dict[str, float]],
):
    """Generate a per-model Dataframe for a submission.

    Args:
        username: Username of the submission
        submission_id: ID of the submission
        model_name: Name of the model
        prompts: List of prompts
        completions: List of completions
        item_scores: Dictionary mapping item IDs to score dictionaries

    Returns:
        DataFrame containing the scores
    """
    data = []
    for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
        if idx in item_scores:
            row = {
                "item_id": idx,
                "prompt": prompt,
                "completion": completion,
            }
            # Add all scores
            row.update(item_scores[idx])
            data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    return df
