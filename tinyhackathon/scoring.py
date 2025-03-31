import csv
import datetime
import json
import re
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from rich.console import Console

console = Console()


class ScoreCategory(str, Enum):
    """Score categories used in evaluations.

    All categories are stored in uppercase but accessed as lowercase in code.
    """

    GRAMMAR = "GRAMMAR"
    CREATIVITY = "CREATIVITY"
    CONSISTENCY = "CONSISTENCY"
    PLOT = "PLOT"
    OVERALL = "OVERALL"


def extract_scores(response):
    """Extract scores from the model's response.

    Uses regex patterns to find scores for each category in the response.
    Stores them as lowercase keys (e.g., 'grammar', 'creativity').

    Args:
        response: Text response from the model

    Returns:
        Tuple of (scores dict, success boolean)
    """
    scores = {}
    # Extract individual category scores with a more flexible regex
    for category in ScoreCategory:
        # More flexible pattern that allows for punctuation and variations
        pattern = f"{category.value}\\s*[:=>]\\s*(\\d+(?:\\.\\d+)?)"
        matches = re.findall(pattern, response, re.IGNORECASE)

        if matches:
            # Use the last match if multiple exist
            scores[category.lower()] = float(matches[-1])

    # Check if all required categories have scores
    success = all(category.lower() in scores for category in ScoreCategory)
    return scores, success


def calc_scores(user_submissions):
    """Calculate best scores for a user across all their submissions.

    This function is used for the global leaderboard to find a user's best submission.
    It's different from calculate_submission_average, which handles a single submission.

    Args:
        user_submissions: Dictionary of user's submissions with scores

    Returns:
        Tuple containing (best_score, avg_individual_score, avg_consistency_score)
    """
    if not user_submissions:
        return (0.0, 0.0, 0.0)

    # Create lists to hold submission data
    submission_data = []

    # Process each submission
    for submission_id, submission in user_submissions.items():
        # Extract all model scores for this submission
        model_scores = [model_data["score"] for model_data in submission.values()]
        if not model_scores:
            continue

        # Calculate average score across all models for this submission
        avg_score = sum(model_scores) / len(model_scores)

        # Extract all item details for all models in this submission
        all_details = []
        for model_data in submission.values():
            if "details" in model_data:
                all_details.extend(model_data["details"])

        if not all_details:
            continue

        # Create a DataFrame from item details
        details_data = []
        for detail in all_details:
            if "scores" in detail:
                row = {"item_id": detail.get("item_id", 0)}
                row.update(detail["scores"])
                details_data.append(row)

        if not details_data:
            continue

        details_df = pd.DataFrame(details_data)

        # Calculate average individual score (grammar, creativity, consistency, plot)
        # Note: This excludes 'overall' which is handled separately
        idv_headers = ["grammar", "creativity", "consistency", "plot"]
        available_headers = [h for h in idv_headers if h in details_df.columns]

        if available_headers:
            avg_individual = details_df[available_headers].mean().mean()
        else:
            avg_individual = 0.0

        # Calculate average consistency score if available
        avg_consistency = details_df["consistency"].mean() if "consistency" in details_df.columns else 0.0

        # Add to submission data
        submission_data.append(
            {"submission_id": submission_id, "avg_score": avg_score, "avg_individual": avg_individual, "avg_consistency": avg_consistency}
        )

    if not submission_data:
        return (0.0, 0.0, 0.0)

    # Convert to DataFrame for easier analysis
    submissions_df = pd.DataFrame(submission_data)

    # Find the submission with the highest average score
    best_idx = submissions_df["avg_score"].idxmax()
    best_submission = submissions_df.iloc[best_idx]

    return (best_submission["avg_score"], best_submission["avg_individual"], best_submission["avg_consistency"])


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

    Extracts scores from model responses and updates the scores dictionary.
    The scores are stored with lowercase category names (e.g., 'grammar', 'creativity').

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

            # Update total score (using 'overall' category)
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
    """Generate a per-model DataFrame for a submission.

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


def save_model_csv(
    username: str,
    submission_id: str,
    model_name: str,
    prompts: List[str],
    completions: List[str],
    item_scores: Dict[int, Dict[str, float]],
    base_dir: str = "submissions",
    overwrite: bool = False,
):
    """Save per-model CSV for a submission.

    Args:
        username: Username of the submission
        submission_id: ID of the submission
        model_name: Name of the model
        prompts: List of prompts
        completions: List of completions
        item_scores: Dictionary mapping item IDs to score dictionaries
        base_dir: Base directory for submissions
        overwrite: Whether to overwrite existing CSV files

    Returns:
        Path to the saved CSV file
    """
    # Create the directory structure
    submission_dir = Path(base_dir) / username / submission_id
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Define the CSV file path
    csv_path = submission_dir / f"{model_name}.csv"

    # Check if the CSV already exists and we're not overwriting
    if csv_path.exists() and not overwrite:
        console.print(f"[yellow]Per-model CSV already exists: {csv_path}[/yellow]")
        return csv_path

    # Generate the DataFrame
    df = generate_model_dataframe(
        username=username,
        submission_id=submission_id,
        model_name=model_name,
        prompts=prompts,
        completions=completions,
        item_scores=item_scores,
    )

    # Save the DataFrame to CSV
    df.to_csv(csv_path, index=False)
    action = "Updated" if csv_path.exists() and overwrite else "Saved"
    console.print(f"[green]{action} per-model CSV to: {csv_path}[/green]")

    return csv_path


def find_model_csvs(username: str, submission_id: str, base_dir: str = "submissions"):
    """Find all model CSV files for a submission.

    Args:
        username: Username of the submission
        submission_id: ID of the submission
        base_dir: Base directory for submissions

    Returns:
        List of model CSV paths
    """
    submission_dir = Path(base_dir) / username / submission_id
    if not submission_dir.exists():
        return []

    # Find all CSV files except summary files
    model_csvs = [file for file in submission_dir.glob("*.csv") if file.stem != "submission_summary"]
    return model_csvs


def calculate_submission_average(username: str, submission_id: str, base_dir: str = "submissions"):
    """Calculate average scores across all models for a submission.

    This function aggregates results from multiple model CSVs for a single submission.
    It's different from calc_scores, which finds the best submission across all of a user's submissions.

    Args:
        username: Username of the submission
        submission_id: ID of the submission
        base_dir: Base directory for submissions

    Returns:
        DataFrame with averaged scores, or None if no model CSVs found
    """
    # Find all model CSVs
    model_csvs = find_model_csvs(username, submission_id, base_dir)

    if not model_csvs:
        console.print(f"[yellow]No model CSVs found for {username}/{submission_id}[/yellow]")
        return None

    # Load and merge all model CSVs
    all_scores = []
    for csv_path in model_csvs:
        model_name = csv_path.stem
        df = pd.read_csv(csv_path)
        # Add model name column
        df["model"] = model_name
        all_scores.append(df)

    if not all_scores:
        return None

    # Combine all model dataframes
    combined_df = pd.concat(all_scores, ignore_index=True)

    # Get score categories - use lowercase versions of the enum values
    score_categories = [cat.lower() for cat in ScoreCategory]
    available_categories = [cat for cat in score_categories if cat in combined_df.columns]

    if not available_categories:
        console.print(f"[red]No score categories found in model CSVs for {username}/{submission_id}[/red]")
        return None

    # Calculate averages across all models for each item
    avg_by_item = combined_df.groupby("item_id")[available_categories].mean().reset_index()

    # Calculate overall submission average (across all items and models)
    overall_avg = {}
    for category in available_categories:
        overall_avg[category] = combined_df[category].mean()

    # Create a summary row
    summary_row = pd.DataFrame([overall_avg])
    summary_row["item_id"] = -1  # Use -1 to indicate this is a summary
    summary_row["username"] = username
    summary_row["submission_id"] = submission_id

    # Return the summary row
    return summary_row


def save_submission_summary(username: str, submission_id: str, base_dir: str = "submissions", overwrite: bool = False):
    """Generate and save a submission summary CSV.

    This creates a summary of scores across all models for a single submission.

    Args:
        username: Username of the submission
        submission_id: ID of the submission
        base_dir: Base directory for submissions
        overwrite: Whether to overwrite existing summary file

    Returns:
        Path to the summary CSV if successful, None otherwise
    """
    # Calculate submission average
    summary_df = calculate_submission_average(username, submission_id, base_dir)

    if summary_df is None:
        return None

    # Create the directory structure
    submission_dir = Path(base_dir) / username / submission_id
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Define the summary CSV path
    summary_path = submission_dir / "submission_summary.csv"

    # Check if the summary already exists and we're not overwriting
    if summary_path.exists() and not overwrite:
        console.print(f"[yellow]Submission summary already exists: {summary_path}[/yellow]")
        return summary_path

    # Save the summary DataFrame
    summary_df.to_csv(summary_path, index=False)
    action = "Updated" if summary_path.exists() and overwrite else "Saved"
    console.print(f"[green]{action} submission summary to: {summary_path}[/green]")

    return summary_path


def update_user_score_history(
    username: str,
    submission_id: str,
    summary_df: pd.DataFrame,
    base_dir: str = "submissions",
    overwrite: bool = True,  # Default to overwrite for history
):
    """Update the user's score history with this submission.

    This maintains a record of all submissions by a user and their scores.

    Args:
        username: Username of the submission
        submission_id: ID of the submission
        summary_df: DataFrame with submission summary
        base_dir: Base directory for submissions
        overwrite: Whether to overwrite existing entries for this submission_id

    Returns:
        Path to the updated history CSV
    """
    # Create the user directory if it doesn't exist
    user_dir = Path(base_dir) / username
    user_dir.mkdir(parents=True, exist_ok=True)

    history_path = user_dir / "score_history.csv"

    # Create a new history entry
    history_entry = summary_df.copy()
    history_entry["timestamp"] = datetime.datetime.now().isoformat()

    # Check if history file exists
    if history_path.exists():
        # Load existing history
        history_df = pd.read_csv(history_path)

        # Check if this submission is already in the history
        if submission_id in history_df["submission_id"].values and overwrite:
            # Update the existing entry
            history_df = history_df[history_df["submission_id"] != submission_id]
        elif submission_id in history_df["submission_id"].values and not overwrite:
            console.print(f"[yellow]Submission {submission_id} already in history and overwrite=False[/yellow]")
            return history_path

        # Append the new entry
        history_df = pd.concat([history_df, history_entry], ignore_index=True)
    else:
        # Create new history
        history_df = history_entry

    # Save the updated history
    history_df.to_csv(history_path, index=False)
    console.print(f"[green]User score history updated: {history_path}[/green]")

    return history_path
