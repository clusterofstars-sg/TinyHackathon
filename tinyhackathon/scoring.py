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
            score_categories = [cat.lower() for cat in ScoreCategory]
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
    The scores are stored with lowercase category names from the ScoreCategory enum.

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

            # Update total score (using the ScoreCategory.OVERALL category)
            overall_category = ScoreCategory.OVERALL.lower()
            total_score += extracted_scores.get(overall_category, 0)
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
    It finds and combines all model evaluation results for one specific submission.

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


def read_user_metadata(username: str, base_dir: str = "submissions") -> str:
    """Read a user's metadata to get their weight class.

    Args:
        username: The username to look up
        base_dir: Base directory for submissions

    Returns:
        Weight class (small, medium, or large), defaults to "small" if not found
    """
    user_dir = Path(base_dir) / username
    metadata_path = user_dir / "metadata.json"

    # Default to small if metadata doesn't exist
    if not metadata_path.exists():
        return "small"

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Get weight class or default to small
        weight_class = metadata.get("weight_class", "small")

        # Validate weight class
        if weight_class not in ["small", "medium", "large"]:
            console.print(f"[yellow]Invalid weight class: {weight_class} for user {username}, defaulting to 'small'[/yellow]")
            return "small"

        return weight_class
    except Exception as e:
        console.print(f"[red]Error reading metadata for {username}: {e}[/red]")
        return "small"


def find_all_user_histories(base_dir: str = "submissions") -> List[Path]:
    """Find all user score history files.

    Args:
        base_dir: Base directory for submissions

    Returns:
        List of paths to score_history.csv files
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    # Find all score_history.csv files
    history_files = list(base_path.glob("*/score_history.csv"))
    return history_files


def generate_global_leaderboard(base_dir: str = "submissions") -> pd.DataFrame:
    """Generate a global leaderboard across all users.

    Args:
        base_dir: Base directory for submissions

    Returns:
        DataFrame containing the global leaderboard
    """
    # Find all user history files
    history_files = find_all_user_histories(base_dir)

    if not history_files:
        console.print(f"[yellow]No user history files found in {base_dir}[/yellow]")
        return pd.DataFrame()

    # Load and combine all histories
    user_data = []

    for history_path in history_files:
        try:
            # Get username from directory path
            username = history_path.parent.name

            # Read history file
            history = pd.read_csv(history_path)

            if history.empty:
                continue

            # For each user, find their best submission based on ScoreCategory.OVERALL score
            overall_score_category = ScoreCategory.OVERALL.lower()
            if overall_score_category in history.columns:
                best_submission = history.loc[history[overall_score_category].idxmax()]
            else:
                # Fall back to first submission if no overall score
                best_submission = history.iloc[0]

            # Get individual category scores
            category_scores = {}
            for category in ScoreCategory:
                category_lower = category.lower()
                if category_lower in best_submission:
                    category_scores[category_lower] = best_submission[category_lower]
                else:
                    category_scores[category_lower] = 0.0

            # Calculate total number of submissions for this user
            submission_count = len(history)

            # Get submission timestamp
            timestamp = best_submission["timestamp"] if "timestamp" in best_submission else ""

            # Get submission_id
            submission_id = best_submission["submission_id"] if "submission_id" in best_submission else ""

            # Get weight class
            weight_class = read_user_metadata(username, base_dir)

            # Create user entry
            user_entry = {
                "username": username,
                "submission_id": submission_id,
                "submission_count": submission_count,
                "submission_date": timestamp,
                "weight_class": weight_class,
                **category_scores,  # Add all category scores individually
            }

            user_data.append(user_entry)

        except Exception as e:
            console.print(f"[red]Error processing history for {history_path.parent.name}: {e}[/red]")
            continue

    if not user_data:
        return pd.DataFrame()

    # Create leaderboard DataFrame
    leaderboard = pd.DataFrame(user_data)

    # Sort by overall score descending
    leaderboard = leaderboard.sort_values(by=ScoreCategory.OVERALL.lower(), ascending=False).reset_index(drop=True)

    # Add rank column
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))

    return leaderboard


def save_global_leaderboard(leaderboard: pd.DataFrame, base_dir: str = ".") -> Dict[str, Path]:
    """Save the global leaderboard to CSV and Markdown.

    Args:
        leaderboard: DataFrame containing the global leaderboard
        base_dir: Base directory to save the leaderboard files

    Returns:
        Dictionary with paths to the saved files
    """
    base_path = Path(base_dir)

    if leaderboard.empty:
        console.print("[yellow]Empty leaderboard, not saving[/yellow]")
        return {}

    saved_files = {}

    # Save CSV
    csv_path = base_path / "leaderboard_global.csv"
    leaderboard.to_csv(csv_path, index=False)
    saved_files["csv"] = csv_path
    console.print(f"[green]Global leaderboard saved to {csv_path}[/green]")

    # Save Markdown
    md_path = base_path / "leaderboard_global.md"

    # Define column order with overall first, then other categories
    score_categories = [ScoreCategory.OVERALL.lower()]
    for category in ScoreCategory:
        if category.lower() != ScoreCategory.OVERALL.lower():
            score_categories.append(category.lower())

    # Create a new DataFrame with columns in the desired order
    display_cols = ["rank", "username"] + score_categories + ["submission_count", "submission_date"]
    display_df = leaderboard[display_cols].copy()

    # Format floating point columns
    for col in score_categories:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")

    # Create markdown with a header
    md_content = "# Global Leaderboard\n\n"
    md_content += display_df.to_markdown(index=False, tablefmt="pipe")

    with open(md_path, "w") as f:
        f.write(md_content)

    saved_files["md"] = md_path
    console.print(f"[green]Global leaderboard markdown saved to {md_path}[/green]")

    return saved_files


def generate_weight_class_leaderboards(leaderboard: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Generate separate leaderboards for each weight class.

    Args:
        leaderboard: The global leaderboard DataFrame

    Returns:
        Dictionary mapping weight class to its leaderboard DataFrame
    """
    if leaderboard.empty:
        return {}

    # Ensure weight_class column exists
    if "weight_class" not in leaderboard.columns:
        console.print("[yellow]Leaderboard missing weight_class column, cannot generate weight class leaderboards[/yellow]")
        return {}

    # Group by weight class
    weight_class_leaderboards = {}

    for weight_class in ["small", "medium", "large"]:
        # Filter the leaderboard by weight class
        class_leaderboard = leaderboard[leaderboard["weight_class"] == weight_class].copy()

        if class_leaderboard.empty:
            console.print(f"[yellow]No entries for weight class '{weight_class}'[/yellow]")
            continue

        # Rerank within this weight class
        class_leaderboard = class_leaderboard.sort_values(by=ScoreCategory.OVERALL.lower(), ascending=False).reset_index(drop=True)
        class_leaderboard["rank"] = range(1, len(class_leaderboard) + 1)

        weight_class_leaderboards[weight_class] = class_leaderboard

    return weight_class_leaderboards


def save_weight_class_leaderboards(weight_class_leaderboards: Dict[str, pd.DataFrame], base_dir: str = ".") -> Dict[str, Path]:
    """Save weight class leaderboards to CSV and Markdown.

    Args:
        weight_class_leaderboards: Dictionary of weight class leaderboards
        base_dir: Base directory to save the leaderboard files

    Returns:
        Dictionary with paths to the saved files
    """
    base_path = Path(base_dir)
    saved_files = {}

    if not weight_class_leaderboards:
        console.print("[yellow]No weight class leaderboards to save[/yellow]")
        return saved_files

    # Create combined weight class DataFrame
    all_classes = []
    for weight_class, leaderboard in weight_class_leaderboards.items():
        if not leaderboard.empty:
            all_classes.append(leaderboard)

    if all_classes:
        combined = pd.concat(all_classes, ignore_index=True)

        # Save combined CSV
        combined_path = base_path / "leaderboard_weight_class.csv"
        combined.to_csv(combined_path, index=False)
        saved_files["combined_csv"] = combined_path
        console.print(f"[green]Combined weight class leaderboard saved to {combined_path}[/green]")

    # Define column order with overall first, then other categories
    score_categories = [ScoreCategory.OVERALL.lower()]
    for category in ScoreCategory:
        if category.lower() != ScoreCategory.OVERALL.lower():
            score_categories.append(category.lower())

    # Save individual weight class files
    for weight_class, leaderboard in weight_class_leaderboards.items():
        if leaderboard.empty:
            continue

        # Save CSV
        csv_path = base_path / f"leaderboard_{weight_class}.csv"
        leaderboard.to_csv(csv_path, index=False)
        saved_files[f"{weight_class}_csv"] = csv_path
        console.print(f"[green]{weight_class.capitalize()} weight class leaderboard saved to {csv_path}[/green]")

        # Save Markdown
        md_path = base_path / f"leaderboard_{weight_class}.md"

        # Create a new DataFrame with columns in the desired order
        display_cols = ["rank", "username"] + score_categories + ["submission_count", "submission_date"]
        display_df = leaderboard[display_cols].copy()

        # Format floating point columns
        for col in score_categories:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")

        # Create markdown with a header
        md_content = f"# {weight_class.capitalize()} Weight Class Leaderboard\n\n"
        md_content += display_df.to_markdown(index=False, tablefmt="pipe")

        with open(md_path, "w") as f:
            f.write(md_content)

        saved_files[f"{weight_class}_md"] = md_path
        console.print(f"[green]{weight_class.capitalize()} weight class markdown saved to {md_path}[/green]")

    return saved_files


def generate_and_save_all_leaderboards(base_dir: str = "submissions", output_dir: str = ".") -> Dict[str, Path]:
    """Generate and save all leaderboards in one function.

    This is a convenience function that combines all leaderboard generation
    and saving into a single call.

    Args:
        base_dir: Base directory for submissions
        output_dir: Directory to save leaderboard files

    Returns:
        Dictionary with paths to all saved files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate global leaderboard
    global_leaderboard = generate_global_leaderboard(base_dir)

    if global_leaderboard.empty:
        console.print("[yellow]No data for leaderboards[/yellow]")
        return {}

    # Save global leaderboard
    saved_files = save_global_leaderboard(global_leaderboard, output_dir)

    # Generate weight class leaderboards
    weight_class_leaderboards = generate_weight_class_leaderboards(global_leaderboard)

    # Save weight class leaderboards
    weight_class_files = save_weight_class_leaderboards(weight_class_leaderboards, output_dir)

    # Combine all saved files
    saved_files.update(weight_class_files)

    return saved_files
