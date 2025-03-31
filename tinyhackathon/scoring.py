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
    """Write scores to a CSV file.

    Args:
        path: Path to output CSV file
        scores: Dictionary containing score data
    """
    header = ["username", "submission_id", "model_arch", "score", "item_id", "grammar", "creativity", "consistency", "plot", "overall"]
    with open(path.as_posix(), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for user in scores:
            for submission in scores[user]:
                for model_arch in scores[user][submission]:
                    score = scores[user][submission][model_arch]["score"]
                    for item in scores[user][submission][model_arch]["details"]:
                        ind_scores = [(item["scores"][h] if h in item["scores"] else "#") for h in header[5:]]
                        writer.writerow([user, submission, model_arch, score, item["item_id"], *ind_scores])


def read_csv(path: str):
    """Read scores from a CSV file.

    Args:
        path: Path to input CSV file

    Returns:
        Dictionary containing score data
    """
    with open(path, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        scores = {}
        for item in reader:
            (
                user,
                submission,
                model_arch,
                score,
            ) = item[0], item[1], item[2], item[3]
            if user not in scores:
                scores[user] = {}
            if submission not in scores[user]:
                scores[user][submission] = {}
            if model_arch not in scores[user][submission]:
                scores[user][submission][model_arch] = {
                    "score": float(score),
                }
            item_id = item[4]
            ind_scores = {h: float(r) for h, r in zip(header[5:], item[5:]) if r != "#"}
            if "details" not in scores[user][submission][model_arch]:
                scores[user][submission][model_arch]["details"] = []
            scores[user][submission][model_arch]["details"] += [{header[4]: int(item_id), "scores": ind_scores}]
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


def generate_model_csv(
    username: str,
    submission_id: str,
    model_name: str,
    prompts: List[str],
    completions: List[str],
    item_scores: Dict[int, Dict[str, float]],
):
    """Generate a per-model CSV for a submission.

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
