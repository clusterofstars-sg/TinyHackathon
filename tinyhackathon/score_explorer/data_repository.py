import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Import necessary functions from scoring.py
from tinyhackathon.scoring import ScoreCategory

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data_repository")

# Path configuration
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "score_explorer_data"
SUBMISSIONS_DIR = DATA_DIR / "downloaded_submissions"
SCORES_DIR = DATA_DIR / "scores"
SUBMISSIONS_SUBDIR = SUBMISSIONS_DIR / "submissions"
SCORES_SUBDIR = SCORES_DIR / "submissions"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
SCORES_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache for loaded submission DataFrames
submission_data_cache: Dict[tuple[str, str], pd.DataFrame] = {}


def find_submission_csv(username: str, submission_id: str) -> Optional[Path]:
    """Find the submission CSV file in either the main or submissions subdirectory."""
    # Try direct path first
    direct_path = SUBMISSIONS_DIR / username / f"{submission_id}.csv"
    if direct_path.exists():
        return direct_path

    # Try submissions subdirectory
    subdir_path = SUBMISSIONS_SUBDIR / username / f"{submission_id}.csv"
    if subdir_path.exists():
        return subdir_path

    return None


def find_score_directory(username: str, submission_id: str) -> Optional[Path]:
    """Find the score directory in either the main or submissions subdirectory."""
    # Try direct path first
    direct_path = SCORES_DIR / username / submission_id
    logger.info("Checking for score directory at direct path: " + str(direct_path))
    if direct_path.exists() and direct_path.is_dir():
        logger.info("Found score directory at direct path")
        # Check if this directory has score files
        model_csvs = list(direct_path.glob("*.csv"))
        if model_csvs:
            logger.info("Found " + str(len(model_csvs)) + " CSV files at direct path: " + str([csv.name for csv in model_csvs]))
            return direct_path
        else:
            logger.info("No CSV files found at direct path")

    # Try submissions subdirectory
    subdir_path = SCORES_SUBDIR / username / submission_id
    logger.info("Checking for score directory at submissions subdir path: " + str(subdir_path))
    if subdir_path.exists() and subdir_path.is_dir():
        logger.info("Found score directory at submissions subdir path")
        # Check if this directory has score files
        model_csvs = list(subdir_path.glob("*.csv"))
        if model_csvs:
            logger.info("Found " + str(len(model_csvs)) + " CSV files at submissions subdir path: " + str([csv.name for csv in model_csvs]))
            return subdir_path
        else:
            logger.info("No CSV files found at submissions subdir path")

    # If we get here, no score directory with CSV files was found
    logger.warning("No score directory with CSV files found for " + username + "/" + submission_id)
    return None


def list_users() -> List[str]:
    """Get all user directories from both main and submissions subdirectories."""
    users = set()

    # Get users from direct path
    if SCORES_DIR.exists():
        for d in SCORES_DIR.iterdir():
            if d.is_dir() and d.name not in [".cache", "submissions"]:
                users.add(d.name)

    # Get users from submissions subdirectory
    if SCORES_SUBDIR.exists():
        for d in SCORES_SUBDIR.iterdir():
            if d.is_dir():
                users.add(d.name)

    return sorted(list(users))


def list_submissions(username: str) -> List[str]:
    """Get all submissions for a user from both main and submissions subdirectories."""
    submissions = set()

    # Try direct path
    direct_path = SCORES_DIR / username
    if direct_path.exists() and direct_path.is_dir():
        for d in direct_path.iterdir():
            if d.is_dir():
                submissions.add(d.name)

    # Try submissions subdirectory
    subdir_path = SCORES_SUBDIR / username
    if subdir_path.exists() and subdir_path.is_dir():
        for d in subdir_path.iterdir():
            if d.is_dir():
                submissions.add(d.name)

    return sorted(list(submissions), reverse=True)  # Most recent first


def load_and_merge_submission_data(username: str, submission_id: str) -> pd.DataFrame:
    """Loads submission and score CSVs, merges them into a single DataFrame with numeric scores."""
    logger.info("Performing full CSV load and merge for " + username + "/" + submission_id)
    submission_csv_path = find_submission_csv(username, submission_id)
    score_dir = find_score_directory(username, submission_id)

    if not submission_csv_path or not score_dir:
        logger.error("Cannot find submission CSV or score directory for " + username + "/" + submission_id)
        return pd.DataFrame()  # Return empty DataFrame if files not found

    # Read submission CSV
    submissions_df = pd.read_csv(submission_csv_path)
    # Ensure item_id is an integer for proper merging
    if "item_id" in submissions_df.columns:
        submissions_df["item_id"] = submissions_df["item_id"].astype(int)

    # Get all model CSVs - explicitly exclude submission_summary.csv and score_history.csv
    model_csvs = [csv_file for csv_file in score_dir.glob("*.csv") if csv_file.name not in ["submission_summary.csv", "score_history.csv"]]

    logger.info("Found " + str(len(model_csvs)) + " model CSV files: " + str([csv.name for csv in model_csvs]))

    if not model_csvs:
        logger.warning("No model CSV files found in " + str(score_dir) + "!")
        return pd.DataFrame()

    # Initialize an empty list to store all model dataframes
    all_model_dfs = []

    # Process each model CSV
    for model_csv in model_csvs:
        model_name = model_csv.stem
        logger.info("Processing scores for model: " + model_name)

        # Read score CSV
        scores_df = pd.read_csv(model_csv)

        # Make sure we have item_id column or add it
        if "item_id" not in scores_df.columns:
            scores_df["item_id"] = scores_df.index

        scores_df["item_id"] = scores_df["item_id"].astype(int)

        # Add model name to the scores dataframe
        scores_df["model_name"] = model_name

        # Merge with submission data
        if "item_id" in submissions_df.columns:
            # Direct merge if item_id is in submissions
            merged_df = pd.merge(scores_df, submissions_df[["item_id", "prompt", "completion"]], on="item_id", how="left")
        else:
            # If no item_id in submissions, use index-based matching
            # First try matching by position
            scores_df = scores_df.reset_index(drop=True)
            temp_submissions = submissions_df.reset_index(drop=True)

            # Add item_id-based columns from submissions
            for col in ["prompt", "completion"]:
                if col in temp_submissions.columns:
                    scores_df[col] = scores_df["item_id"].apply(
                        lambda x: temp_submissions.iloc[x][col] if x < len(temp_submissions) else None
                    )

            merged_df = scores_df

        # Extract and expand score categories
        score_cols = []
        for score_category in ScoreCategory.__members__:
            category_lower = score_category.lower()
            if category_lower in merged_df.columns:
                score_cols.append(category_lower)

        # Select only needed columns
        cols_to_keep = ["item_id", "prompt", "completion", "model_name"] + score_cols
        merged_df = merged_df[[col for col in cols_to_keep if col in merged_df.columns]]

        all_model_dfs.append(merged_df)

    # Concatenate all model dataframes
    if not all_model_dfs:
        logger.warning("No model data could be processed for " + username + "/" + submission_id)
        return pd.DataFrame()

    final_df = pd.concat(all_model_dfs, ignore_index=True)

    # Convert score columns to numeric
    numeric_cols = [col for col in final_df.columns if col not in ["item_id", "prompt", "completion", "model_name"]]
    for col in numeric_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

    logger.info("Created final merged DataFrame with " + str(len(final_df)) + " total rows. Columns: " + str(final_df.columns.tolist()))

    return final_df


def get_cached_submission_data(username: str, submission_id: str) -> pd.DataFrame:
    """Retrieves the merged DataFrame from cache or loads it if not present."""
    cache_key = (username, submission_id)
    if cache_key in submission_data_cache:
        logger.info("Returning cached data for " + username + "/" + submission_id)
        return submission_data_cache[cache_key]
    else:
        logger.info("No cache found for " + username + "/" + submission_id + ". Loading from CSVs.")
        df = load_and_merge_submission_data(username, submission_id)
        submission_data_cache[cache_key] = df  # Store in cache
        return df


def calculate_dataframe_averages(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Calculate model and submission averages from the merged DataFrame."""
    if df.empty:
        return {}, {}

    # Identify all score categories (assuming they're the numeric columns except item_id)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    score_categories = [col for col in numeric_cols if col != "item_id"]

    # Calculate averages per model
    model_averages = {}
    if "model_name" in df.columns:
        # Use groupby and agg to calculate means for all score categories at once
        model_aggs = df.groupby("model_name")[score_categories].mean().reset_index()

        # Convert to dictionary format
        for _, row in model_aggs.iterrows():
            model_name = row["model_name"]
            model_scores = {category: row[category] for category in score_categories if not pd.isna(row[category])}
            model_averages[model_name] = model_scores

    # Calculate averages per item_id
    item_averages = {}
    if "item_id" in df.columns:
        # Use groupby and agg to calculate means for all score categories at once
        item_aggs = df.groupby("item_id")[score_categories].mean().reset_index()

        # Convert to dictionary format
        for _, row in item_aggs.iterrows():
            item_id = row["item_id"]
            item_scores = {category: row[category] for category in score_categories if not pd.isna(row[category])}
            item_averages[item_id] = item_scores

    return model_averages, item_averages


def apply_search(df: pd.DataFrame, search_string: str) -> pd.DataFrame:
    """Filter DataFrame based on search criteria in prompt, completion, or item_id."""
    if not search_string or df.empty:
        return df

    search_lower = search_string.lower()

    # Use vectorized operations for searching
    prompt_mask = df["prompt"].astype(str).str.lower().str.contains(search_lower, na=False)
    completion_mask = df["completion"].astype(str).str.lower().str.contains(search_lower, na=False)
    item_id_mask = df["item_id"].astype(str).str.contains(search_string, na=False)

    # Combine masks with OR operation
    combined_mask = prompt_mask | completion_mask | item_id_mask

    return df[combined_mask]


def apply_sort(df: pd.DataFrame, sort_category: str = "overall", ascending: bool = False) -> pd.DataFrame:
    """Sort DataFrame by specified category."""
    if df.empty or sort_category not in df.columns:
        return df

    return df.sort_values(by=sort_category, ascending=ascending)


def apply_pagination(df: pd.DataFrame, page: int = 1, items_per_page: int = 20) -> Tuple[pd.DataFrame, int]:
    """
    Apply pagination to DataFrame.

    Returns:
        Tuple containing the paginated DataFrame and the total number of pages
    """
    if df.empty:
        return df, 0

    total_items = len(df)
    total_pages = (total_items + items_per_page - 1) // items_per_page  # Ceiling division

    page = max(1, min(page, total_pages))  # Ensure page is within bounds
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)

    return df.iloc[start_idx:end_idx], total_pages


def get_unique_item_ids(df: pd.DataFrame) -> List[int]:
    """Get a list of unique item_ids from the DataFrame."""
    if df.empty or "item_id" not in df.columns:
        return []

    return sorted(df["item_id"].unique().tolist())


def get_item_data(df: pd.DataFrame, item_id: int) -> Tuple[str, str, List[Dict], Dict]:
    """
    Get all data for a specific item_id.

    Returns:
        Tuple containing (prompt, completion, evaluations, item_averages)
    """
    if df.empty or "item_id" not in df.columns:
        return "", "", [], {}

    item_data = df[df["item_id"] == item_id]

    if item_data.empty:
        return "", "", [], {}

    # Get prompt/completion from first row (should be same across models)
    first_row = item_data.iloc[0]
    prompt = first_row.get("prompt", "")
    completion = first_row.get("completion", "")

    # Get score categories (all numeric columns except item_id)
    numeric_cols = item_data.select_dtypes(include=["number"]).columns.tolist()
    categories = [col for col in numeric_cols if col != "item_id"]

    # Process evaluations (scores per model)
    evaluations = []
    for model_name, model_df in item_data.groupby("model_name"):
        model_row = model_df.iloc[0]
        scores = {cat: model_row.get(cat) for cat in categories if pd.notna(model_row.get(cat))}
        evaluations.append({"model_name": model_name, "scores": scores})

    # Calculate averages for this specific item
    item_averages = {}
    for category in categories:
        item_averages[category] = item_data[category].mean()

    return prompt, completion, evaluations, item_averages


def read_leaderboard_file(filename: str) -> Optional[pd.DataFrame]:
    """Read a leaderboard CSV file into a pandas DataFrame."""
    leaderboard_path = DATA_DIR / filename
    if not leaderboard_path.exists():
        logger.warning("Leaderboard file not found: " + str(leaderboard_path))
        return None

    try:
        df = pd.read_csv(leaderboard_path)
        logger.info("Read leaderboard file " + filename + " with " + str(len(df)) + " entries")
        return df
    except Exception as e:
        logger.error("Error reading leaderboard file " + filename + ": " + str(e))
        return None


def get_all_leaderboards() -> Dict[str, pd.DataFrame]:
    """Get all available leaderboard data."""
    leaderboards = {}

    # Look for leaderboard files in the data directory
    leaderboard_files = list(DATA_DIR.glob("leaderboard_*.csv"))
    logger.info("Found " + str(len(leaderboard_files)) + " leaderboard files")

    for file in leaderboard_files:
        name = file.stem.replace("leaderboard_", "")
        df = read_leaderboard_file(file.name)
        if df is not None:
            leaderboards[name] = df

    return leaderboards


def compute_user_stats() -> pd.DataFrame:
    """Compute statistics for all users."""
    users = list_users()
    user_stats = []

    for username in users:
        submissions = list_submissions(username)
        submission_count = len(submissions)

        # Get the best submission (first in the sorted list)
        best_submission = submissions[0] if submissions else None
        best_score = None

        if best_submission:
            # Try to find best score from leaderboard or calculate from data
            leaderboards = get_all_leaderboards()
            global_leaderboard = leaderboards.get("global")

            if global_leaderboard is not None and "username" in global_leaderboard.columns:
                user_row = global_leaderboard[global_leaderboard["username"] == username]
                if not user_row.empty and "score" in user_row.columns:
                    best_score = user_row["score"].values[0]

            # If not found in leaderboard, calculate from submission data
            if best_score is None and best_submission:
                df = get_cached_submission_data(username, best_submission)
                if not df.empty and "overall" in df.columns:
                    best_score = df["overall"].mean()

        user_stats.append(
            {"username": username, "submission_count": submission_count, "best_submission": best_submission, "best_score": best_score}
        )

    return pd.DataFrame(user_stats)


def check_data_directory_structure() -> bool:
    """Check if the data directories have a valid structure for browsing."""
    logger.info("Checking data directory structure")

    # Check if directories exist
    if not DATA_DIR.exists():
        logger.error("Data directory " + str(DATA_DIR) + " does not exist")
        return False

    if not SUBMISSIONS_DIR.exists() and not SUBMISSIONS_SUBDIR.exists():
        logger.error("Neither submissions directory " + str(SUBMISSIONS_DIR) + " nor " + str(SUBMISSIONS_SUBDIR) + " exists")
        return False

    if not SCORES_DIR.exists() and not SCORES_SUBDIR.exists():
        logger.error("Neither scores directory " + str(SCORES_DIR) + " nor " + str(SCORES_SUBDIR) + " exists")
        return False

    # Check for at least one user directory
    users = list_users()
    if not users:
        logger.error("No user directories found")
        return False

    logger.info("Found " + str(len(users)) + " user directories")

    # Check for at least one submission per user
    valid_users = 0
    for username in users:
        submissions = list_submissions(username)
        if submissions:
            valid_users += 1
            logger.info("User " + username + " has " + str(len(submissions)) + " submissions")

    if valid_users == 0:
        logger.error("No users with valid submissions found")
        return False

    logger.info("Found " + str(valid_users) + " users with valid submissions")
    return True
