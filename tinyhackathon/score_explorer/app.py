from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import sys
import logging
from pathlib import Path
from typing import Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("score_explorer")

# Add the parent directory to the path to import from scoring.py and llm_scoring.py
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))
tinyhackathon_dir = Path(__file__).parent.parent
sys.path.append(str(tinyhackathon_dir))

# Import necessary functions from scoring.py
from tinyhackathon.scoring import (
    EnvConfig,
    setup_api_client,
    download_submissions_and_scores,
    ScoreCategory,
)

app = FastAPI(title="Score Explorer", description="A web app to explore TinyHackathon scores with merged prompts and completions")

# Setup templates - using relative path from the current file
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Create data directories in the project instead of temp
DATA_DIR = Path(__file__).parent.parent.parent / "score_explorer_data"
SUBMISSIONS_DIR = DATA_DIR / "downloaded_submissions"
SCORES_DIR = DATA_DIR / "scores"

# Define submission subdirectories path - but also handle direct paths
SUBMISSIONS_SUBDIR = SUBMISSIONS_DIR / "submissions"
SCORES_SUBDIR = SCORES_DIR / "submissions"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
SCORES_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache for loaded submission DataFrames
submission_data_cache: Dict[tuple[str, str], pd.DataFrame] = {}


def find_submission_csv(username, submission_id):
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


def find_score_directory(username, submission_id):
    """Find the score directory in either the main or submissions subdirectory."""
    # Try direct path first
    direct_path = SCORES_DIR / username / submission_id
    logger.info(f"Checking for score directory at direct path: {direct_path}")
    if direct_path.exists() and direct_path.is_dir():
        logger.info(f"Found score directory at direct path")
        # Check if this directory has score files
        model_csvs = list(direct_path.glob("*.csv"))
        if model_csvs:
            logger.info(f"Found {len(model_csvs)} CSV files at direct path: {[csv.name for csv in model_csvs]}")
            return direct_path
        else:
            logger.info(f"No CSV files found at direct path")

    # Try submissions subdirectory
    subdir_path = SCORES_SUBDIR / username / submission_id
    logger.info(f"Checking for score directory at submissions subdir path: {subdir_path}")
    if subdir_path.exists() and subdir_path.is_dir():
        logger.info(f"Found score directory at submissions subdir path")
        # Check if this directory has score files
        model_csvs = list(subdir_path.glob("*.csv"))
        if model_csvs:
            logger.info(f"Found {len(model_csvs)} CSV files at submissions subdir path: {[csv.name for csv in model_csvs]}")
            return subdir_path
        else:
            logger.info(f"No CSV files found at submissions subdir path")

    # If we get here, no score directory with CSV files was found
    logger.warning(f"No score directory with CSV files found for {username}/{submission_id}")
    return None


def get_all_user_directories():
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


def get_user_submissions(username):
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


def merge_score_and_submission(submission_csv_path, score_csv_path):
    """Merge score and submission data for viewing"""
    try:
        # Read submission CSV (prompts and completions)
        submissions_df = pd.read_csv(submission_csv_path)
        logger.info(f"Read submission file with {len(submissions_df)} entries")
        logger.info(f"Submission CSV columns: {submissions_df.columns.tolist()}")

        # Read score CSV
        scores_df = pd.read_csv(score_csv_path)
        logger.info(f"Read score file with {len(scores_df)} entries")
        logger.info(f"Score CSV columns: {scores_df.columns.tolist()}")

        # Merge based on item_id
        merged_data = []

        # Check if scores_df has numeric indices that we can use as item_ids
        if "item_id" not in scores_df.columns:
            # Create item_id column based on the index
            scores_df["item_id"] = scores_df.index
            logger.info("Added item_id column based on index")

        # Ensure item_id is an integer for proper comparison
        scores_df["item_id"] = scores_df["item_id"].astype(int)

        for idx, row in scores_df.iterrows():
            item_id = int(row["item_id"])
            logger.info(f"Processing score for item_id: {item_id}")

            submission_row = None

            # First try to match by item_id if available in submissions
            if "item_id" in submissions_df.columns:
                # Force item_id to be integer in submissions_df too
                submissions_df["item_id"] = submissions_df["item_id"].astype(int)
                matching_rows = submissions_df[submissions_df["item_id"] == item_id]
                if not matching_rows.empty:
                    submission_row = matching_rows.iloc[0]
                    logger.info(f"Found matching submission row by item_id")

            # If no match found by item_id, use the index as a fallback
            if submission_row is None:
                # If the item_id is a valid index in the submissions dataframe
                if item_id < len(submissions_df):
                    submission_row = submissions_df.iloc[item_id]
                    logger.info(f"Using item_id {item_id} as index to match with submission")
                # As last resort, use the row index from scores dataframe
                elif idx < len(submissions_df):
                    submission_row = submissions_df.iloc[idx]
                    logger.info(f"Using row index {idx} to match with submission as last resort")
                else:
                    logger.warning(f"No matching submission found for score item_id {item_id}")

            if submission_row is not None:
                scores = {}
                # Extract available score categories
                for score_category in ScoreCategory.__members__:
                    category_lower = score_category.lower()
                    if category_lower in row:
                        score_value = row[category_lower]
                        # Convert numpy types to Python native types for better JSON serialization
                        if hasattr(score_value, "item"):
                            score_value = score_value.item()
                        scores[category_lower] = score_value

                # Log available scores
                logger.info(f"Found scores: {scores}")

                prompt = ""
                completion = ""

                # Extract prompt and completion, handling both string and object access
                if "prompt" in submission_row:
                    prompt = submission_row["prompt"]
                    if isinstance(prompt, pd.Series):
                        prompt = prompt.iloc[0] if not prompt.empty else ""

                if "completion" in submission_row:
                    completion = submission_row["completion"]
                    if isinstance(completion, pd.Series):
                        completion = completion.iloc[0] if not completion.empty else ""

                merged_item = {
                    "item_id": item_id,
                    "prompt": prompt,
                    "completion": completion,
                    "scores": scores,
                }
                merged_data.append(merged_item)
                logger.info(f"Added merged item for item_id {item_id} with {len(scores)} scores")
            else:
                logger.warning(f"No matching submission found for score item_id {item_id}")

        logger.info(f"Total merged items: {len(merged_data)}")
        return merged_data
    except Exception as e:
        logger.exception(f"Error merging score and submission: {e}")
        return []


def check_data_directory_structure():
    """Check if the data directories have a valid structure for browsing"""
    if not SCORES_DIR.exists() or not SUBMISSIONS_DIR.exists():
        return False

    # Get all users
    users = get_all_user_directories()
    if not users:
        return False

    # Check if there's at least one submission with scores
    for username in users:
        submissions = get_user_submissions(username)
        if not submissions:
            continue

        for submission_id in submissions:
            # Find score directory
            score_dir = find_score_directory(username, submission_id)
            if not score_dir:
                continue

            # Check for model CSV files
            model_csvs = list(score_dir.glob("*.csv"))
            if not model_csvs:
                continue

            # Check for corresponding submission file
            submission_csv = find_submission_csv(username, submission_id)
            if submission_csv:
                return True

    return False


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with options to download data and view submissions"""
    # Check if we have data
    has_data = False
    if SCORES_DIR.exists():
        has_data = any(SCORES_DIR.glob("**/*.csv"))

    # Get leaderboard data
    leaderboards = get_all_leaderboards()
    has_leaderboards = len(leaderboards) > 0

    # Get the global leaderboard for display on the home page
    global_leaderboard = None
    if "global" in leaderboards:
        global_leaderboard = leaderboards["global"].head(10).to_dict("records")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "has_data": has_data,
            "has_leaderboards": has_leaderboards,
            "leaderboards": leaderboards,
            "global_leaderboard": global_leaderboard,
        },
    )


@app.get("/download", response_class=HTMLResponse)
async def download_data(request: Request, refresh: bool = False):
    """Download or refresh submission and scoring data"""
    try:
        env_config = EnvConfig(is_test_mode=False, sub_test_override=False, score_test_override=False)
        download_error = None
        leaderboard_results = {}

        # Log environment configuration
        logger.info(f"Environment config: {env_config}")
        logger.info(f"Using DATA_DIR: {DATA_DIR}")

        # Download new data if refresh is requested or no data exists
        if refresh or not any(SCORES_DIR.glob("**/*.csv")):
            # Setup API client (no upload needed)
            logger.info("Setting up API client")
            api_client = setup_api_client(True)

            if api_client is None:
                logger.warning("No Hugging Face API client available - authentication failed")
                download_error = """
                    No Hugging Face API client available. Authentication required for downloading data.

                    If scores already exist in your score_explorer_data directory, you can continue browsing those.
                """
            else:
                # Only download if we have a valid API client
                logger.info("API client authenticated successfully, proceeding with download")
                # Download submissions and scores
                results = download_submissions_and_scores(
                    env_config=env_config, submission_dir=str(SUBMISSIONS_DIR), scores_dir=str(SCORES_DIR), api_client=api_client
                )
                logger.info(f"Download results: {results}")

                # Download leaderboard files
                leaderboard_results = download_leaderboards(env_config=env_config, api_client=api_client)
                logger.info(f"Leaderboard download results: {leaderboard_results}")

        # Count submission files in both main dir and submissions subdir
        submission_count = 0
        if SUBMISSIONS_DIR.exists():
            submission_count += len(list(SUBMISSIONS_DIR.glob("**/*.csv")))

        # Count score files in both main dir and submissions subdir
        score_count = 0
        if SCORES_DIR.exists():
            score_count += len(list(SCORES_DIR.glob("**/*.csv")))

        # Count leaderboard files
        leaderboard_count = 0
        if DATA_DIR.exists():
            leaderboard_count = len(list(DATA_DIR.glob("leaderboard_*.csv")))

        # Get statistics
        download_stats = {
            "submissions": submission_count,
            "scores": score_count,
            "leaderboards": leaderboard_count,
        }

        # Get a list of all users with submissions
        users = get_all_user_directories()
        download_stats["users"] = len(users)

        # Get count of submissions
        submission_count = 0
        for username in users:
            submission_count += len(get_user_submissions(username))
        download_stats["submission_folders"] = submission_count

        # Get all scored submissions using our own helper instead of the imported one
        scored_submissions = {"scored_submissions": {}}
        for username in users:
            scored_submissions["scored_submissions"][username] = {}
            submissions = get_user_submissions(username)

            for submission_id in submissions:
                score_dir = find_score_directory(username, submission_id)
                if not score_dir:
                    continue

                # Get model CSV files
                model_csvs = list(score_dir.glob("*.csv"))
                if not model_csvs:
                    continue

                # Check for corresponding submission file
                submission_csv = find_submission_csv(username, submission_id)
                if submission_csv:
                    scored_models = [csv.stem for csv in model_csvs]
                    scored_submissions["scored_submissions"][username][submission_id] = {
                        "models": scored_models,
                        "timestamp": submission_id,  # Use submission_id as timestamp
                    }

        # Get leaderboard data
        leaderboards = get_all_leaderboards()

        return templates.TemplateResponse(
            "download.html",
            {
                "request": request,
                "download_stats": download_stats,
                "scored_submissions": scored_submissions,
                "download_error": download_error,
                "leaderboard_results": leaderboard_results,
                "leaderboards": leaderboards,
            },
        )
    except Exception as e:
        logger.exception("Error downloading data")
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error downloading data: {str(e)}"})


@app.get("/users", response_class=HTMLResponse)
async def list_users(request: Request):
    """List all users, their submission counts, and their best submission score."""
    try:
        # Check if we have any data
        if not SCORES_DIR.exists():
            return templates.TemplateResponse(
                "error.html", {"request": request, "message": "No score data available. Please download the data first."}
            )

        # Get all users with submissions
        users = get_all_user_directories()

        user_stats = {}
        for user in users:
            user_submissions_dir = SCORES_SUBDIR / user
            try:
                submissions = [d.name for d in user_submissions_dir.iterdir() if d.is_dir() and d.name != ".DS_Store"]
                submission_count = len(submissions)
                best_score = -1
                best_submission_id = None

                for submission_id in submissions:
                    score_dir = find_score_directory(user, submission_id)
                    if score_dir:
                        summary_path = score_dir / "submission_summary.csv"
                        if summary_path.exists():
                            try:
                                summary_df = pd.read_csv(summary_path)
                                average_row = summary_df[summary_df["model_name"].str.lower() == "average"]
                                if not average_row.empty and "overall" in average_row.columns:
                                    current_score = average_row.iloc[0]["overall"]
                                    if pd.notna(current_score) and current_score > best_score:
                                        best_score = current_score
                                        best_submission_id = submission_id
                            except Exception as e:
                                logger.error(f"Error processing summary {summary_path} for user {user}: {e}")

                user_stats[user] = {
                    "count": submission_count,
                    "best_submission_id": best_submission_id,
                    "best_score": best_score if best_submission_id else None,
                }
            except FileNotFoundError:
                logger.warning(f"Submissions directory not found for user {user}: {user_submissions_dir}")
                user_stats[user] = 0

        return templates.TemplateResponse("users.html", {"request": request, "users": users, "user_stats": user_stats})
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error listing users: {str(e)}"})


@app.get("/users/{username}", response_class=HTMLResponse)
async def list_submissions(request: Request, username: str):
    """List all submissions for a specific user"""
    try:
        # Get all submissions for the user
        submissions = get_user_submissions(username)
        if not submissions:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found or has no submissions")

        submissions_list = []
        for submission_id in submissions:
            score_dir = find_score_directory(username, submission_id)
            models = []
            submission_scores = {}
            if score_dir:
                # Get model names from CSV files, excluding submission_summary.csv
                model_csvs = [f for f in score_dir.glob("*.csv") if f.name != "submission_summary.csv"]
                models = [f.stem for f in model_csvs]

                # Try to read all average scores from submission_summary.csv
                summary_path = score_dir / "submission_summary.csv"
                if summary_path.exists():
                    try:
                        summary_df = pd.read_csv(summary_path)
                        if not summary_df.empty:
                            # Find the row where model_name is 'average'
                            average_row = summary_df[summary_df["model_name"].str.lower() == "average"]
                            if not average_row.empty:
                                # Define expected score columns
                                score_cols = ["grammar", "coherence", "creativity", "consistency", "plot", "overall"]
                                # Extract scores from these columns in the average row
                                submission_scores = average_row.iloc[0][score_cols].to_dict()
                                # Filter out any potential NaN values
                                submission_scores = {k: v for k, v in submission_scores.items() if pd.notna(v)}
                                logger.debug(f"Extracted average scores for {submission_id}: {submission_scores}")
                            else:
                                logger.warning(f"No row with model_name='average' found in {summary_path}")
                    except Exception as e:
                        logger.error(f"Error reading summary file {summary_path}: {e}")

            submissions_list.append({"id": submission_id, "models": models, "scores": submission_scores})

        category_set = set(cat for sub in submissions_list for cat in sub["scores"].keys())
        all_categories = sorted(list(category_set))

        # Ensure 'overall' is the first category if it exists
        if "overall" in all_categories:
            all_categories.remove("overall")
            all_categories.insert(0, "overall")

        return templates.TemplateResponse(
            "user_submissions.html",
            {"request": request, "username": username, "submissions": submissions_list, "categories": all_categories},
        )
    except HTTPException:
        raise
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


@app.get("/users/{username}/{submission_id}", response_class=HTMLResponse)
async def view_submission(
    request: Request, 
    username: str, 
    submission_id: str, 
    sort: str = "id", 
    sort_category: str = "overall", 
    page: int = 1, 
    items_per_page: int = 24,
    search: str = "",
    item_id: Optional[int] = None
):
    """View a specific submission with scores from different models and detailed comparison"""
    try:
        # Find score directory
        submission_dir = find_score_directory(username, submission_id)
        logger.info(f"Looking for submission scores at: {submission_dir}")
        if not submission_dir:
            logger.error(f"Submission directory not found for {username}/{submission_id}")

            # Desperate attempt: try directly accessing the path without checking if files exist first
            potential_dirs = [SCORES_DIR / username / submission_id, SCORES_SUBDIR / username / submission_id]

            logger.info(f"Last resort attempt - checking these paths: {potential_dirs}")
            for path in potential_dirs:
                if path.exists() and path.is_dir():
                    logger.info(f"Found directory at {path}, trying to use it directly")
                    submission_dir = path
                    break

            if not submission_dir:
                raise HTTPException(status_code=404, detail=f"Submission '{submission_id}' not found")

        # Get the submission CSV
        submission_csv_path = find_submission_csv(username, submission_id)
        logger.info(f"Looking for submission CSV at: {submission_csv_path}")
        if not submission_csv_path:
            logger.error(f"Submission CSV not found for {username}/{submission_id}")
            raise HTTPException(status_code=404, detail=f"Submission file not found")

        # Get all model CSVs - explicitly exclude submission_summary.csv and score_history.csv
        model_csvs = []
        csv_files = list(submission_dir.glob("*.csv"))
        logger.info(f"All CSV files in directory: {[csv.name for csv in csv_files]}")

        for csv_file in csv_files:
            if csv_file.name not in ["submission_summary.csv", "score_history.csv"]:
                model_csvs.append(csv_file)

        logger.info(f"Found {len(model_csvs)} model CSV files: {[csv.name for csv in model_csvs]}")

        if not model_csvs:
            logger.warning(f"No model CSV files found in {submission_dir}!")
            # Try one more desperate attempt - list all files in directory
            logger.info(f"All files in directory: {list(submission_dir.iterdir())}")

        # Load or create the processed DataFrame
        df = get_cached_or_load_data(username, submission_id)

        if df.empty:
            raise HTTPException(status_code=500, detail="Failed to load or create processed data.")

        # Calculate averages for the entire submission (these stay constant regardless of which item is viewed)
        model_averages, submission_averages, avg_scores_per_item, categories = calculate_dataframe_averages(df)

        # Get all story IDs
        all_story_ids = sorted(df["item_id"].unique())
        total_items = len(all_story_ids)
        
        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            filtered_story_ids = []
            for sid in all_story_ids:
                item_data = df[df["item_id"] == sid]
                if not item_data.empty:
                    prompt = str(item_data.iloc[0].get("prompt", "")).lower()
                    completion = str(item_data.iloc[0].get("completion", "")).lower()
                    # Check if search term is in prompt, completion, or item_id
                    if (search_lower in prompt or 
                        search_lower in completion or 
                        search in str(sid)):
                        filtered_story_ids.append(sid)
            all_story_ids = filtered_story_ids
            total_items = len(all_story_ids)
        
        # Apply sorting
        if sort == "score":
            # Sort by the specified score category
            if sort_category in categories:
                # Create a dictionary mapping item_id to the score for the specified category
                category_scores = {}
                for sid in all_story_ids:
                    item_data = df[df["item_id"] == sid]
                    if not item_data.empty:
                        category_scores[sid] = item_data[sort_category].mean() if sort_category in item_data.columns else 0
                all_story_ids = sorted(all_story_ids, key=lambda sid: category_scores.get(sid, 0), reverse=True)
            else:
                # Default to overall score if category not found
                all_story_ids = sorted(all_story_ids, key=lambda sid: avg_scores_per_item.get(sid, 0), reverse=True)
        elif sort == "id":
            # Already sorted by default
            pass
        
        # Calculate pagination
        total_pages = (total_items + items_per_page - 1) // items_per_page  # Ceiling division
        page = max(1, min(page, total_pages))  # Ensure page is within bounds
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        # Get current page's story IDs
        story_ids = all_story_ids[start_idx:end_idx]
        
        # Determine the current item to display
        current_item = None
        if item_id is not None and item_id in all_story_ids:
            current_item = item_id
            # If the current item is not on the current page, adjust the page
            if item_id not in story_ids:
                item_index = all_story_ids.index(item_id)
                page = (item_index // items_per_page) + 1
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_items)
                story_ids = all_story_ids[start_idx:end_idx]
            logger.info(f"Using item_id {item_id} from URL parameter as current_item.")
        else:
            if item_id is not None:
                logger.warning(f"URL item_id {item_id} not found in available IDs {all_story_ids}. Defaulting to first item.")
            else:
                logger.info("No item_id in URL. Defaulting to first item.")
            current_item = story_ids[0] if story_ids else None

        logger.info(f"Final determined current_item: {current_item}")

        # Calculate prev/next items for navigation
        if current_item is not None:
            current_index = all_story_ids.index(current_item)
            prev_item = all_story_ids[current_index - 1] if current_index > 0 else None
            next_item = all_story_ids[current_index + 1] if current_index < len(all_story_ids) - 1 else None
        else:
            prev_item = None
            next_item = None

        # Get scores for the current item only (this is separate from the overall averages)
        item_data = df[df["item_id"] == current_item] if current_item is not None else pd.DataFrame()
        prompt = ""
        completion = ""
        evaluations = []  # Initialize evaluations outside the block
        item_averages = {}

        # Prepare evaluations (scores per model for this item)
        if not item_data.empty:
            # Get prompt/completion from the first row (should be same across models for the same item_id)
            first_row = item_data.iloc[0]
            prompt = first_row.get("prompt", "Prompt not found")
            completion = first_row.get("completion", "Completion not found")

            # Process each row (each model's data for this item_id)
            for record in item_data.to_dict("records"):
                # Ensure 'scores' is extracted correctly based on the expanded columns
                # Need to select the category columns from the record
                scores = {cat: record.get(cat) for cat in categories if pd.notna(record.get(cat))}
                evaluations.append({"model_name": record.get("model_name", "Unknown Model"), "scores": scores})

            # Calculate averages for this specific item
            item_averages = {}
            # Calculate average for each category across all models for this item
            for category in categories:
                category_scores = [eval["scores"].get(category) for eval in evaluations if category in eval["scores"]]
                if category_scores:
                    item_averages[category] = sum(category_scores) / len(category_scores)
            
            # For the overall score, directly average the overall scores from each model
            overall_scores = []
            for eval in evaluations:
                # Calculate overall score for this model (average of categories)
                if eval["scores"]:
                    model_overall = sum(eval["scores"].values()) / len(eval["scores"])
                    overall_scores.append(model_overall)
            
            if overall_scores:
                item_averages["overall"] = sum(overall_scores) / len(overall_scores)
        else:
            if current_item is not None:
                logger.error(f"Data for item_id {current_item} not found in DataFrame!")
                prompt = f"Error: Data for item {current_item} not found."
            else:
                logger.error("No items found matching the criteria!")
                prompt = "Error: No items found matching the criteria!"
            completion = ""
            # evaluations remains empty as initialized
            item_averages = {}

        logger.info(f"Debugging view_submission for item {current_item}:")
        logger.info(f"  Categories passed to template: {categories}")
        logger.info(f"  Evaluations passed to template: {evaluations}")
        logger.info(f"  Item averages: {item_averages}")

        return templates.TemplateResponse(
            "new_submission.html",
            {
                "request": request,
                "username": username,
                "submission_id": submission_id,
                "story_ids": story_ids,
                "all_story_ids": all_story_ids,
                "avg_scores": avg_scores_per_item,
                "current_sort": sort,
                "current_sort_category": sort_category,
                "current_page": page,
                "total_pages": total_pages,
                "items_per_page": items_per_page,
                "total_items": total_items,
                "search": search,
                "current_item": current_item,
                "prev_item": prev_item,
                "next_item": next_item,
                "prompt": prompt,
                "completion": completion,
                "evaluations": evaluations,
                "categories": categories,
                "model_averages": model_averages,
                "submission_averages": submission_averages,
                "item_averages": item_averages,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in view_submission: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


@app.get("/compare/{username}/{submission_id}/{item_id}", response_class=HTMLResponse)
async def compare_scores(request: Request, username: str, submission_id: str, item_id: int):
    """Compare scores from different models for a specific item"""
    try:
        # Find score directory
        submission_dir = find_score_directory(username, submission_id)
        logger.info(f"Looking for submission scores at: {submission_dir}")
        if not submission_dir:
            logger.error(f"Submission directory not found for {username}/{submission_id}")

            # Desperate attempt: try directly accessing the path without checking if files exist first
            potential_dirs = [SCORES_DIR / username / submission_id, SCORES_SUBDIR / username / submission_id]

            logger.info(f"Last resort attempt - checking these paths: {potential_dirs}")
            for path in potential_dirs:
                if path.exists() and path.is_dir():
                    logger.info(f"Found directory at {path}, trying to use it directly")
                    submission_dir = path
                    break

            if not submission_dir:
                raise HTTPException(status_code=404, detail=f"Submission '{submission_id}' not found")

        # Get the submission CSV
        submission_csv_path = find_submission_csv(username, submission_id)
        logger.info(f"Looking for submission CSV at: {submission_csv_path}")
        if not submission_csv_path:
            logger.error(f"Submission CSV not found for {username}/{submission_id}")
            raise HTTPException(status_code=404, detail=f"Submission file not found")

        # Get all model CSVs - explicitly exclude summary and history files
        model_csvs = []
        csv_files = list(submission_dir.glob("*.csv"))
        logger.info(f"All CSV files in directory: {[csv.name for csv in csv_files]}")

        for csv_file in csv_files:
            if csv_file.name not in ["submission_summary.csv", "score_history.csv"]:
                model_csvs.append(csv_file)

        logger.info(f"Found {len(model_csvs)} model CSV files: {[csv.name for csv in model_csvs]}")

        # Read submission data
        submissions_df = pd.read_csv(submission_csv_path)

        # Ensure we have the item_id in range
        if item_id >= len(submissions_df):
            raise HTTPException(status_code=404, detail=f"Item ID {item_id} not found in submission (max index: {len(submissions_df) - 1})")

        # Get the prompt and completion from the submission
        story_prompt = ""
        story_completion = ""

        if "prompt" in submissions_df.columns:
            story_prompt = submissions_df.iloc[item_id]["prompt"]

        if "completion" in submissions_df.columns:
            story_completion = submissions_df.iloc[item_id]["completion"]

        # Collect evaluations for the specific story across all models
        model_evaluations = []

        for model_csv in model_csvs:
            model_name = model_csv.stem
            logger.info(f"Processing scores for comparison from model: {model_name}")

            # Read score data
            try:
                scores_df = pd.read_csv(model_csv)
                logger.info(f"Score file for {model_name} has {len(scores_df)} entries")

                # Make sure we have item_id column or add it
                if "item_id" not in scores_df.columns:
                    scores_df["item_id"] = scores_df.index

                # Find the row with matching item_id
                matching_rows = scores_df[scores_df["item_id"] == item_id]

                if not matching_rows.empty:
                    score_row = matching_rows.iloc[0]
                    logger.info(f"Found matching score row for item_id {item_id}")
                elif item_id < len(scores_df):
                    # Fallback to index
                    score_row = scores_df.iloc[item_id]
                    logger.info(f"Using index {item_id} to find score row")
                else:
                    logger.warning(f"Item ID {item_id} not found in scores for model {model_name}")
                    continue

                # Extract scores
                scores = {}
                for score_category in ScoreCategory.__members__:
                    category_lower = score_category.lower()
                    if category_lower in score_row:
                        score_value = score_row[category_lower]
                        # Convert numpy types to Python native types
                        if hasattr(score_value, "item"):
                            score_value = score_value.item()
                        scores[category_lower] = score_value

                logger.info(f"Found scores for model {model_name}: {scores}")
                model_evaluations.append({"model_name": model_name, "scores": scores})

            except Exception as e:
                logger.exception(f"Error processing scores for model {model_name}: {e}")

        if not model_evaluations:
            raise HTTPException(status_code=404, detail=f"No evaluations found for item ID {item_id}")

        logger.info(f"Returning {len(model_evaluations)} model evaluations for compare view")

        return templates.TemplateResponse(
            "compare_scores.html",
            {
                "request": request,
                "username": username,
                "submission_id": submission_id,
                "item_id": item_id,
                "prompt": story_prompt,
                "completion": story_completion,
                "evaluations": model_evaluations,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in compare_scores: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


def load_and_merge_submission_data(username: str, submission_id: str) -> pd.DataFrame:
    """Loads submission and score CSVs, merges them into a single DataFrame with numeric scores."""
    logger.info(f"Performing full CSV load and merge for {username}/{submission_id}")
    submission_csv_path = find_submission_csv(username, submission_id)
    score_dir = find_score_directory(username, submission_id)

    # Read submission CSV
    submissions_df = pd.read_csv(submission_csv_path)

    # Get all model CSVs - explicitly exclude submission_summary.csv and score_history.csv
    model_csvs = []
    csv_files = list(score_dir.glob("*.csv"))
    logger.info(f"All CSV files in directory: {[csv.name for csv in csv_files]}")

    for csv_file in csv_files:
        if csv_file.name not in ["submission_summary.csv", "score_history.csv"]:
            model_csvs.append(csv_file)

    logger.info(f"Found {len(model_csvs)} model CSV files: {[csv.name for csv in model_csvs]}")

    if not model_csvs:
        logger.warning(f"No model CSV files found in {score_dir}!")
        # Try one more desperate attempt - list all files in directory
        logger.info(f"All files in directory: {list(score_dir.iterdir())}")

    # Merge data
    merged_data = []
    for model_csv in model_csvs:
        model_name = model_csv.stem
        logger.info(f"Processing scores for model: {model_name}")

        # Read score CSV
        scores_df = pd.read_csv(model_csv)
        logger.info(f"Score file for {model_name} has {len(scores_df)} entries")

        # Make sure we have item_id column or add it
        if "item_id" not in scores_df.columns:
            scores_df["item_id"] = scores_df.index

        # Merge based on item_id
        for index, row in scores_df.iterrows():
            item_id = int(row["item_id"])
            logger.info(f"Processing score for item_id: {item_id}")

            submission_row = None

            # First try to match by item_id if available in submissions
            if "item_id" in submissions_df.columns:
                # Force item_id to be integer in submissions_df too
                submissions_df["item_id"] = submissions_df["item_id"].astype(int)
                matching_rows = submissions_df[submissions_df["item_id"] == item_id]
                if not matching_rows.empty:
                    submission_row = matching_rows.iloc[0]
                    logger.info(f"Found matching submission row by item_id")

            # If no match found by item_id, use the index as a fallback
            if submission_row is None:
                # If the item_id is a valid index in the submissions dataframe
                if item_id < len(submissions_df):
                    submission_row = submissions_df.iloc[item_id]
                    logger.info(f"Using item_id {item_id} as index to match with submission")
                # As last resort, use the row index from scores dataframe
                elif index < len(submissions_df):
                    submission_row = submissions_df.iloc[index]
                    logger.info(f"Using row index {index} to match with submission as last resort")
                else:
                    logger.warning(f"No matching submission found for score item_id {item_id}")

            if submission_row is not None:
                scores = {}
                # Extract available score categories
                for score_category in ScoreCategory.__members__:
                    category_lower = score_category.lower()
                    if category_lower in row:
                        score_value = row[category_lower]
                        # Convert numpy types to Python native types for better JSON serialization
                        if hasattr(score_value, "item"):
                            score_value = score_value.item()
                        scores[category_lower] = score_value

                # Log available scores
                logger.info(f"Found scores: {scores}")

                prompt = ""
                completion = ""

                # Extract prompt and completion, handling both string and object access
                if "prompt" in submission_row:
                    prompt = submission_row["prompt"]
                    if isinstance(prompt, pd.Series):
                        prompt = prompt.iloc[0] if not prompt.empty else ""

                if "completion" in submission_row:
                    completion = submission_row["completion"]
                    if isinstance(completion, pd.Series):
                        completion = completion.iloc[0] if not completion.empty else ""

                merged_item = {
                    "item_id": item_id,
                    "prompt": prompt,
                    "completion": completion,
                    "scores": scores,
                    "model_name": model_name,
                }
                merged_data.append(merged_item)
                logger.info(f"Added merged item for item_id {item_id} with {len(scores)} scores")
            else:
                logger.warning(f"No matching submission found for score item_id {item_id}")

        logger.info(f"Total merged items for model {model_name}: {len(merged_data)}")

    # Convert merged data to DataFrame
    df = pd.DataFrame(merged_data)

    if df.empty:
        logger.warning(f"Merged DataFrame is empty for {username}/{submission_id}")
        return df

    # Expand the 'scores' dictionary into separate columns
    if "scores" in df.columns:
        scores_expanded = df["scores"].apply(pd.Series)
        df = pd.concat([df.drop(["scores"], axis=1), scores_expanded], axis=1)

        # Convert potential score columns to numeric, coercing errors to NaN
        potential_score_cols = scores_expanded.columns
        for col in potential_score_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Created final merged DataFrame with {len(df)} total rows (items * models). Columns: {df.columns.tolist()}")

    return df


def get_cached_or_load_data(username: str, submission_id: str) -> pd.DataFrame:
    """Retrieves the merged DataFrame from cache or loads it if not present."""
    cache_key = (username, submission_id)
    if cache_key in submission_data_cache:
        logger.info(f"Returning cached data for {username}/{submission_id}")
        return submission_data_cache[cache_key]
    else:
        logger.info(f"No cache found for {username}/{submission_id}. Loading from CSVs.")
        df = load_and_merge_submission_data(username, submission_id)
        submission_data_cache[cache_key] = df  # Store in cache
        return df


def calculate_dataframe_averages(df):
    """Calculate model and submission averages from the merged DataFrame."""
    # Explicitly define potential score categories or try to infer safely
    potential_score_cols = ["grammar", "coherence", "creativity", "consistency", "plot", "overall"]
    categories = [col for col in potential_score_cols if col in df.columns]
    if not categories:
        logger.warning("No score categories found in DataFrame columns.")
        # Fallback: try inferring numeric columns excluding known non-score ones
        excluded_cols = ["item_id", "prompt", "completion", "model_name", "original_row_index"]
        categories = [col for col in df.columns if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])]
        logger.info(f"Inferred numeric categories: {categories}")

    model_averages = {}
    submission_averages = {cat: [] for cat in categories}
    avg_scores_per_item = {}

    for model_name in df["model_name"].unique():
        model_df = df[df["model_name"] == model_name]
        model_averages[model_name] = {}
        for category in categories:
            scores = model_df[category]
            model_averages[model_name][category] = scores.mean()

        # Calculate overall average for this model
        all_scores = []
        for category in categories:
            all_scores.append(model_averages[model_name][category])

        if all_scores:
            model_averages[model_name]["overall"] = sum(all_scores) / len(all_scores)

    # Collect all scores for submission averages
    for index, row in df.iterrows():
        for category in categories:
            submission_averages[category].append(row[category])

    # Calculate item-specific scores
    item_scores = {}
    for item_id in df["item_id"].unique():
        item_df = df[df["item_id"] == item_id]
        item_scores[item_id] = {}
        
        # Calculate average for each category across all models for this item
        for category in categories:
            item_scores[item_id][category] = item_df[category].mean()
        
        # For each model, calculate its overall score for this item
        model_overall_scores = []
        for model_name in item_df["model_name"].unique():
            model_item_df = item_df[item_df["model_name"] == model_name]
            if not model_item_df.empty:
                # Get all category scores for this model and item
                model_item_scores = [model_item_df[cat].iloc[0] for cat in categories if pd.notna(model_item_df[cat].iloc[0])]
                if model_item_scores:
                    # Calculate overall score for this model (average of all categories)
                    model_overall = sum(model_item_scores) / len(model_item_scores)
                    model_overall_scores.append(model_overall)
        
        # Average the overall scores across models
        if model_overall_scores:
            avg_scores_per_item[item_id] = sum(model_overall_scores) / len(model_overall_scores)
        else:
            avg_scores_per_item[item_id] = 0  # Fallback

    # Calculate submission averages
    for category in categories:
        if submission_averages[category]:
            submission_averages[category] = sum(submission_averages[category]) / len(submission_averages[category])
        else:
            del submission_averages[category]

    # Calculate overall submission average
    all_submission_scores = []
    for category in categories:
        if category in submission_averages:
            all_submission_scores.append(submission_averages[category])

    if all_submission_scores:
        submission_averages["overall"] = sum(all_submission_scores) / len(all_submission_scores)

    return model_averages, submission_averages, avg_scores_per_item, categories


def read_leaderboard_file(filename):
    """Read a leaderboard CSV file into a pandas DataFrame.

    Args:
        filename: Name of the leaderboard file (e.g., 'leaderboard_global.csv')

    Returns:
        DataFrame containing the leaderboard data, or None if file doesn't exist
    """
    filepath = DATA_DIR / filename
    if not filepath.exists():
        logger.warning(f"Leaderboard file not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully read leaderboard file: {filename} with {len(df)} entries")
        return df
    except Exception as e:
        logger.error(f"Error reading leaderboard file {filename}: {e}")
        return None


def get_all_leaderboards():
    """Get all available leaderboard data.

    Returns:
        Dictionary mapping leaderboard names to DataFrames
    """
    leaderboards = {}

    # List of leaderboard files to check
    leaderboard_files = ["leaderboard_global.csv", "leaderboard_large.csv", "leaderboard_medium.csv", "leaderboard_small.csv"]

    for filename in leaderboard_files:
        df = read_leaderboard_file(filename)
        if df is not None:
            # Remove .csv extension and leaderboard_ prefix for the key
            key = filename.replace(".csv", "").replace("leaderboard_", "")
            leaderboards[key] = df

    return leaderboards


@app.get("/leaderboard/{board_type}", response_class=HTMLResponse)
async def view_leaderboard(request: Request, board_type: str):
    """View a specific leaderboard"""
    leaderboards = get_all_leaderboards()

    if board_type not in leaderboards:
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Leaderboard '{board_type}' not found"})

    leaderboard_data = leaderboards[board_type]

    return templates.TemplateResponse(
        "leaderboard.html",
        {
            "request": request,
            "board_type": board_type,
            "leaderboard_data": leaderboard_data.to_dict("records"),
            "leaderboards": leaderboards,
        },
    )


@app.get("/leaderboards", response_class=HTMLResponse)
async def view_all_leaderboards(request: Request):
    """View all leaderboards on a single page"""
    leaderboards_dict = get_all_leaderboards()

    # Convert all DataFrames to records for template rendering
    leaderboards = {}
    for board_type, df in leaderboards_dict.items():
        leaderboards[board_type] = df.to_dict("records")

    return templates.TemplateResponse("leaderboards.html", {"request": request, "leaderboards": leaderboards})


def download_leaderboards(env_config: EnvConfig, api_client=None) -> Dict[str, bool]:
    """Download leaderboard CSV files from the root of the scores repository.

    Args:
        env_config: Environment configuration
        api_client: Optional authenticated API client

    Returns:
        Dictionary with status of each leaderboard file download
    """
    if api_client is None:
        logger.warning("No API client available - cannot download leaderboard files")
        return {}

    # Leaderboard files to look for
    leaderboard_files = ["leaderboard_global.csv", "leaderboard_large.csv", "leaderboard_medium.csv", "leaderboard_small.csv"]

    results = {}

    try:
        # List files in the repository
        repo_files = api_client.list_repo_files(repo_id=env_config.scores_repo_id, repo_type="dataset")

        # Filter for leaderboard files at the root level
        for leaderboard_file in leaderboard_files:
            if leaderboard_file in repo_files:
                logger.info(f"Found leaderboard file: {leaderboard_file}")
                try:
                    # Download to the DATA_DIR
                    api_client.hf_hub_download(
                        repo_id=env_config.scores_repo_id, repo_type="dataset", filename=leaderboard_file, local_dir=DATA_DIR
                    )
                    results[leaderboard_file] = True
                    logger.info(f"Downloaded leaderboard file: {leaderboard_file}")
                except Exception as e:
                    logger.error(f"Error downloading leaderboard file {leaderboard_file}: {e}")
                    results[leaderboard_file] = False
            else:
                logger.info(f"Leaderboard file not found: {leaderboard_file}")
                results[leaderboard_file] = False
    except Exception as e:
        logger.error(f"Error listing repository files: {e}")

    return results
