import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import HfApi

# Import data repository functions
from tinyhackathon.score_explorer.data_repository import (
    check_data_directory_structure,
    compute_user_stats,
    get_all_leaderboards,
    get_cached_submission_data,
    get_item_data,
    get_unique_item_ids,
    list_submissions,
    list_users,
    apply_search,
    calculate_dataframe_averages,
    read_leaderboard_file,
    find_score_directory,
    find_submission_csv,
    DATA_DIR,
    SUBMISSIONS_DIR,
    SCORES_DIR,
)

# Add the parent directory to the path to import from scoring.py
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))
tinyhackathon_dir = Path(__file__).parent.parent
sys.path.append(str(tinyhackathon_dir))

# Import necessary functions from scoring.py
from tinyhackathon.scoring import (
    EnvConfig,
    download_submissions_and_scores,
    setup_api_client,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("score_explorer")

app = FastAPI(title="Score Explorer", description="A web app to explore TinyHackathon scores with merged prompts and completions")

# Setup templates - using relative path from the current file
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Check if the data directories have a valid structure for browsing
check_data_directory_structure()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with options to download data and view submissions"""
    try:
        # Check if we have users and submissions to display
        users = list_users()

        # Try to get the global leaderboard
        global_leaderboard = None
        try:
            # Use the same approach as in the leaderboards route
            leaderboards = get_all_leaderboards()
            if "global" in leaderboards:
                # Get top 5 rows of the leaderboard
                leaderboard_df = leaderboards["global"].head(5)

                # Add is_partially_scored flag to each row
                global_leaderboard = []
                for _, row in leaderboard_df.iterrows():
                    row_dict = row.to_dict()

                    # Check if this submission has been partially scored
                    if "username" in row_dict and "submission_id" in row_dict:
                        username = row_dict["username"]
                        submission_id = row_dict["submission_id"]

                        # Get the score directory and check how many model files it has
                        score_dir = find_score_directory(username, submission_id)
                        if score_dir:
                            # Count model CSV files (excluding summary and history files)
                            model_csvs = [
                                csv_file
                                for csv_file in score_dir.glob("*.csv")
                                if csv_file.name not in ["submission_summary.csv", "score_history.csv"]
                            ]
                            row_dict["is_partially_scored"] = len(model_csvs) < 3
                        else:
                            row_dict["is_partially_scored"] = False
                    else:
                        row_dict["is_partially_scored"] = False

                    global_leaderboard.append(row_dict)

                # Debug: Print the keys in the first row
                if global_leaderboard and len(global_leaderboard) > 0:
                    logger.info(f"Global leaderboard keys: {list(global_leaderboard[0].keys())}")
        except Exception as e:
            logger.warning(f"Could not load global leaderboard for home page: {e}")

        return templates.TemplateResponse(
            "home.html", {"request": request, "has_users": len(users) > 0, "global_leaderboard": global_leaderboard}
        )
    except Exception as e:
        logger.exception(f"Error in home: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


@app.get("/download", response_class=HTMLResponse)
async def download_data(request: Request, refresh: bool = False):
    """Download or refresh submission and scoring data"""
    try:
        # Create environment config (non-test mode)
        env_config = EnvConfig(is_test_mode=False)

        # Set up API client - will use API token from env if available
        api_client = setup_api_client(env_config)

        # Check if running with API client
        has_token = api_client is not None

        # Download leaderboard files
        leaderboard_results = download_leaderboards(env_config, api_client, force_refresh=refresh)

        # Process submissions separately
        results = {}
        submission_dir = str(SUBMISSIONS_DIR)
        scores_dir = str(SCORES_DIR)

        # Skip if not forcing refresh and directories exist with content
        if not refresh and SUBMISSIONS_DIR.exists() and any(SUBMISSIONS_DIR.iterdir()):
            results["submissions"] = {"status": "skipped", "message": "Submission directory already exists"}
        else:
            # Download submissions and scores
            try:
                submission_results = download_submissions_and_scores(
                    env_config=env_config, submission_dir=submission_dir, scores_dir=scores_dir, api_client=api_client
                )
                results.update(submission_results)
            except Exception as e:
                logger.exception(f"Error downloading submissions: {e}")
                results["submissions"] = {"status": "error", "message": str(e)}

        # Combine results
        all_results = {**results, **leaderboard_results}

        total_files = len(all_results)
        success_count = sum(1 for result in all_results.values() if isinstance(result, dict) and result.get("status") == "success")
        skipped_count = sum(1 for result in all_results.values() if isinstance(result, dict) and result.get("status") == "skipped")
        error_count = sum(1 for result in all_results.values() if isinstance(result, dict) and result.get("status") == "error")

        # Count submissions, scores, and leaderboards
        submission_count = 0
        leaderboard_count = sum(1 for key in all_results if "leaderboard" in str(key))
        score_count = 0

        # Check actual files on disk rather than just download results
        if SCORES_DIR.exists():
            # We're specifically looking for the CSV files in the submission directories
            # Each submission should have 4 files: 3 model CSVs + 1 summary CSV
            submission_dirs = []

            # Find submission directories
            for user_dir in SCORES_DIR.glob("submissions/*"):
                if user_dir.is_dir():
                    for sub_dir in user_dir.glob("*"):
                        if sub_dir.is_dir():
                            submission_dirs.append(sub_dir)

            # Count CSV files in submission directories
            score_count = sum(len(list(sub_dir.glob("*.csv"))) for sub_dir in submission_dirs)

        if SUBMISSIONS_DIR.exists():
            # Count unique users with submissions
            user_dirs = [d for d in SUBMISSIONS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]

            # Count submissions for each user
            submission_count = 0
            for user_dir in user_dirs:
                submission_files = list(user_dir.glob("*.csv"))
                submission_count += len(submission_files)

        # Check actual leaderboard files
        leaderboard_files = list(DATA_DIR.glob("leaderboard_*.csv"))
        if leaderboard_files:
            leaderboard_count = len(leaderboard_files)

        # Calculate expected number of score files (4 per submission: 3 models + summary)
        expected_scores = submission_count * 4

        # Create download stats for template
        download_stats = {
            "submissions": submission_count,
            "scores": score_count,
            "leaderboards": leaderboard_count,
            "expected_scores": expected_scores,
        }

        logger.info(f"Download complete. Total: {total_files}, Success: {success_count}, Skipped: {skipped_count}, Error: {error_count}")

        return templates.TemplateResponse(
            "download.html",
            {
                "request": request,
                "results": all_results,
                "total_files": total_files,
                "success_count": success_count,
                "skipped_count": skipped_count,
                "error_count": error_count,
                "download_stats": download_stats,
                "needs_token": not has_token,
            },
        )
    except Exception as e:
        logger.exception(f"Error in download_data: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


@app.get("/users", response_class=HTMLResponse)
async def list_all_users(request: Request):
    """List all users with submission counts and best scores"""
    try:
        # Get user statistics from the repository
        user_stats_df = compute_user_stats()

        if user_stats_df.empty:
            return templates.TemplateResponse(
                "users.html", {"request": request, "users": [], "message": "No users found. Please download data first."}
            )

        # Convert DataFrame to list of dictionaries for the template
        users = user_stats_df.to_dict(orient="records")

        return templates.TemplateResponse("users.html", {"request": request, "users": users})
    except Exception as e:
        logger.exception(f"Error in list_all_users: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


@app.get("/users/{username}", response_class=HTMLResponse)
async def list_user_submissions(request: Request, username: str):
    """List all submissions for a specific user"""
    try:
        # Get submissions for the user
        submissions = list_submissions(username)

        if not submissions:
            return templates.TemplateResponse("error.html", {"request": request, "message": f"No submissions found for user {username}"})

        submission_data = []
        scored_count = 0
        unscored_count = 0

        for submission_id in submissions:
            # Check if this submission has score files
            score_dir = find_score_directory(username, submission_id)
            is_scored = score_dir is not None

            if is_scored:
                scored_count += 1
            else:
                unscored_count += 1
                # Check if the submission file exists
                submission_csv = find_submission_csv(username, submission_id)
                if not submission_csv:
                    # Skip this submission if the CSV doesn't exist
                    continue

            # Get data for each submission
            df = get_cached_submission_data(username, submission_id)

            # Get model names if available
            models = []
            if is_scored and score_dir:
                # Get model names directly from CSV files in the score directory
                model_csvs = [
                    csv_file for csv_file in score_dir.glob("*.csv") if csv_file.name not in ["submission_summary.csv", "score_history.csv"]
                ]
                models = [csv_file.stem for csv_file in model_csvs]
            elif not df.empty and "model_name" in df.columns:
                # Fallback to DataFrame if needed
                model_names = df["model_name"].unique().tolist()
                models = [m for m in model_names if m != "No scores available"]

            # Calculate category scores
            category_scores = {}

            if not df.empty:
                # Get all score categories
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                score_categories = [col for col in numeric_cols if col != "item_id"]

                # Calculate averages for each category
                for category in score_categories:
                    if category in df.columns:
                        # Only include real scores, not NaN placeholders
                        if is_scored:
                            # Filter out NaN values before calculating mean
                            valid_scores = df[df[category].notna()][category]
                            if not valid_scores.empty:
                                category_scores[category] = valid_scores.mean()

                # If no scores found, use the calculate_dataframe_averages function
                if not category_scores and "model_name" in df.columns and is_scored:
                    model_averages, _ = calculate_dataframe_averages(df)
                    # Merge all model scores to get average by category
                    if model_averages:
                        all_categories = set()
                        for model_scores in model_averages.values():
                            all_categories.update(model_scores.keys())

                        for category in all_categories:
                            scores = [model_scores.get(category, 0) for model_scores in model_averages.values() if category in model_scores]
                            if scores:
                                category_scores[category] = sum(scores) / len(scores)

            # Calculate overall average if not already present
            if "overall" not in category_scores and category_scores:
                category_scores["overall"] = sum(category_scores.values()) / len(category_scores)

            # For backward compatibility
            avg_score = category_scores.get("overall") if category_scores else None

            submission_data.append(
                {
                    "submission_id": submission_id,
                    "is_scored": is_scored,
                    "average_score": avg_score,
                    "category_scores": category_scores,
                    "models": models,
                }
            )

        # Sort submissions by submission_id (most recent first)
        submission_data = sorted(submission_data, key=lambda x: x["submission_id"], reverse=True)

        return templates.TemplateResponse(
            "submissions.html",
            {
                "request": request,
                "username": username,
                "submissions": submission_data,
                "scored_count": scored_count,
                "unscored_count": unscored_count,
            },
        )
    except Exception as e:
        logger.exception(f"Error in list_user_submissions: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


@app.get("/users/{username}/{submission_id}", response_class=HTMLResponse)
async def view_submission(
    request: Request,
    username: str,
    submission_id: str,
    sort: str = "id",
    sort_category: str = "overall",
    sort_order: str = "asc",
    page: int = 1,
    items_per_page: int = 24,
    search: str = "",
    item_id: Optional[int] = None,
):
    """View a specific submission with scores from different models and detailed comparison"""
    try:
        # Get merged data from repository
        df = get_cached_submission_data(username, submission_id)

        if df.empty:
            return templates.TemplateResponse(
                "error.html", {"request": request, "message": f"No data found for submission {username}/{submission_id}"}
            )

        # Check if this submission has any scores
        has_scores = True
        if "model_name" in df.columns and df["model_name"].unique().tolist() == ["No scores available"]:
            has_scores = False
            logger.warning(f"Submission {username}/{submission_id} has no score files available")

        # Calculate averages for all models and items
        model_averages, avg_scores_per_item = calculate_dataframe_averages(df)

        # Get all available score categories
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categories = [col for col in numeric_cols if col != "item_id"]

        # Get all unique item_ids
        all_story_ids = get_unique_item_ids(df)

        # Apply search filter if provided
        if search:
            filtered_df = apply_search(df, search)
            filtered_story_ids = get_unique_item_ids(filtered_df)
            all_story_ids = filtered_story_ids

        total_items = len(all_story_ids)

        # Apply sorting - only ID sorting is supported
        # Sort by ID (ascending or descending)
        reverse_sort = sort_order == "desc"
        all_story_ids = sorted(all_story_ids, reverse=reverse_sort)

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
        elif story_ids:  # Make sure we have story IDs before setting a default
            current_item = story_ids[0]  # Default to first item on the page

        # Calculate prev/next items for navigation
        prev_item = None
        next_item = None
        if current_item is not None and all_story_ids:
            current_index = all_story_ids.index(current_item)
            # Use explicit None check instead of relying on truthiness
            prev_item = all_story_ids[current_index - 1] if current_index > 0 else None
            next_item = all_story_ids[current_index + 1] if current_index < len(all_story_ids) - 1 else None

        # Get data for current item
        prompt = ""
        completion = ""
        evaluations = []
        item_averages = {}

        if current_item is not None:
            prompt, completion, evaluations, item_averages = get_item_data(df, current_item)

        # Calculate submission averages
        submission_averages = {}
        for category in categories:
            if category in df.columns:
                submission_averages[category] = df[category].mean()

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
                "current_sort_order": sort_order,
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
                "has_scores": has_scores,
            },
        )
    except Exception as e:
        logger.exception(f"Error in view_submission: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


@app.get("/compare/{username}/{submission_id}/{item_id}", response_class=HTMLResponse)
async def compare_scores(request: Request, username: str, submission_id: str, item_id: int):
    """Compare scores from different models for a specific item"""
    try:
        # Get merged data from repository
        df = get_cached_submission_data(username, submission_id)

        if df.empty:
            return templates.TemplateResponse(
                "error.html", {"request": request, "message": f"No data found for submission {username}/{submission_id}"}
            )

        # Get data for the specific item
        prompt, completion, evaluations, item_averages = get_item_data(df, item_id)

        if not evaluations:
            return templates.TemplateResponse("error.html", {"request": request, "message": f"No evaluations found for item {item_id}"})

        # Get all available score categories from the evaluations
        all_categories = set()
        for eval_data in evaluations:
            all_categories.update(eval_data["scores"].keys())

        categories = sorted(list(all_categories))

        # Create data for the radar chart
        chart_data = {"categories": categories, "datasets": []}

        # Generate colors for each model
        colors = ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40"]

        for i, eval_data in enumerate(evaluations):
            model_name = eval_data["model_name"]
            scores = eval_data["scores"]

            # Use modulo to cycle through colors if we have more models than colors
            color = colors[i % len(colors)]

            dataset = {
                "label": model_name,
                "data": [scores.get(cat, 0) for cat in categories],
                "backgroundColor": f"{color}33",  # Add transparency
                "borderColor": color,
                "pointBackgroundColor": color,
                "pointBorderColor": "#fff",
                "pointHoverBackgroundColor": "#fff",
                "pointHoverBorderColor": color,
            }

            chart_data["datasets"].append(dataset)

        return templates.TemplateResponse(
            "compare_scores.html",
            {
                "request": request,
                "username": username,
                "submission_id": submission_id,
                "item_id": item_id,
                "prompt": prompt,
                "completion": completion,
                "evaluations": evaluations,
                "categories": categories,
                "chart_data": chart_data,
                "item_averages": item_averages,
            },
        )
    except Exception as e:
        logger.exception(f"Error in compare_scores: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


@app.get("/leaderboard/{board_type}", response_class=HTMLResponse)
async def view_leaderboard(request: Request, board_type: str):
    """View a specific leaderboard"""
    try:
        # Use repository to get leaderboard data
        leaderboard_df = read_leaderboard_file(f"leaderboard_{board_type}.csv")

        if leaderboard_df is None:
            return templates.TemplateResponse("error.html", {"request": request, "message": f"Leaderboard {board_type} not found"})

        # Add is_partially_scored flag to each row
        leaderboard_data = []
        for _, row in leaderboard_df.iterrows():
            row_dict = row.to_dict()

            # Check if this submission has been partially scored
            if "username" in row_dict and "submission_id" in row_dict:
                username = row_dict["username"]
                submission_id = row_dict["submission_id"]

                # Get the score directory and check how many model files it has
                score_dir = find_score_directory(username, submission_id)
                if score_dir:
                    # Count model CSV files (excluding summary and history files)
                    model_csvs = [
                        csv_file
                        for csv_file in score_dir.glob("*.csv")
                        if csv_file.name not in ["submission_summary.csv", "score_history.csv"]
                    ]
                    row_dict["is_partially_scored"] = len(model_csvs) < 3
                else:
                    row_dict["is_partially_scored"] = False
            else:
                row_dict["is_partially_scored"] = False

            leaderboard_data.append(row_dict)

        # Rename columns for better display in UI
        column_mapping = {
            "submission_count": "Submissions",
            "submission_date": "Date",
            "weight_class": "Class"
        }

        # Apply column renames if needed
        for old_name, new_name in column_mapping.items():
            if old_name in leaderboard_df.columns:
                # Create a new column list with the renamed columns
                columns = []
                for col in leaderboard_df.columns:
                    if col == old_name:
                        columns.append(new_name)
                    else:
                        columns.append(col)
                leaderboard_df.columns = columns

                # Update the column name in each data item
                for item in leaderboard_data:
                    if old_name in item:
                        item[new_name] = item.pop(old_name)

        return templates.TemplateResponse(
            "leaderboard.html",
            {"request": request, "board_type": board_type, "leaderboard": leaderboard_data, "columns": list(leaderboard_df.columns)},
        )
    except Exception as e:
        logger.exception(f"Error in view_leaderboard: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


@app.get("/leaderboards", response_class=HTMLResponse)
async def view_all_leaderboards(request: Request):
    """View all leaderboards on a single page"""
    try:
        # Use repository to get all leaderboard data
        leaderboards = get_all_leaderboards()

        if not leaderboards:
            return templates.TemplateResponse("error.html", {"request": request, "message": "No leaderboards found"})

        # Convert each DataFrame to list of dictionaries and add is_partially_scored flag
        leaderboard_data = {}
        for board_name, df in leaderboards.items():
            processed_data = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()

                # Check if this submission has been partially scored
                if "username" in row_dict and "submission_id" in row_dict:
                    username = row_dict["username"]
                    submission_id = row_dict["submission_id"]

                    # Get the score directory and check how many model files it has
                    score_dir = find_score_directory(username, submission_id)
                    if score_dir:
                        # Count model CSV files (excluding summary and history files)
                        model_csvs = [
                            csv_file
                            for csv_file in score_dir.glob("*.csv")
                            if csv_file.name not in ["submission_summary.csv", "score_history.csv"]
                        ]
                        row_dict["is_partially_scored"] = len(model_csvs) < 3
                    else:
                        row_dict["is_partially_scored"] = False
                else:
                    row_dict["is_partially_scored"] = False

                processed_data.append(row_dict)

            leaderboard_data[board_name] = {"data": processed_data, "columns": list(df.columns) + ["is_partially_scored"]}

            # Rename columns for better display in UI
            column_mapping = {
                "submission_count": "Submissions",
                "submission_date": "Date",
                "weight_class": "Class"
            }

            # Apply column renames if needed
            for old_name, new_name in column_mapping.items():
                if old_name in leaderboard_data[board_name]["columns"]:
                    # Create a new column list with the renamed columns
                    columns = []
                    for col in leaderboard_data[board_name]["columns"]:
                        if col == old_name:
                            columns.append(new_name)
                        else:
                            columns.append(col)
                    leaderboard_data[board_name]["columns"] = columns

                    # Update the column name in each data item
                    for item in leaderboard_data[board_name]["data"]:
                        if old_name in item:
                            item[new_name] = item.pop(old_name)

        return templates.TemplateResponse(
            "all_leaderboards.html",
            {"request": request, "leaderboards": leaderboard_data},
        )
    except Exception as e:
        logger.exception(f"Error in view_all_leaderboards: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


def download_leaderboards(env_config: EnvConfig, api_client: Optional[HfApi] = None, force_refresh: bool = False):
    """Download leaderboard CSV files from the root of the scores repository."""
    try:
        leaderboard_files = ["leaderboard_global.csv", "leaderboard_large.csv", "leaderboard_small.csv", "leaderboard_medium.csv"]
        results = {}

        for leaderboard_file in leaderboard_files:
            try:
                # Set up paths
                output_path = DATA_DIR / leaderboard_file
                target_path = Path(output_path)

                # Create directory if it doesn't exist
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Skip download if file exists and not forcing refresh
                if target_path.exists() and not force_refresh:
                    logger.info("Skipping " + leaderboard_file + " (already exists)")
                    results[leaderboard_file] = {"status": "skipped", "message": "File already exists"}
                    continue

                # Download file from the repository
                api_client.hf_hub_download(
                    repo_id=env_config.scores_repo_id, filename=leaderboard_file, local_dir=str(DATA_DIR), repo_type="dataset"
                )

                logger.info(f"Downloaded {leaderboard_file} to {str(output_path)}")
                results[leaderboard_file] = {"status": "success", "message": "Downloaded to " + str(output_path)}

            except Exception as e:
                logger.exception(f"Error downloading {leaderboard_file}: {e}")
                results[leaderboard_file] = {"status": "error", "message": f"Error downloading {leaderboard_file}: {str(e)}"}

        return results
    except Exception as e:
        logger.exception("Error in download_leaderboards: " + str(e))
        return {"leaderboards": {"status": "error", "message": str(e)}}
