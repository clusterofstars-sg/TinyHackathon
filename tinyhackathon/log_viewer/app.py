from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any

app = FastAPI(title="Log Viewer", description="A simple app to view logs in a human-readable format")

# Setup templates - using relative path from the current file
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Base directory for logs - go one level up to find logs directory
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


def extract_story(prompt):
    """Extract the story content from the prompt"""
    if not prompt:
        return None

    # Use regex to find content between <story> tags
    story_match = re.search(r"<story>(.*?)</story>", prompt, re.DOTALL)
    if story_match:
        story_content = story_match.group(1).strip()

        # Check for the separator first
        has_separator = "***" in story_content
        separator_html = "<span class='separator' style='font-weight: bold; padding: 2px 5px;'>***</span>"

        if has_separator:
            # Replace the separator first to avoid splitting formatted paragraphs
            story_content = story_content.replace("***", separator_html)

        # Format paragraphs - wrap each line in a paragraph tag for better spacing
        lines = story_content.split("\n")
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                formatted_lines.append(f"<p style='margin-bottom: 1em;'>{line}</p>")

        story_content = "".join(formatted_lines)

        return story_content
    return None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with list of available log directories"""
    try:
        # Get list of log directories (usernames)
        if not LOGS_DIR.exists():
            return templates.TemplateResponse("error.html", {"request": request, "message": f"Logs directory '{LOGS_DIR}' not found"})

        users = [d.name for d in LOGS_DIR.iterdir() if d.is_dir()]

        return templates.TemplateResponse("index.html", {"request": request, "users": users})
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


@app.get("/logs/{username}", response_class=HTMLResponse)
async def user_logs(request: Request, username: str):
    """List all logs for a specific user"""
    try:
        user_dir = LOGS_DIR / username
        if not user_dir.exists() or not user_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")

        log_files = [f.name for f in user_dir.iterdir() if f.is_file() and f.suffix == ".json"]

        return templates.TemplateResponse("user_logs.html", {"request": request, "username": username, "log_files": log_files})
    except HTTPException:
        raise
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


@app.get("/logs/{username}/{log_file}", response_class=HTMLResponse)
async def view_log(request: Request, username: str, log_file: str):
    """View a specific log file in a human-readable format"""
    try:
        log_path = LOGS_DIR / username / log_file
        if not log_path.exists() or not log_path.is_file():
            raise HTTPException(status_code=404, detail=f"Log file '{log_file}' not found")

        # Read and parse the log file
        with open(log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        # Extract model data
        model_data = {}
        for model_name, model_info in log_data.items():
            timestamp = model_info.get("timestamp", "Unknown")
            username = model_info.get("username", "Unknown")
            submission_id = model_info.get("submission_id", "Unknown")
            evaluations = model_info.get("evaluations", [])

            # Extract stories from each evaluation prompt
            for evaluation in evaluations:
                evaluation["story"] = extract_story(evaluation.get("prompt", ""))

            model_data[model_name] = {
                "timestamp": timestamp,
                "username": username,
                "submission_id": submission_id,
                "evaluations": evaluations,
            }

        return templates.TemplateResponse(
            "view_log.html", {"request": request, "log_file": log_file, "username": username, "model_data": model_data}
        )
    except HTTPException:
        raise
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error: {str(e)}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
