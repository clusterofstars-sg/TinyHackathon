import uvicorn
import sys
import os
import argparse
from pathlib import Path
import tinyhackathon.score_explorer.app as app_module

# Add the parent directories to the Python path
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))

# Also add the tinyhackathon directory to fix possible import issues
tinyhackathon_dir = Path(__file__).parent.parent
sys.path.append(str(tinyhackathon_dir))

# Parse command-line arguments for dataset overrides and server config
parser = argparse.ArgumentParser(description="Run Score Explorer webapp")
parser.add_argument("--submissions-dataset", help="HF repo for submissions")
parser.add_argument("--scores-dataset", help="HF repo for scores")
parser.add_argument("--host", default="0.0.0.0", help="Server host")
parser.add_argument("--port", type=int, default=8001, help="Server port")
parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
args = parser.parse_args()

# Override defaults in app module
app_module.DEFAULT_SUBMISSIONS_DATASET = args.submissions_dataset
app_module.DEFAULT_SCORES_DATASET     = args.scores_dataset

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run("tinyhackathon.score_explorer.app:app", host=args.host, port=args.port, reload=args.reload)
