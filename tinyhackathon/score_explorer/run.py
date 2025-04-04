import uvicorn
import sys
import os
from pathlib import Path

# Add the parent directories to the Python path
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))

# Also add the tinyhackathon directory to fix possible import issues
tinyhackathon_dir = Path(__file__).parent.parent
sys.path.append(str(tinyhackathon_dir))

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
