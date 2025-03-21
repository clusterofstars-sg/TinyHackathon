 ## Setup

 ```bash
# swap cuda-toolkit for cuda if you want to compile cuda packages
conda create -n tinyhackathon python=3.12 uv cuda-toolkit -c nvidia/label/cuda-12.4.1 -c conda-forge
conda activate tinyhackathon
# This sets uv to use the active Conda environment whether using uv or uv pip commands.
# You'll need to run this command every time you open a new terminal to run a uv command.
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
```

## Install

```bash
uv sync --dev

# Install flash attention if you have a Ampere (RTX 30xx series) or newer GPU
uv sync --dev --extra flash --no-cache
```
