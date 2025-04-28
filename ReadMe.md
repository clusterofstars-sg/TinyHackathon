# TinyStories Hackathon

## Setup Submission

This repository uses [uv](https://docs.astral.sh/uv) to managing dependencies.

After installing uv, you can either install the dependencies by running `uv sync --no-dev` or by using `uv run` to run the submission script directly.

## Submission Instructions

To submit your entries to the TinyStories Hackathon, use the provided `tinyhackathon/submission.py` script. Before submitting, ensure you have a Hugging Face account and token with write access to the Cluster of Stars organization.

The submission script is under the `tinyhackathon` directory.

### Log Into Hugging Face

You can use the following command to check if you are already logged in:

```bash
python submission.py whoami
# or
uv run submission.py whoami
```

If not, follow the [Hugging Face instructions to log in](https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login).

### Submit Your Entries

> Important: Your submissions will be located under your username in the submissions dataset. Only upload using the script so you submissions are placed in the correct location.

1. Download evaluation dataset:
```bash
python submission.py download-eval
# or
uv run submission.py download-eval
```

This will download the evaluation dataset to `evaluation_prompts.csv` in the local directory.

The evaluation dataset is has two columns: `prompt` and `completion`. Pass the prompt to your TinyStories model and generate the completion. Only subit the model's output, not the prompt to the completion column.

Each prompt is repeated four times, generate four unique completions for each prompt.

2. Submit your file to the hackathon:
```bash
python submission.py submit --submission-path PATH_TO_FILE [--submission-name NAME] --submit
# or
uv run submission.py submit --submission-path PATH_TO_FILE [--submission-name NAME] --submit
```

where `submission-name` is an optional human-readable name for the submission.

By default, the submission script will submit to the test dataset which will not be scored. For a realsubmission, use the `--submit` flag.

Set `submission-name` to optionally give your submission a human-friendly identifier. If not set, it will only have a timestamp.

> Note: You are limited to one submission per day.

## Final Submission

During the weekend of April 12-13th, we will upload the final submission dataset. You will only submit one version with your best model's completions. These will be different prompts from the evaluation dataset.

## Running the Score Explorer Web Application

To run the Score Explorer web application locally:

1. Make sure you have all dependencies installed
2. Run the following command from the project root:

```bash
python tinyhackathon/score_explorer/run.py
```

3. Open your browser and navigate to `http://localhost:8001`

## Setup LLM Evaluation

 ```bash
# swap cuda-toolkit for cuda if you want to compile cuda packages
conda create -n tinyhackathon python=3.12 uv cuda-toolkit -c nvidia/label/cuda-12.4.1 -c conda-forge
conda activate tinyhackathon
# This sets uv to use the active Conda environment whether using uv or uv pip commands.
# You'll need to run this command every time you open a new terminal to run a uv command.
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
```

### Install

```bash
uv sync --dev

# Install flash attention if you have a Ampere (RTX 30xx series) or newer GPU
uv sync --dev --extra flash --no-cache
```

*Note: This application is over 90% vibecoded - because sometimes the best code comes from pure vibes and good LLMs.*
