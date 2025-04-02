import csv
import datetime
import json
import re
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import transformers
import yaml
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from submission import get_hf_user

# Import from scoring only what we need
from scoring import ScoreCategory, extract_scores, process_scores

console = Console()


def load_submissions(submissions_dir: Union[str, Path]) -> List[Path]:
    "Load submission files from the specified directory."
    submissions_dir = Path(submissions_dir)
    if not submissions_dir.exists():
        raise ValueError(f"Submissions directory {submissions_dir} does not exist")
    files = list(submissions_dir.glob("**/*.csv"))
    console.print(f"[green]Found {len(files)} submissions to evaluate[/green]")
    return files


def create_evaluation_prompt(
    story_start: str,
    completion: str,
    prompt_file: Union[str, Path] = "prompts/simple_prompt.yaml",
    previous_response: Optional[str] = None,
    reasoning_template: Optional[str] = None,
):
    "Create a structured prompt for evaluating story completions from a YAML file, with optional follow-up"
    prompt_file = Path(prompt_file) if not isinstance(prompt_file, Path) else prompt_file

    if not prompt_file.exists():
        console.print(f"[red]Prompt file {prompt_file} not found[/red]")
        raise ValueError(f"Prompt file {prompt_file} not found. Please provide a valid prompt file.")

    # Load prompts from YAML file
    try:
        with open(prompt_file, "r") as f:
            prompts = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Error loading prompts from {prompt_file}: {str(e)}[/red]")
        raise

    system_prompt = prompts.get("system_prompt", "")
    user_prompt = prompts.get("user_prompt", "")
    followup_prompt = prompts.get("followup_prompt", "")

    # If reasoning template is provided and the system prompt has the placeholder
    if reasoning_template and "{reasoning_prompt}" in system_prompt:
        try:
            # Use specified reasoning template
            reasoning_file = Path(reasoning_template)

            if not reasoning_file.exists():
                console.print(f"[red]Reasoning template file {reasoning_file} not found[/red]")
                raise ValueError(f"Reasoning template file {reasoning_file} not found")

            with open(reasoning_file, "r") as f:
                reasoning_template_data = yaml.safe_load(f)

            reasoning_prompt = reasoning_template_data.get("reasoning_prompt", "")

            # Format the system prompt with the reasoning prompt
            system_prompt = system_prompt.format(reasoning_prompt=reasoning_prompt)
        except Exception as e:
            console.print(f"[red]Error applying reasoning template: {str(e)}[/red]")
            # Continue with the original system prompt if there's an error

    # Format the user prompt with provided values
    user_prompt = user_prompt.format(story_start=story_start, completion=completion)

    # Create messages based on whether this is initial or follow-up
    if previous_response is None:
        # Initial prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        # Follow-up prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": previous_response},
            {"role": "user", "content": followup_prompt},
        ]

    return messages


def load_model(
    model_dir: str,
    cache_size: int = 1024 * 50,
    draft_model_dir: Optional[str] = None,
    draft_cache_size: Optional[int] = None,
) -> Tuple[ExLlamaV2DynamicGenerator, str]:
    """Load an ExLlama2 model and return the generator and model architecture.

    Args:
        model_dir: Directory containing the model files
        cache_size: Cache size in tokens
        draft_model_dir: Directory containing the draft model files
        draft_cache_size: Draft cache size in tokens

    Returns:
        Tuple of (generator, model_architecture)
    """
    console.print(f"[yellow]Loading model from {model_dir}...[/yellow]")
    config = ExLlamaV2Config(model_dir)
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len=cache_size, lazy=True)
    model.load_autosplit(cache)
    tokenizer = ExLlamaV2Tokenizer(config)

    # Get HF tokenizer for chat template
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    tokenizer.apply_chat_template = hf_tokenizer.apply_chat_template

    if draft_model_dir is not None:
        console.print(f"[yellow]Loading draft model from {draft_model_dir}...[/yellow]")
        draft_config = ExLlamaV2Config(draft_model_dir)
        draft_model = ExLlamaV2(draft_config)
        draft_cache = ExLlamaV2Cache(draft_model, max_seq_len=draft_cache_size, lazy=True)
        draft_model.load_autosplit(draft_cache)
    else:
        draft_model = None
        draft_cache = None

    generator = ExLlamaV2DynamicGenerator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        max_batch_size=None,
        draft_model=draft_model,
        draft_cache=draft_cache,
    )
    generator.warmup()

    # Return the generator and model architecture
    return generator, config.architecture


def process_submission(
    submission_file: Path,
    generator: ExLlamaV2DynamicGenerator,
    model_arch: str,
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_new_tokens: int = 20,
    sample: Optional[int] = None,
    log_prompts: bool = False,
    prompt_file: Union[str, Path] = "prompts/simple_prompt.yaml",
    parent_progress: Optional["Progress"] = None,
    reasoning_template: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Process a single submission file and evaluate its completions.

    Args:
        submission_file: Path to the submission CSV file
        generator: ExLlama2 generator instance
        model_arch: Model architecture name
        temperature: Temperature for generation sampling
        top_p: Top-p (nucleus) sampling value
        max_new_tokens: Maximum number of tokens to generate
        sample: Optional limit to process only the first N items
        log_prompts: Whether to log prompts and responses
        prompt_file: Path to YAML file with evaluation prompts
        parent_progress: Optional parent Progress instance to use instead of creating a new one
        reasoning_template: Optional reasoning template to apply

    Returns:
        Dictionary containing scores data
    """
    username = submission_file.parent.name
    submission_id = submission_file.stem

    console.print(f"[yellow]Evaluating submission: {username}/{submission_id}[/yellow]")
    scores = {username: {submission_id: {model_arch: {"score": 0, "details": []}}}}

    try:
        # Start timing
        start_time = time.time()

        df = pd.read_csv(submission_file)
        prompts = df["prompt"].tolist()
        completions = df["completion"].tolist()

        if sample is not None:
            prompts = prompts[:sample]
            completions = completions[:sample]

        total_items = len(prompts)
        console.print(f"[blue]Found {total_items} items to evaluate in {username}/{submission_id}[/blue]")

        # Create log file path if logging is enabled
        log_file = None
        if log_prompts:
            log_dir = Path("logs") / username
            log_dir.mkdir(exist_ok=True, parents=True)
            log_file = log_dir / f"{submission_id}.json"
            console.print(f"[yellow]Will log prompts and responses to {log_file}[/yellow]")

        # Run evaluation
        scores = eval_completions(
            prompts,
            completions,
            generator,
            model_arch,
            username,
            submission_id,
            scores,
            temperature,
            top_p,
            max_new_tokens,
            sample=sample,
            log_file=log_file,
            prompt_file=prompt_file,
            parent_progress=parent_progress,
            reasoning_template=reasoning_template,
        )

        # Compute average time per item for reporting
        elapsed_time = time.time() - start_time
        avg_time_per_item = elapsed_time / total_items if total_items > 0 else 0
        items_per_second = total_items / elapsed_time if elapsed_time > 0 else 0

        # Report detailed stats
        console.print(
            f"[green]Completed evaluation for {username}/{submission_id} with score: {scores[username][submission_id][model_arch]['score']:.2f}[/green]"
        )
        console.print(
            f"[blue]Stats: {total_items} items in {elapsed_time:.2f}s ({avg_time_per_item:.2f}s per item, {items_per_second:.2f} items/s)[/blue]"
        )

    except Exception as e:
        console.print(f"[red]Error processing {username}/{submission_id}: {str(e)}[/red]")
        traceback.print_exc()

    return scores


def run_batch_evaluation(
    generator: ExLlamaV2DynamicGenerator,
    jobs_data: List[Dict[str, Any]],
    description: str,
    parent_progress: Optional["Progress"] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Run a batch of evaluation jobs through the generator.

    Args:
        generator: The ExLlamaV2DynamicGenerator instance
        jobs_data: List of dicts containing job data with keys 'prompt_ids' and 'identifier'
        description: Description for the progress bar
        parent_progress: Optional parent Progress instance to use instead of creating a new one

    Returns:
        Dictionary mapping identifiers to dicts with 'response' and 'metadata'
    """
    # Queue all jobs
    for job_data in jobs_data:
        job = ExLlamaV2DynamicJob(
            input_ids=job_data["prompt_ids"],
            gen_settings=job_data["gen_settings"],
            max_new_tokens=job_data["max_new_tokens"],
            stop_conditions=[generator.tokenizer.eos_token_id],
            identifier=job_data["identifier"],
        )
        generator.enqueue(job)

    # Process jobs and collect results
    results = {}
    num_completions = 0
    num_tokens = 0
    total_jobs = generator.num_remaining_jobs()
    time_begin = time.time()

    # Use either the provided progress or create a new one
    if parent_progress is not None:
        # Use the parent progress
        progress = parent_progress
        # Make sure to include the submissions_per_min field with a default value to avoid KeyError
        task = progress.add_task(f"[green]{description}", total=total_jobs, submissions_per_min=0.0)

        while generator.num_remaining_jobs():
            try:
                batch_results = generator.iterate()

                # Track tokens processed
                bsz = len(set([r["identifier"] for r in batch_results]))
                num_tokens += bsz

                for result in batch_results:
                    idx = result["identifier"]

                    if not result["eos"]:
                        continue

                    # For EOS results, get the full completion
                    results[idx] = {"response": result["full_completion"], "metadata": result}
                    num_completions += 1

                    # Update progress
                    progress.update(task, completed=num_completions)

            except Exception as e:
                console.print(f"[red]Error in generator iteration: {str(e)}[/red]")
                traceback.print_exc()
                continue

        # Remove the task when done since we don't want to close the parent progress
        progress.remove_task(task)
    else:
        # Generate all completions with our own progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # Create the main progress task
            task = progress.add_task(f"[green]{description}", total=total_jobs)

            while generator.num_remaining_jobs():
                try:
                    batch_results = generator.iterate()

                    # Track tokens processed
                    bsz = len(set([r["identifier"] for r in batch_results]))
                    num_tokens += bsz

                    for result in batch_results:
                        idx = result["identifier"]

                        if not result["eos"]:
                            continue

                        # For EOS results, get the full completion
                        results[idx] = {"response": result["full_completion"], "metadata": result}
                        num_completions += 1

                        # Update progress
                        progress.update(task, completed=num_completions)

                except Exception as e:
                    console.print(f"[red]Error in generator iteration: {str(e)}[/red]")
                    traceback.print_exc()
                    continue

    # Output statistics
    elapsed_time = time.time() - time_begin
    rpm = num_completions / (elapsed_time / 60) if elapsed_time > 0 else 0
    tps = num_tokens / elapsed_time if elapsed_time > 0 else 0
    console.print()
    console.print(f"[blue]Avg. completions/minute: {rpm:.2f}[/blue]")
    console.print(f"[blue]Avg. output tokens/second: {tps:.2f}[/blue]")
    console.print()

    return results


def eval_completions(
    prompts: List[str],
    completions: List[str],
    generator: ExLlamaV2DynamicGenerator,
    model_arch: str,
    username: str,
    submission_id: str,
    scores: Dict[str, Dict[str, Dict[str, Any]]],
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_new_tokens: int = 20,
    sample: Optional[int] = None,
    log_file: Optional[Path] = None,
    prompt_file: Union[str, Path] = "prompts/simple_prompt.yaml",
    parent_progress: Optional["Progress"] = None,
    reasoning_template: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Evaluate each completion in a submission using batch processing with re-prompting for missing scores.

    Args:
        prompts: List of story beginnings
        completions: List of story completions to evaluate
        generator: ExLlama2 generator instance
        model_arch: Model architecture name
        username: Username of the submission
        submission_id: ID of the submission
        scores: Dictionary to store scores
        temperature: Temperature for generation sampling
        top_p: Top-p (nucleus) sampling value
        max_new_tokens: Maximum number of tokens to generate
        sample: Optional limit to process only the first N items
        log_file: Optional path to save logs
        prompt_file: Path to YAML file with evaluation prompts
        parent_progress: Optional parent Progress instance to use instead of creating a new one
        reasoning_template: Optional reasoning template to apply

    Returns:
        Updated scores dictionary
    """
    try:
        # Create sampler settings
        gen_settings = ExLlamaV2Sampler.Settings(temperature=temperature, top_p=top_p, token_repetition_penalty=1.0, top_k=0)

        # Reset generator queue if needed
        if generator.num_remaining_jobs() > 0:
            console.print("[yellow]Clearing existing generator queue...[/yellow]")
            generator.clear_queue()

        # Prepare initial evaluation jobs
        console.print(f"[yellow]Queueing {len(completions)} evaluation jobs...[/yellow]")
        initial_jobs_data = []
        prompts_log = {}
        followup_prompts_log = {}

        for i, (beginning, completion) in enumerate(zip(prompts, completions)):
            raw_prompt = create_evaluation_prompt(
                story_start=beginning, completion=completion, prompt_file=prompt_file, reasoning_template=reasoning_template
            )
            templated_prompt = generator.tokenizer.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=False)

            # Store prompt for logging
            prompts_log[i] = templated_prompt

            prompt_ids = generator.tokenizer.encode(templated_prompt, encode_special_tokens=True)
            initial_jobs_data.append(
                {
                    "prompt_ids": prompt_ids,
                    "gen_settings": gen_settings,
                    "max_new_tokens": max_new_tokens,
                    "identifier": i,
                }
            )

        # Run initial evaluation
        console.print("[yellow]Processing evaluation jobs...[/yellow]")
        initial_results = run_batch_evaluation(
            generator, initial_jobs_data, f"Evaluating {username}/{submission_id}", parent_progress=parent_progress
        )

        # Process all responses and check for missing scores
        console.print("[yellow]Processing responses and calculating scores...[/yellow]")

        # Extract responses and check which need follow-up
        responses = {}
        followup_responses = {}
        metadata = {}
        items_needing_followup = set()

        # Extract responses and check which need follow-up
        for idx, result_data in initial_results.items():
            response = result_data["response"]
            responses[idx] = response
            metadata[idx] = result_data["metadata"]

            # Extract score
            try:
                extracted_scores, success = extract_scores(response)
                for key, value in list(extracted_scores.items()):
                    extracted_scores[key] = min(10, max(1, value))

                # If we're missing any scores, mark for follow-up
                if not success:
                    items_needing_followup.add(idx)
                    console.print(f"[yellow]Item {idx} missing scores, will re-prompt[/yellow]")

            except Exception as e:
                console.print(f"[red]Error extracting score from response for item {idx}: {str(e)}[/red]")
                items_needing_followup.add(idx)

        # Handle follow-up prompts if needed
        follow_up_count = len(items_needing_followup)

        if follow_up_count > 0:
            console.print(f"[yellow]Re-prompting {follow_up_count} items with missing scores...[/yellow]")

            # Clear generator queue
            if generator.num_remaining_jobs() > 0:
                generator.clear_queue()

            # Prepare follow-up jobs
            followup_jobs_data = []
            followup_id_mapping = {}  # Maps new queue IDs to original IDs
            followup_count = 0

            for original_idx in items_needing_followup:
                beginning = prompts[original_idx]
                completion = completions[original_idx]
                previous_response = responses[original_idx]

                # Create follow-up prompt
                raw_prompt = create_evaluation_prompt(
                    story_start=beginning,
                    completion=completion,
                    prompt_file=prompt_file,
                    previous_response=previous_response,
                    reasoning_template=reasoning_template,
                )

                templated_prompt = generator.tokenizer.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=False)

                # Extract the portion of the templated prompt that comes after the previous response
                # First, escape any special characters in the previous response
                escaped_response = re.escape(previous_response)
                # Find where the previous response ends in the templated prompt
                match = re.search(escaped_response, templated_prompt)
                if match:
                    # Extract everything after the previous response
                    followup_part = templated_prompt[match.end() :]
                    followup_prompts_log[original_idx] = followup_part

                prompt_ids = generator.tokenizer.encode(templated_prompt, encode_special_tokens=True)

                # Create new job with new identifier
                followup_idx = 1000000 + followup_count  # Use a large offset to avoid ID conflicts
                followup_id_mapping[followup_idx] = original_idx

                followup_jobs_data.append(
                    {
                        "prompt_ids": prompt_ids,
                        "gen_settings": gen_settings,
                        "max_new_tokens": max_new_tokens,
                        "identifier": followup_idx,
                    }
                )

                followup_count += 1

            # Run follow-up evaluation
            console.print(f"[yellow]Processing {followup_count} follow-up evaluation jobs...[/yellow]")
            followup_results = run_batch_evaluation(
                generator, followup_jobs_data, "Re-evaluating items with missing scores", parent_progress=parent_progress
            )

            # Process follow-up responses
            console.print(f"[yellow]Processing {len(followup_results)} follow-up responses...[/yellow]")

            for followup_idx, result_data in followup_results.items():
                original_idx = followup_id_mapping[followup_idx]
                followup_response = result_data["response"]

                # Store follow-up response separately
                followup_responses[original_idx] = followup_response

        # Process all responses and extract scores
        scores, item_scores, total_score, processed_count = process_scores(
            responses,
            followup_responses,
            username,
            submission_id,
            model_arch,
            scores,
        )

        # Log prompts and responses if log_file is provided
        if log_file:
            log_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "username": username,
                "submission_id": submission_id,
                "evaluations": [],
            }

            # Add each evaluation to the log
            for idx in range(len(prompts)):
                if idx in item_scores:
                    log_item = {
                        "item_id": idx,
                        "prompt": prompts_log.get(idx, ""),
                        "response": responses.get(idx, ""),
                        "scores": item_scores[idx],
                    }

                    # Add follow-up prompt and response if they exist
                    if idx in followup_prompts_log:
                        log_item["followup_prompt"] = followup_prompts_log.get(idx, "")

                    if idx in followup_responses:
                        log_item["followup_response"] = followup_responses.get(idx, "")

                    # Add metadata if available
                    if idx in metadata:
                        log_item["metadata"] = {
                            k: v.item() if isinstance(v, torch.Tensor) else v
                            for k, v in metadata.get(idx, {}).items()
                            if k not in ["full_completion", "job", "held"]
                        }

                    log_data["evaluations"].append(log_item)

            # Ensure log directory exists
            log_file.parent.mkdir(exist_ok=True, parents=True)

            # Read existing logs organized by model
            existing_logs = {}
            if log_file.exists():
                try:
                    existing_logs = json.loads(log_file.read_text())
                    if not isinstance(existing_logs, dict):
                        existing_logs = {}
                except:
                    existing_logs = {}

            # Create model_arch key if it doesn't exist and store log directly (overwrite)
            model_key = model_arch.replace("/", "_")  # Ensure safe key name
            existing_logs[model_key] = log_data  # Overwrite instead of append to a list

            # Write updated logs back to file
            log_file.write_text(json.dumps(existing_logs, indent=2))
            console.print(f"[green]Saved prompt and response logs for model {model_arch} to {log_file}[/green]")

        # Final update
        if processed_count > 0:
            console.print(
                f"[green]Completed processing {processed_count} responses with final avg score: {scores[username][submission_id][model_arch]['score']:.2f}[/green]"
            )
            if follow_up_count > 0:
                success_count = len([idx for idx in items_needing_followup if idx in item_scores])
                console.print(f"[blue]Re-prompted {follow_up_count} items, successfully retrieved scores for {success_count} items[/blue]")

    except Exception as e:
        console.print(f"[red]Error in eval_completions: {str(e)}[/red]")
        traceback.print_exc()

    return scores


def score_submission(
    submission_file: Union[Path, List[Path]],
    model_dir: str,
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_new_tokens: int = 20,
    cache_size: int = 1024 * 60,
    log_prompts: bool = False,
    prompt_file: Union[str, Path] = "prompts/simple_prompt.yaml",
    generator: Optional[ExLlamaV2DynamicGenerator] = None,
    model_arch: Optional[str] = None,
    sample: Optional[int] = None,
    draft_model_dir: Optional[str] = None,
    draft_cache_size: Optional[int] = None,
    reasoning_template: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Score one or more submissions using the specified model.

    Args:
        submission_file: Path to submission CSV or list of submission CSV paths
        model_dir: Directory containing model files
        temperature: Temperature for generation
        top_p: Top-p sampling value
        max_new_tokens: Maximum tokens to generate
        cache_size: Cache size in tokens
        log_prompts: Whether to log prompts and responses
        prompt_file: Path to prompt file
        generator: Optional pre-loaded generator to reuse
        model_arch: Optional model architecture name if generator is provided
        sample: Optional number of samples to score per submission
        draft_model_dir: Directory containing the draft model files
        draft_cache_size: Draft cache size in tokens
        reasoning_template: Optional reasoning template to apply

    Returns:
        Dictionary with scores for all processed submissions
    """
    # Only load the model if neither generator nor model_arch is provided
    if generator is None and model_arch is None:
        generator, model_arch = load_model(model_dir, cache_size, draft_model_dir, draft_cache_size)
        console.print(f"[green]Loaded model {model_arch} successfully[/green]")
    # Handle case where one is provided but not the other
    elif generator is None:
        console.print(f"[yellow]Loading model (architecture {model_arch} specified but generator missing)...[/yellow]")
        generator, _ = load_model(model_dir, cache_size, draft_model_dir, draft_cache_size)
        console.print("[green]Loaded model successfully[/green]")
    elif model_arch is None:
        console.print("[yellow]Warning: Using provided generator but model architecture name unknown[/yellow]")
        model_arch = "unknown"
    else:
        console.print(f"[green]Using pre-loaded model {model_arch}[/green]")

    # Prepare a list of submissions to process
    if isinstance(submission_file, list):
        submission_files = submission_file
    else:
        submission_files = [submission_file]

    # Initialize combined scores dictionary
    all_scores = {}

    # Process each submission with progress tracking
    start_time = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[cyan]{task.fields[submissions_per_min]:.2f} submissions/min"),
        console=console,
    ) as progress:
        task = progress.add_task(f"[green]Scoring submissions with {model_arch}", total=len(submission_files), submissions_per_min=0.0)

        for i, csv_path in enumerate(submission_files):
            # Update description with current submission
            progress.update(task, description=f"[green]Scoring submission {i + 1}/{len(submission_files)}: {csv_path.name}")

            try:
                # Process the submission
                scores = process_submission(
                    submission_file=csv_path,
                    generator=generator,
                    model_arch=model_arch,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    sample=sample,  # Pass the sample limit
                    log_prompts=log_prompts,
                    prompt_file=prompt_file,
                    parent_progress=progress,  # Pass the progress object
                    reasoning_template=reasoning_template,
                )

                # Merge with combined scores
                for username in scores:
                    if username not in all_scores:
                        all_scores[username] = {}

                    for submission_id in scores[username]:
                        all_scores[username][submission_id] = scores[username][submission_id]

                # Update progress
                elapsed = time.time() - start_time
                submissions_per_min = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
                progress.update(task, advance=1, submissions_per_min=submissions_per_min)

            except Exception as e:
                console.print(f"[red]Error processing {csv_path}: {str(e)}[/red]")
                traceback.print_exc()
                # Still advance the progress bar even on error
                progress.update(task, advance=1)
                continue

    console.print(f"[green]Completed scoring {len(submission_files)} submissions in {time.time() - start_time:.2f} seconds[/green]")
    return all_scores
