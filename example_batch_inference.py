import time
import traceback
from pathlib import Path

import typer
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)
console = Console()


# modified from https://github.com/turboderp-org/exllamav2/blob/master/examples/bulk_inference.py
@app.command()
def test_llm_generation(
    model_path: Path,
    num_prompts: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    batch_size: int = 32,
    cache_size: int = 50 * 1024,
):
    """Generate multiple 'tell me a story' prompts to test ExLlamaV2DynamicJob setup."""
    try:
        config = ExLlamaV2Config(model_path)
        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, batch_size=1, max_seq_len=cache_size, lazy=True)
        model.load_autosplit(cache)
        tokenizer = ExLlamaV2Tokenizer(config)
        generator = ExLlamaV2DynamicGenerator(model=model, cache=cache, tokenizer=tokenizer, max_batch_size=batch_size, max_q_size=1)

        # Create sampler settings
        gen_settings = ExLlamaV2Sampler.Settings(temperature=temperature, top_p=top_p, token_repetition_penalty=1.0, top_k=0)

        # Queue all jobs
        console.print(f"[yellow]Queueing {num_prompts} story generation jobs...[/yellow]")

        # Reset generator queue if needed
        if generator.num_remaining_jobs() > 0:
            console.print("[yellow]Clearing existing generator queue...[/yellow]")
            generator.clear_queue()

        # Queue all generation jobs with simple prompts
        responses = {}
        for i in range(num_prompts):
            # Create simple variations of "tell me a story" prompts
            themes = ["adventure", "mystery", "romance", "sci-fi", "fantasy", "horror", "comedy", "drama", "historical", "fairy tale"]
            theme = themes[i % len(themes)]
            prompt = f"Tell me a short {theme} story."

            prompt_ids = generator.tokenizer.encode(prompt, encode_special_tokens=True)
            job = ExLlamaV2DynamicJob(
                input_ids=prompt_ids,
                gen_settings=gen_settings,
                max_new_tokens=max_new_tokens,
                identifier=i,
                stop_conditions=[generator.tokenizer.eos_token_id],
            )
            generator.enqueue(job)
            responses[i] = ""  # Initialize empty response

        # Process jobs and collect results
        console.print("[yellow]Processing story generation jobs...[/yellow]")

        time_begin = time.time()
        num_completions = 0
        num_tokens = 0
        total_jobs = generator.num_remaining_jobs()

        # Generate all completions with progress bar
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
            task = progress.add_task("[green]Generating stories", total=total_jobs)

            while generator.num_remaining_jobs():
                try:
                    results = generator.iterate()

                    # Track tokens processed
                    bsz = len(set([r["identifier"] for r in results]))
                    num_tokens += bsz

                    for result in results:
                        if not result["eos"]:
                            continue

                        idx = result["identifier"]

                        # EOS signal is accompanied by the full completion
                        responses[idx] = result["full_completion"]
                        num_completions += 1

                    # Update progress only once with all info
                    progress.update(task, completed=num_completions)

                except Exception as e:
                    console.print(f"[red]Error in generator iteration: {str(e)}[/red]")
                    traceback.print_exc()
                    continue

        elapsed_time = time.time() - time_begin
        rpm = num_completions / (elapsed_time / 60)
        tps = num_tokens / elapsed_time
        console.print()
        console.print(f"[blue]Avg. completions/minute: {rpm:.2f}[/blue]")
        console.print(f"[blue]Avg. output tokens/second: {tps:.2f}[/blue]")
        console.print()

        # Display generated stories
        console.print("[yellow]Generated stories:[/yellow]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Prompt ID")
        table.add_column("Prompt")
        table.add_column("Generated Story")

        themes = ["adventure", "mystery", "romance", "sci-fi", "fantasy", "horror", "comedy", "drama", "historical", "fairy tale"]
        for idx, response in responses.items():
            theme = themes[idx % len(themes)]
            prompt = f"Tell me a short {theme} story."
            # Truncate long responses for display
            truncated_response = response[:100] + "..." if len(response) > 100 else response
            table.add_row(str(idx), prompt, truncated_response)

        console.print(table)
        console.print(f"[green]Completed generating {num_completions} stories[/green]")

    except Exception as e:
        console.print(f"[red]Error in test_llm_generation: {str(e)}[/red]")
        traceback.print_exc()

    return responses


if __name__ == "__main__":
    app()
