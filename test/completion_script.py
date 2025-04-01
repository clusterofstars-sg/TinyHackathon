from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rich.console import Console
import typer
import pandas as pd
from typing import Annotated, Optional

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)
console = Console()

def generate_completion(prompt: str,model_hf:str, max_new_tokens: int=512):
    tokenizer = AutoTokenizer.from_pretrained(model_hf)
    model = AutoModelForCausalLM.from_pretrained(model_hf)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7,num_beams=5, pad_token_id=tokenizer.eos_token_id)
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip()

@app.command()
def generate(evaluation_csv: Annotated[str, typer.Argument(help="Path to evaluation_prompts.csv")], model_hf: Annotated[str, typer.Option(help="Load model from huggingface")] = 'roneneldan/TinyStories-8M'):
    df = pd.read_csv(evaluation_csv)
    
    #Get rid of duplicates, uniform sample, 300//20=15 total
    selected_rows = df#[df.index % 20 == 3].copy() 

    completions= [generate_completion(prompt, model_hf) for prompt in selected_rows['prompt']]
    selected_rows['completion'] = completions
    filename = f'completions_big_{model_hf.split("/")[-1]}.csv'
    console.print(f"[green]Completions saved to {filename}[/green]")
    selected_rows.to_csv(filename, index=False)

if __name__ == "__main__":
    app()
