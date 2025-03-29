# TinyStories Hackathon Rules
This hackathon is intended to be a fun competition to give ourselves practice pretraining LLMs on consumer hardware. We will follow the [TinyStories paper](<https://arxiv.org/abs/2305.07759>) and train small language models on small datasets and hardware.

The hackathon will end on April 7th, [AOE](<https://en.wikipedia.org/wiki/Anywhere_on_Earth>).

## Datasets
1. [**TinyStories:**](<https://huggingface.co/datasets/roneneldan/TinyStories>)
   Note that the TinyStories dataset is split into two versions both in the HF dataset:
     - GPT-3.5 generated TinyStories
    - GPT-4 generated TinyStories
   The tar file appears to have the cleanest versions with the least number of duplicates.
2. **[Simple Wikipedia](<https://huggingface.co/datasets/lsb/simplewiki2023>)** (optional)
   This dataset can be used to give your model more world knowledge than from just the TinyStories dataset. But be careful that it doesn't cause your model to use words which a typical 3 to 4-year-olds doesn't understand. It may need to be cleaned.

## Evaluation
Models will be evaluated by LLM-as-a-judge following the methodology outlined in the TinyStories paper. More details including how to submit your model's outputs early next week.

## Model Size Limits
Participants will be slotted into one of the following categories based on their hardware:
- **Small**: Up to 30M parameters. Low-to-mid range laptop GPUs and Apple Silicon.
- **Medium**: Up to 60M parameters. Mid-range GPUs (including high-end laptop GPUs and Apple Silicon)
- **Large**: Up to 120M parameters. High-end GPUs and multi-GPU systems.

You are welcome to train a larger model if you want, up to 120M parameters.

## Tokenizers
While you must train your model from scratch, you are welcome to use any pre-trained tokenizer or train your own tokenizer.

## Model Architecture
You are welcome to use any model architecture you want provided you stay within the parameter budget of your hardware by following the parameter counting rules below.

## Parameter Counting
The Parameter budget is the number of unique floating-point weights receiving gradient updates:
- Unique Weights: Count each distinct floating-point weight stored in the model once.
- Reuse Multiplier: For each weight, multiply by the number of distinct times it contributes to forward computation (e.g., due to layer-sharing, layer reuse, or non-standard head-sharing). Weight-tied embedding and decoder weights are the exception and are only counted once. MQA/GQA doesn't count as head-sharing.

## Teams
Teams are limited to a maximum of 2 members and must be formed and declared within the first week.

## Training Frameworks
You might want to take a look at the following libraries and frameworks and adopt one for pretraining:
- [Composer](https://docs.mosaicml.com/projects/composer/en/stable/index.html) and optionally [LLM Foundry](https://github.com/mosaicml/llm-foundry)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and optionally [LitGPT](https://github.com/Lightning-AI/litgpt)
- Hugging Face [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer), [Accelerate](https://huggingface.co/docs/accelerate/en/index), and optionally [Axolotl](https://axolotl-ai-cloud.github.io/axolotl/) (a wrapper on top of HF)
- [fastai](https://docs.fast.ai/) with either [fastxtend](https://fastxtend.benjaminwarner.dev/text.huggingface.html)/[blurr](https://ohmeow.github.io/blurr/)

Or if you don't want an entire framework:
- [nanoGPT](https://github.com/karpathy/nanoGPT)

## Parameter Counting Code
You can count the number of parameters in your model by running the following code (excluding weight reuse logic):
```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```