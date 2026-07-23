# Sales Effect of Perceived Influencer Agreeableness: Leveraging the Power of LMM and Custom LLM

## Overview

This repository contains the code and data used in the research paper "Sales Effect of Perceived Influencer Agreeableness: Leveraging the Power of LMM and Custom LLM".

## Installation

Before running the scripts, ensure you have the following dependencies installed:

- Python 3.11
- PyTorch (CUDA build)
- Transformers
- TRL
- PEFT
- Accelerate
- Datasets
- Pandas
- Numpy
- Scipy

You can install these packages using pip:

```bash
pip install -r requirements.txt
```

You will also need the base model [`Meta-Llama-3-8B-Instruct`](https://llama.meta.com/llama3/) (Meta AI, released under the [Meta Llama 3 Community License](https://llama.meta.com/llama3/license/)), which is not redistributed in this repository and must be obtained separately.

## Dataset

 - `data/test.pkl` 

## Model Checkpoint

`checkpoint/` contains a LoRA adapter trained on top of `Meta-Llama-3-8B-Instruct`:

- Configuration: `r=64, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj","v_proj"]`.
- LoRA was chosen over Retrieval-Augmented Generation (RAG) because it only adjusts a small subset of parameters, which lowers the risk of catastrophic forgetting during fine-tuning while still supporting elaborate text-generation tasks.
- The adapter was trained on a task-specific external store of agreeableness scripts (balanced high/low), using prompts crafted with the RACE framework so the model better internalizes the concept of "agreeableness".
- The weight file (`adapter_model.safetensors`, ~105MB) is tracked with Git LFS and is downloaded automatically with `git clone`.

## Usage

To run the analysis, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/multimodal-agreeableness/multimodal_agreeableness.git
   ```

2. Navigate to the cloned directory.
3. Run the scripts under `code/` (training, generation, evaluation).

## Code Structure

- `sft.py` — LoRA supervised fine-tuning script (built on `trl.SFTTrainer`) used to produce `checkpoint/`, with completion-only loss masking so loss is only computed on the target script, not the instruction.
- `generate_comparison.py` — generates short-video marketing scripts from both the base model and the LoRA-merged custom model, under matched high/low agreeableness prompts.
- `evaluate_agreeableness_similarity.py` — computes BERT-embedding cosine similarity of generated scripts to the concepts "agreeableness" and "dried fruit", and runs Welch's t-tests between conditions.

## Evaluation Metrics

The code includes functions for evaluating generation quality:

- Cosine similarity (BERT embedding space) between generated text and target concept words
- Welch's t-test, comparing custom vs. base model and high- vs. low-agreeableness conditions

## License

Code and data in this repository are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (see `LICENSE`). The base model `Meta-Llama-3-8B-Instruct` is governed separately by Meta's own Llama 3 Community License.
