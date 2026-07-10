# Custom Llama3 for Agreeableness: LoRA Fine-Tuning of a Text-Generation Model

## Overview

This repository contains the code, data, and model checkpoint used in the research paper's web appendix study "Text generation with Custom LLM". A custom LLM (Llama3-LoRA) is fine-tuned to generate short-video marketing scripts that display either high or low agreeableness.

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

`data/train.pkl` (800 examples) and `data/test.pkl` (200 examples) contain the supervised fine-tuning data. Each file is a pickled list of `{"text": ...}` dicts, pairing an instruction to write a short-video marketing script displaying high or low agreeableness with a real annotated script of the corresponding label. Labels are balanced (50% high / 50% low) and there is no overlap between the train and test splits.

The original raw annotated corpus is not included in this repository; only the processed fine-tuning data above is released.

## Model Checkpoint

`checkpoint/` contains a LoRA adapter (`r=64, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj","v_proj"]`) trained on top of `Meta-Llama-3-8B-Instruct`. The weight file (`adapter_model.safetensors`, ~105MB) is tracked with Git LFS and is downloaded automatically with `git clone`.

## Usage

To run the analysis, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/multimodal-agreeableness/multimodal_agreeableness.git
   ```
2. Navigate to the cloned directory.
3. Run the scripts under `code/` (training, generation, evaluation).

## Code Structure

- `sft.py` — LoRA supervised fine-tuning script (built on `trl.SFTTrainer`), with completion-only loss masking so loss is only computed on the target script, not the instruction. Loads `train.pkl` / `test.pkl`.
- `generate_comparison.py` — generates short-video marketing scripts from both the base model and the LoRA-merged custom model, under matched high/low agreeableness prompts.
- `evaluate_agreeableness_similarity.py` — computes BERT-embedding cosine similarity of generated scripts to the concepts "agreeableness" and "dried fruit", and runs Welch's t-tests between conditions.

## Evaluation Metrics

The code includes functions for evaluating generation quality:

- Cosine similarity (BERT embedding space) between generated text and target concept words
- Welch's t-test, comparing custom vs. base model and high- vs. low-agreeableness conditions

## License

Code and data in this repository are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (see `LICENSE`). The base model `Meta-Llama-3-8B-Instruct` is governed separately by Meta's own Llama 3 Community License.
