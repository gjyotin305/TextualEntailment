# TextualEntailment

## Task and Setup:
This repository implements a Textual Entailment model, determining if a hypothesis logically follows from a given premise.

### Hugging Face Finetuned Models
- [LoRA Adapter](https://huggingface.co/gjyotin305/TextualEntailment-LoRA)
- [GGUF f16](https://huggingface.co/gjyotin305/TextualEntailment-Llama)

### Setup:
These models were trained on a proprietary Textual Entailment Dataset.

The Prompt Template to use is given below:
```bash
### Premise: <premise>
### Hypothesis: <hypothesis>
```

Response format:
```bash
### Response <response>
```

## Instructions to run:

### Run training
- Involves SFT with Noisy Embeddings along with AdamW_8bit for lower memory consumption.

```bash
python train.py
```

### Run evaluation

```bash
python inference.py
```

## WANDB Reports

Report Link : [Report](https://api.wandb.ai/links/gjyotin1724/0cq9uc10)
