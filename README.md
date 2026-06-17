# Text-to-SQL LoRA Fine-Tuning

Fine-tuning `TinyLlama-1.1B-Chat` with LoRA (Low-Rank Adaptation) to turn a chatty, general-purpose chat model into a terse, reliable text-to-SQL assistant — trained end-to-end on a free Google Colab T4 GPU.

This project is a hands-on walkthrough of the standard LoRA fine-tuning workflow: load a base model, probe it before training, attach LoRA adapters, train on a small instruction dataset, and directly compare before/after behavior on the same prompts.

## Why this project

Most LoRA tutorials either skip the "why" or bury it in abstractions. This notebook is built to make the effect of fine-tuning *visible*: the same three prompts are run through the model before and after training, side by side, so the shift from verbose chat-style answers to clean SQL output is obvious rather than asserted.

## What it does

1. Loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` in FP16 on a T4 GPU
2. Defines a chat-style prompt template for the task: *schema + question → SQL query*
3. Runs three fixed probe prompts through the **base** model to capture "before" behavior
4. Loads `b-mc2/sql-create-context` (78k+ examples) and takes a 3,000-row slice for a fast demo
5. Formats every example with the tokenizer's chat template and tokenizes for training
6. Attaches LoRA adapters (rank 16) to the attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
7. Trains with `SFTTrainer` from TRL for 1 epoch
8. Re-runs the same three probes through the **fine-tuned** model for a direct comparison
9. Saves the LoRA adapter (not the full model) to disk

## Results

| Metric | Value |
|---|---|
| Base model | TinyLlama-1.1B-Chat-v1.0 (FP16) |
| Base model memory footprint | 2.05 GB |
| GPU | Tesla T4, 14.56 GB VRAM (Colab free tier) |
| Trainable parameters (LoRA) | 4,505,600 / 1,104,553,984 (0.41%) |
| Training examples | 2,850 train / 150 eval (3,000 total, sampled from 78,577) |
| Epochs | 1 |
| Final training loss | 0.701 |
| Training time | ~10.9 minutes (654.5s) |
| Peak GPU memory during training | 4.10 GB |
| Saved adapter size | 20.67 MB |

### Before vs. after, same prompt

**Prompt:** *List the names of employees in the Engineering department earning more than 100000.*

**Before (base model)** — verbose, wraps the query in explanation and markdown:
```
To answer this question, you can use the following SQL query:

SELECT name
FROM employees
WHERE department = 'Engineering'
AND salary > 100000;

This query will return a list of names of employees in the Engineering
department who have a salary greater than 100,000.
```

**After (LoRA fine-tuned)** — direct SQL, no commentary:
```
SELECT name FROM employees WHERE department = "engineering" AND salary > 100000
```

The same pattern held across all three probe prompts: fine-tuning didn't teach the model new SQL knowledge so much as it taught the model the *response style* the task demands — answer only with the query, skip the explanation.

## Key design choices

- **Pure LoRA, no quantization.** The base model is loaded in plain FP16, not 4-bit/QLoRA. At 1.1B parameters it comfortably fits a T4's 16GB, so this keeps the workflow simpler and easier to reason about before introducing quantization.
- **Gradient checkpointing** is enabled to keep peak VRAM usage low (4.1GB peak) and leave headroom for larger batches or models.
- **Rank 16, alpha 32** (scaling factor 2.0) applied only to attention projections — a standard, lightweight starting configuration that keeps trainable parameters under 0.5% of the base model.
- **Small data slice (3,000 of 78,577 rows), 1 epoch** by design — the goal is a visible style shift in minutes on a free GPU, not a production-accuracy model. The notebook calls this out directly: style learning is not the same as task mastery.
- **Fixed probe prompts** captured before training and reused after training, so the comparison is apples-to-apples rather than anecdotal.

## What this is *not*

This is a demonstration of the LoRA fine-tuning workflow, not a production-grade text-to-SQL model. With a 1.1B base model, 3,000 training examples, and a single epoch, the model reliably learns the *style* of terse SQL output but is not always semantically correct on harder queries. The notebook's own conclusion: scaling up the same workflow (larger base model such as Qwen2.5-3B or a 7B with QLoRA, the full 78k-example dataset, and 2–3 epochs) would be the natural next step toward real accuracy — the code stays the same, only the scale changes.

## Tech stack

- `transformers` 5.0.0
- `peft` 0.18.1 (LoRA)
- `trl` 1.1.0 (`SFTTrainer`)
- `accelerate` 1.13.0
- `datasets` 4.8.4
- PyTorch (CUDA), Google Colab T4 GPU

## Dataset

[`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) — `(question, context, answer)` triples pairing a `CREATE TABLE` schema and a natural-language question with the correct SQL query.

## Repository structure

```
text-to-sql-lora-finetuning/
├── notebooks/
│   └── text_to_sql_lora_finetuning.ipynb   # full, runnable Colab notebook
├── README.md
├── requirements.txt
└── LICENSE
```

## Running it yourself

1. Open `notebooks/text_to_sql_lora_finetuning.ipynb` in Google Colab.
2. Set the runtime to a GPU (T4 is sufficient — Colab's free tier works).
3. Run all cells top to bottom. Total runtime is roughly 15–20 minutes, dominated by the ~11-minute training step.
4. The fine-tuned LoRA adapter is saved to `./tinyllama-sql-lora-adapter` at the end — load it later with `PeftModel.from_pretrained(base_model, "./tinyllama-sql-lora-adapter")`.

## Possible extensions

- Swap in a larger base model (Qwen2.5-3B, or a 7B model with QLoRA) for better SQL accuracy
- Train on the full 78k-example dataset for 2–3 epochs
- Add an automatic SQL-validity/execution check instead of eyeballing outputs
- Swap the dataset entirely to adapt the same pipeline to a different instruction-following task

## License

MIT — see [LICENSE](LICENSE).
