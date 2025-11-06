# Guide: Saving and Using Edited Models

This guide explains how to save MEMIT/ROME edited models and use them for inference later.

## Overview

The MEMIT codebase now supports saving weight deltas (the changes made to the model) to disk in a compact format. Instead of saving the entire multi-gigabyte model, we save only the delta tensors (typically just a few megabytes), which can be reapplied to the original model for inference.

## Quick Start

### 1. Running Evaluation with Delta Saving

To save deltas during evaluation, add the `--save_deltas` flag:

```bash
python experiments/evaluate.py \
    --alg_name MEMIT \
    --model_name gpt2-xl \
    --hparams_fname gpt2-xl.json \
    --ds_name mcf \
    --dataset_size_limit 10 \
    --save_deltas
```

Deltas will be saved in `results/<alg_name>/<run_id>/deltas/` alongside the evaluation results.

**Optional: Skip Weight Restoration**

By default, the evaluation script restores the original model weights after each edit case. If you want to keep the edited weights (e.g., for further processing), use the `--no_restore` flag:

```bash
python experiments/evaluate.py \
    --alg_name MEMIT \
    --model_name gpt2-xl \
    --hparams_fname gpt2-xl.json \
    --ds_name mcf \
    --dataset_size_limit 10 \
    --save_deltas \
    --no_restore
```

⚠️ **Warning**: Using `--no_restore` means each subsequent edit will be applied on top of previous edits. This is generally not recommended for evaluation.

### 2. Using Saved Deltas for Inference

Once you have saved deltas, you can use them for inference:

```bash
# Basic usage
python experiments/inference_with_deltas.py \
    --delta-path results/MEMIT/run_000/deltas/case_0_deltas.pt \
    --prompt "The Space Needle is located in the city of"

# Interactive mode
python experiments/inference_with_deltas.py \
    --delta-path results/MEMIT/run_000/deltas/case_0_deltas.pt \
    --interactive

# Compare with original model
python experiments/inference_with_deltas.py \
    --delta-path results/MEMIT/run_000/deltas/case_0_deltas.pt \
    --prompt "The Space Needle is located in the city of" \
    --compare-original
```

### 3. Running the Demo

A complete demo showing the save/load workflow:

```bash
python experiments/demo_delta_save.py
```

This demo will:
1. Load GPT-2 XL and apply a MEMIT edit
2. Save the deltas to disk
3. Restore the original model
4. Load the deltas and reapply them
5. Verify the outputs match

## File Structure

When you save deltas, two files are created for each case:

```
results/MEMIT/run_000/deltas/
├── case_0_deltas.pt        # PyTorch tensor file with weight deltas
├── case_0_metadata.json    # Metadata (model name, edit info, etc.)
├── case_1_deltas.pt
└── case_1_metadata.json
```

### Delta File Format

The `.pt` file contains a dictionary:
```python
{
    "transformer.h.13.mlp.c_proj.weight": (key_matrix, value_matrix),
    "transformer.h.14.mlp.c_proj.weight": (key_matrix, value_matrix),
    ...
}
```

Each entry is a low-rank factorization: `update = key_matrix @ value_matrix.T`

### Metadata File Format

The `.json` file contains:
```json
{
  "case_id": 0,
  "model_name": "gpt2-xl",
  "timestamp": "2025-11-06T10:30:00",
  "modified_weights": ["transformer.h.13.mlp.c_proj.weight", ...],
  "delta_shapes": {...},
  "hparams": {...},
  "requests": [
    {
      "prompt": "The Space Needle is located in the city of",
      "subject": "Space Needle",
      "target_new": " Berlin"
    }
  ]
}
```

## Programmatic Usage

### Saving Deltas

```python
from util.delta_saver import save_deltas
from memit import MEMITHyperParams, apply_memit_to_model

# Apply MEMIT with return_deltas=True
edited_model, weights_info = apply_memit_to_model(
    model, tok, requests, hparams,
    return_orig_weights=True,
    return_deltas=True  # Important!
)

# Save the deltas
deltas = weights_info["deltas"]
save_deltas(
    deltas,
    save_dir="./my_deltas",
    case_id=0,
    model_name="gpt2-xl",
    hparams=hparams,
    requests=requests
)
```

### Loading and Applying Deltas

```python
from util.delta_saver import load_and_apply_deltas
from transformers import AutoModelForCausalLM

# Load fresh model
model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda()

# Load and apply deltas
edited_model, metadata = load_and_apply_deltas(
    model,
    delta_path="./my_deltas/case_0_deltas.pt"
)

# Now use edited_model for inference
```

### Manual Loading (Advanced)

```python
from util.delta_saver import load_deltas, apply_deltas_to_model

# Load deltas separately
deltas, metadata = load_deltas("./my_deltas/case_0_deltas.pt")

# Inspect metadata
print(f"Original edit: {metadata['requests'][0]['subject']} -> {metadata['requests'][0]['target_new']}")

# Apply to model
apply_deltas_to_model(model, deltas)
```

## Storage Requirements

Deltas are very compact compared to full models:

| Model | Full Model Size | Delta Size (typical) |
|-------|----------------|---------------------|
| GPT-2 XL (1.5B) | ~6 GB | ~5-20 MB |
| GPT-J (6B) | ~24 GB | ~20-50 MB |

The exact delta size depends on:
- Number of layers edited (configured in hparams)
- Number of simultaneous edits (`num_edits` parameter)

## Supported Algorithms

Currently supported:
- ✅ **MEMIT** - Full support
- ✅ **ROME** - Full support
- ❌ **FT** (Fine-tuning) - Not supported (modifies all weights)
- ❌ **MEND** - Not supported (uses hypernetwork)

## Tips and Best Practices

1. **Storage**: Deltas are small, so it's safe to save them for all evaluation cases
2. **Versioning**: The metadata includes timestamps and hparams for reproducibility
3. **Model Compatibility**: Always use the same base model that was used for editing
4. **Multiple Edits**: When using `num_edits > 1`, all edits are combined into a single delta file
5. **Device**: Deltas are saved on CPU and automatically moved to the correct device when loaded
6. **Weight Restoration**: By default, original weights are restored after each edit. Use `--no_restore` only if you have a specific reason to keep the edited state

## Troubleshooting

### "No module named 'util.delta_saver'"
Make sure you're running from the MEMIT repository root directory.

### Outputs don't match after reloading
- Verify you're using the same base model
- Check that the model hasn't been modified between saving and loading
- Ensure you're using the same random seed for generation

### Out of memory when loading
- The model needs to fit in memory (same as during evaluation)
- Use `--device cpu` if you don't have a GPU
- Delta files themselves are very small and shouldn't cause OOM

## Examples

### Example 1: Batch Evaluation with Saved Deltas

```bash
# Run evaluation and save all deltas
python experiments/evaluate.py \
    --alg_name MEMIT \
    --model_name gpt2-xl \
    --hparams_fname gpt2-xl.json \
    --ds_name cf \
    --dataset_size_limit 100 \
    --save_deltas

# Later, use any saved delta for inference
python experiments/inference_with_deltas.py \
    --delta-path results/MEMIT/run_000/deltas/case_42_deltas.pt \
    --interactive
```

### Example 2: Comparing Multiple Edits

```bash
# Generate output for first edit
python experiments/inference_with_deltas.py \
    --delta-path results/MEMIT/run_000/deltas/case_0_deltas.pt \
    --prompt "The Eiffel Tower is located in" > output_0.txt

# Generate output for second edit
python experiments/inference_with_deltas.py \
    --delta-path results/MEMIT/run_000/deltas/case_1_deltas.pt \
    --prompt "The Eiffel Tower is located in" > output_1.txt

# Compare
diff output_0.txt output_1.txt
```

### Example 3: Custom Script with Deltas

```python
#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer
from util.delta_saver import load_and_apply_deltas
from util.generate import generate_fast

# Load model and deltas
model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda()
tok = AutoTokenizer.from_pretrained("gpt2-xl")
tok.pad_token = tok.eos_token

model, metadata = load_and_apply_deltas(
    model,
    "results/MEMIT/run_000/deltas/case_0_deltas.pt"
)

# Custom generation
prompts = [
    "The Space Needle is in",
    "You can visit the Space Needle in",
    "The city of the Space Needle is",
]

for prompt in prompts:
    output = generate_fast(model, tok, [prompt], max_out_len=15)
    print(f"{prompt} -> {output[0]}")
```

## API Reference

See the docstrings in:
- `util/delta_saver.py` - Core save/load functions
- `experiments/inference_with_deltas.py` - Inference script
- `memit/memit_main.py` - MEMIT with `return_deltas` parameter
- `rome/rome_main.py` - ROME with `return_deltas` parameter
