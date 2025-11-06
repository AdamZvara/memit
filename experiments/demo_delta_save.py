#!/usr/bin/env python3
"""
Demo script showing how to:
1. Edit a model with MEMIT
2. Save the deltas
3. Load deltas and apply to a fresh model
4. Run inference

This is a minimal example for testing the delta save/load functionality.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit import MEMITHyperParams, apply_memit_to_model
from util.delta_saver import save_deltas, load_and_apply_deltas
from util.generate import generate_fast


def main():
    print("=" * 80)
    print("MEMIT Delta Save/Load Demo")
    print("=" * 80)

    # Configuration
    model_name = "gpt2-xl"
    hparams_path = "hparams/MEMIT/gpt2-xl.json"
    save_dir = Path("./demo_deltas")

    # Example edit request
    request = {
        "prompt": "The Space Needle is located in the city of",
        "subject": "Space Needle",
        "target_new": {"str": " Berlin"},
        "case_id": 0,
    }

    print("\n" + "=" * 80)
    print("Step 1: Load model and apply MEMIT edit")
    print("=" * 80)

    # Load model
    print(f"\nLoading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    # Load hyperparameters
    hparams = MEMITHyperParams.from_json(hparams_path)
    print(f"Loaded hyperparameters: {hparams}")

    # Test original model
    print("\n[Before edit] Generating with original model:")
    original_output = generate_fast(
        model, tok, [request["prompt"]], n_gen_per_prompt=1, max_out_len=20
    )
    print(f"  {original_output[0]}")

    # Apply MEMIT with return_deltas=True
    print("\nApplying MEMIT edit...")
    edited_model, weights_info = apply_memit_to_model(
        model,
        tok,
        [request],
        hparams,
        copy=False,
        return_orig_weights=True,
        return_deltas=True,
    )

    # Test edited model
    print("\n[After edit] Generating with edited model:")
    edited_output = generate_fast(
        edited_model, tok, [request["prompt"]], n_gen_per_prompt=1, max_out_len=20
    )
    print(f"  {edited_output[0]}")

    print("\n" + "=" * 80)
    print("Step 2: Save deltas to disk")
    print("=" * 80)

    # Save deltas
    deltas = weights_info["deltas"]
    delta_path, metadata_path = save_deltas(
        deltas,
        save_dir,
        case_id=request["case_id"],
        model_name=model_name,
        hparams=hparams,
        requests=[request],
    )

    print(f"\nDeltas saved successfully!")
    print(f"  Delta file: {delta_path}")
    print(f"  Metadata file: {metadata_path}")

    print("\n" + "=" * 80)
    print("Step 3: Restore original model weights")
    print("=" * 80)

    # Restore original weights (simulating fresh model)
    print("\nRestoring original weights...")
    with torch.no_grad():
        for k, v in weights_info.items():
            if k != "deltas":  # Skip the deltas key
                from util import nethook
                nethook.get_parameter(model, k)[...] = v.cuda()

    # Verify restoration
    print("\n[After restoration] Generating with restored model:")
    restored_output = generate_fast(
        model, tok, [request["prompt"]], n_gen_per_prompt=1, max_out_len=20
    )
    print(f"  {restored_output[0]}")

    print("\n" + "=" * 80)
    print("Step 4: Load deltas and apply to fresh model")
    print("=" * 80)

    # Load and apply deltas
    print(f"\nLoading deltas from: {delta_path}")
    model_reloaded, metadata = load_and_apply_deltas(model, delta_path, verbose=True)

    # Test reloaded model
    print("\n[After reloading deltas] Generating with reloaded model:")
    reloaded_output = generate_fast(
        model_reloaded, tok, [request["prompt"]], n_gen_per_prompt=1, max_out_len=20
    )
    print(f"  {reloaded_output[0]}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\nOriginal output:  {original_output[0]}")
    print(f"Edited output:    {edited_output[0]}")
    print(f"Restored output:  {restored_output[0]}")
    print(f"Reloaded output:  {reloaded_output[0]}")

    if edited_output[0] == reloaded_output[0]:
        print("\n✓ SUCCESS! Edited and reloaded outputs match!")
    else:
        print("\n✗ WARNING: Edited and reloaded outputs don't match!")

    print(f"\nDelta files are saved in: {save_dir}")
    print("You can use these deltas with: python experiments/inference_with_deltas.py")
    print(f'  Example: python experiments/inference_with_deltas.py --delta-path {delta_path} --prompt "{request["prompt"]}"')

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
