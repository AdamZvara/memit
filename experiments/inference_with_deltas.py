#!/usr/bin/env python3
"""
Inference script for models with applied MEMIT/ROME deltas.

This script loads a pretrained model, applies saved weight deltas,
and runs text generation with the edited model.

Usage:
    python inference_with_deltas.py \
        --delta-path results/MEMIT/deltas/case_0_deltas.pt \
        --prompt "The Space Needle is located in the city of"

    # Interactive mode
    python inference_with_deltas.py \
        --delta-path results/MEMIT/deltas/case_0_deltas.pt \
        --interactive
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.delta_saver import load_and_apply_deltas
from util.generate import generate_fast


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a model edited using saved deltas"
    )
    parser.add_argument(
        "--delta-path",
        type=str,
        required=True,
        help="Path to saved deltas file (.pt)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (e.g., 'gpt2-xl'). If not provided, reads from metadata.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (enter prompts repeatedly)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k sampling parameter (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--compare-original",
        action="store_true",
        help="Also show output from original (unedited) model for comparison",
    )

    args = parser.parse_args()

    # Load deltas and metadata
    delta_path = Path(args.delta_path)
    if not delta_path.exists():
        raise FileNotFoundError(f"Delta file not found: {delta_path}")

    print("=" * 80)
    print("Loading deltas and metadata...")
    print("=" * 80)

    from util.delta_saver import load_deltas
    deltas, metadata = load_deltas(delta_path, device=args.device)

    # Determine model name
    model_name = args.model_name
    if model_name is None:
        if "model_name" in metadata:
            model_name = metadata["model_name"]
            print(f"Using model from metadata: {model_name}")
        else:
            raise ValueError(
                "Model name not found in metadata. Please specify --model-name"
            )

    # Print metadata information
    print("\nDelta Information:")
    print(f"  Case ID: {metadata.get('case_id', 'N/A')}")
    print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"  Modified weights: {len(deltas)}")

    if "requests" in metadata:
        print(f"\nEdit requests ({len(metadata['requests'])}):")
        for i, req in enumerate(metadata["requests"]):
            print(f"  {i+1}. Subject: {req.get('subject', 'N/A')}")
            print(f"     Target: {req.get('target_new', 'N/A')}")

    # Load model and tokenizer
    print("\n" + "=" * 80)
    print(f"Loading model: {model_name}")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(model_name).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Keep original model for comparison if requested
    original_model = None
    if args.compare_original:
        print("\nCreating copy of original model for comparison...")
        original_model = AutoModelForCausalLM.from_pretrained(model_name).to(args.device)

    # Apply deltas
    print("\n" + "=" * 80)
    from util.delta_saver import apply_deltas_to_model
    model = apply_deltas_to_model(model, deltas, verbose=True)
    print("=" * 80)

    # Generation function
    def generate_text(prompt: str):
        print("\n" + "-" * 80)
        print(f"Prompt: {prompt}")
        print("-" * 80)

        if args.compare_original and original_model is not None:
            print("\n[ORIGINAL MODEL]:")
            orig_output = generate_fast(
                original_model,
                tokenizer,
                [prompt],
                n_gen_per_prompt=1,
                max_out_len=args.max_tokens,
                top_k=args.top_k,
                temperature=args.temperature,
            )
            print(orig_output[0])

        print("\n[EDITED MODEL]:")
        edited_output = generate_fast(
            model,
            tokenizer,
            [prompt],
            n_gen_per_prompt=1,
            max_out_len=args.max_tokens,
            top_k=args.top_k,
            temperature=args.temperature,
        )
        print(edited_output[0])
        print("-" * 80)

    # Run generation
    print("\n" + "=" * 80)
    print("GENERATION")
    print("=" * 80)

    if args.interactive:
        print("\nInteractive mode. Type 'quit' or 'exit' to stop.")
        while True:
            try:
                prompt = input("\nEnter prompt: ").strip()
                if prompt.lower() in ["quit", "exit", "q"]:
                    break
                if not prompt:
                    continue
                generate_text(prompt)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break
    else:
        if args.prompt is None:
            # Use default prompts from metadata if available
            if "requests" in metadata and len(metadata["requests"]) > 0:
                print("\nNo prompt provided. Using prompt from first edit request:")
                prompt = metadata["requests"][0].get("prompt", "")
                if prompt:
                    generate_text(prompt)
                else:
                    print("Error: No prompt found in metadata. Please specify --prompt")
            else:
                print("Error: No prompt provided. Use --prompt or --interactive")
        else:
            generate_text(args.prompt)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
