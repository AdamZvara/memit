"""
Utilities for saving and loading MEMIT/ROME weight deltas.

This module provides functions to:
- Save weight deltas (low-rank factorizations) to disk
- Load deltas from saved files
- Apply deltas to fresh model instances for inference
"""

import json
import torch
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from datetime import datetime

from util import nethook


def save_deltas(
    deltas: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    save_dir: Path,
    case_id: int,
    model_name: str,
    hparams: Any,
    requests: Optional[list] = None,
) -> Tuple[Path, Path]:
    """
    Save weight deltas and metadata to disk.

    Args:
        deltas: Dictionary mapping weight names to (key_matrix, value_matrix) tuples
        save_dir: Directory to save deltas (will be created if doesn't exist)
        case_id: Identifier for this edit case
        model_name: Name of the model being edited (e.g., "gpt2-xl")
        hparams: Hyperparameters object used for editing
        requests: Optional list of edit requests (for metadata)

    Returns:
        Tuple of (delta_path, metadata_path)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save delta tensors
    delta_path = save_dir / f"case_{case_id}_deltas.pt"

    # Convert deltas to CPU and detach for saving
    deltas_cpu = {
        weight_name: (key_mat.detach().cpu(), val_mat.detach().cpu())
        for weight_name, (key_mat, val_mat) in deltas.items()
    }

    torch.save(deltas_cpu, delta_path)

    # Save metadata
    metadata_path = save_dir / f"case_{case_id}_metadata.json"
    metadata = {
        "case_id": case_id,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "delta_shapes": {
            name: {
                "key_matrix_shape": list(key_mat.shape),
                "value_matrix_shape": list(val_mat.shape),
            }
            for name, (key_mat, val_mat) in deltas.items()
        },
        "modified_weights": list(deltas.keys()),
        "hparams": {
            k: v for k, v in vars(hparams).items()
            if not k.startswith("_") and not callable(v)
        },
    }

    # Add request information if provided
    if requests is not None:
        # Handle both single request and list of requests
        if isinstance(requests, list):
            metadata["requests"] = [
                {
                    "prompt": req.get("prompt", ""),
                    "subject": req.get("subject", ""),
                    "target_new": req.get("target_new", {}).get("str", "") if isinstance(req.get("target_new"), dict) else req.get("target_new", ""),
                }
                for req in requests
            ]
        else:
            metadata["requests"] = [{
                "prompt": requests.get("prompt", ""),
                "subject": requests.get("subject", ""),
                "target_new": requests.get("target_new", {}).get("str", "") if isinstance(requests.get("target_new"), dict) else requests.get("target_new", ""),
            }]

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved deltas to {delta_path}")
    print(f"Saved metadata to {metadata_path}")

    return delta_path, metadata_path


def load_deltas(
    delta_path: Path,
    device: str = "cuda"
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], dict]:
    """
    Load weight deltas and metadata from disk.

    Args:
        delta_path: Path to saved deltas file (.pt)
        device: Device to load tensors to ("cuda" or "cpu")

    Returns:
        Tuple of (deltas_dict, metadata_dict)
    """
    delta_path = Path(delta_path)

    # Load deltas
    deltas = torch.load(delta_path, map_location=device)

    # Load metadata
    metadata_path = delta_path.parent / delta_path.name.replace("_deltas.pt", "_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
        print(f"Warning: No metadata file found at {metadata_path}")

    print(f"Loaded deltas from {delta_path}")
    print(f"Modified weights: {list(deltas.keys())}")

    return deltas, metadata


def apply_deltas_to_model(
    model: torch.nn.Module,
    deltas: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    verbose: bool = True
) -> torch.nn.Module:
    """
    Apply weight deltas to a model in-place.

    This replicates the logic from memit_main.apply_memit_to_model()
    and rome_main.apply_rome_to_model().

    Args:
        model: PyTorch model to modify
        deltas: Dictionary mapping weight names to (key_matrix, value_matrix) tuples
        verbose: Whether to print progress information

    Returns:
        Modified model (same object, modified in-place)
    """
    if verbose:
        print(f"Applying {len(deltas)} weight deltas to model...")

    with torch.no_grad():
        for weight_name, (key_mat, val_mat) in deltas.items():
            # Compute update matrix from low-rank factorization
            upd_matrix = key_mat @ val_mat.T

            # Get the parameter and add the update
            weight = nethook.get_parameter(model, weight_name)

            if verbose:
                print(f"  Updating {weight_name}: {weight.shape}")

            # Apply update (convert to same dtype as original weight)
            weight[...] += upd_matrix.to(weight.device).to(weight.dtype)

    if verbose:
        print("Deltas applied successfully!")

    return model


def load_and_apply_deltas(
    model: torch.nn.Module,
    delta_path: Path,
    verbose: bool = True
) -> Tuple[torch.nn.Module, dict]:
    """
    Convenience function to load deltas and apply them to a model.

    Args:
        model: PyTorch model to modify
        delta_path: Path to saved deltas file
        verbose: Whether to print progress information

    Returns:
        Tuple of (modified_model, metadata)
    """
    deltas, metadata = load_deltas(delta_path, device=str(model.device))
    model = apply_deltas_to_model(model, deltas, verbose=verbose)
    return model, metadata
