"""Safe model checkpoints using the non-executable Safetensors format."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from core.neural.layers import Module
from core.neural.optimizers import Optimizer


def _compute_state_sha256(state_dict: Dict[str, np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state_dict):
        arr = np.ascontiguousarray(state_dict[key])
        hasher.update(key.encode("utf-8"))
        hasher.update(str(arr.dtype).encode("ascii"))
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _safetensors():
    try:
        from safetensors import safe_open
        from safetensors.numpy import save_file
    except ImportError as exc:  # pragma: no cover - configured dependency
        raise RuntimeError("Install safetensors to read or write model checkpoints") from exc
    return safe_open, save_file


def _load_safetensors(path: Path) -> Dict[str, np.ndarray]:
    safe_open, _ = _safetensors()
    state: Dict[str, np.ndarray] = {}
    # safe_open reads through Safetensors' memory-mapped file handle and exposes
    # only named tensor entries; no Python object deserialization is involved.
    with safe_open(str(path), framework="numpy") as handle:
        for key in handle.keys():
            state[key] = handle.get_tensor(key)
    return state


def save_checkpoint(model: Module, checkpoint_dir: Union[str, Path], checkpoint_id: str,
                    optimizer: Optional[Optimizer] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    target_dir = Path(checkpoint_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    digest = _compute_state_sha256(state)
    _, save_file = _safetensors()
    weights_path = target_dir / f"{checkpoint_id}_weights.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in state.items()}, str(weights_path))
    manifest: Dict[str, Any] = {
        "checkpoint_id": checkpoint_id,
        "agent_name": "Veronica",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights_file": weights_path.name,
        "format": "safetensors",
        "sha256": digest,
        "parameter_counts": model.parameter_counts(),
        "tensor_shapes": {k: list(v.shape) for k, v in state.items()},
        "metadata": metadata or {},
    }
    if optimizer is not None:
        opt_state = optimizer.state_dict()
        opt_weights: Dict[str, np.ndarray] = {}
        for family in ("m", "v"):
            for key, value in opt_state.get(family, {}).items():
                opt_weights[f"{family}_{key}"] = np.asarray(value)
        if opt_weights:
            opt_path = target_dir / f"{checkpoint_id}_optimizer.safetensors"
            save_file({k: np.ascontiguousarray(v) for k, v in opt_weights.items()}, str(opt_path))
            manifest["optimizer_file"] = opt_path.name
            manifest["optimizer_format"] = "safetensors"
        manifest["optimizer_meta"] = {k: v for k, v in opt_state.items() if k not in ("m", "v")}
    (target_dir / f"{checkpoint_id}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _read_tensor_file(path: Path, file_format: str) -> Dict[str, np.ndarray]:
    if file_format == "safetensors" or path.suffix == ".safetensors":
        return _load_safetensors(path)
    # Compatibility for existing NumPy-only checkpoints. np.load is restricted
    # to arrays and explicitly refuses pickle/object payloads.
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def load_checkpoint(model: Module, checkpoint_dir: Union[str, Path], checkpoint_id: str,
                    optimizer: Optional[Optimizer] = None,
                    verify_hash: bool = True) -> Dict[str, Any]:
    target_dir = Path(checkpoint_dir)
    manifest_path = target_dir / f"{checkpoint_id}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_path = target_dir / manifest["weights_file"]
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")
    state = _read_tensor_file(weights_path, manifest.get("format", "npz"))
    if verify_hash:
        expected = manifest.get("sha256")
        actual = _compute_state_sha256(state)
        # Old manifests hashed float64 arrays without dtype metadata.
        if expected and actual != expected:
            legacy = hashlib.sha256()
            for key in sorted(state):
                legacy.update(key.encode("utf-8"))
                legacy.update(np.ascontiguousarray(state[key], dtype=np.float64).tobytes())
            if legacy.hexdigest() != expected:
                raise ValueError(f"Checkpoint SHA-256 mismatch for '{checkpoint_id}'")
    model.load_state_dict(state, strict=True)
    if optimizer is not None and "optimizer_meta" in manifest:
        opt_state = dict(manifest["optimizer_meta"])
        opt_file = manifest.get("optimizer_file")
        if opt_file and (target_dir / opt_file).exists():
            arrays = _read_tensor_file(target_dir / opt_file, manifest.get("optimizer_format", "npz"))
            opt_state["m"] = {k[2:]: v for k, v in arrays.items() if k.startswith("m_")}
            opt_state["v"] = {k[2:]: v for k, v in arrays.items() if k.startswith("v_")}
        optimizer.load_state_dict(opt_state)
    return manifest
