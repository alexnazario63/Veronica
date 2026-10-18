"""
Serialização segura de pesos e gerenciamento de Checkpoints da Veronica.
Salva tensores em formato binário NumPy (.npz) acompanhado de manifesto JSON
e hash de integridade SHA-256.
"""

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
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        arr = np.ascontiguousarray(state_dict[key], dtype=np.float64)
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def save_checkpoint(
    model: Module,
    checkpoint_dir: Union[str, Path],
    checkpoint_id: str,
    optimizer: Optional[Optimizer] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Salva pesos do modelo, estado do otimizador, configuração e hash SHA-256.
    """
    target_dir = Path(checkpoint_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    sha256_digest = _compute_state_sha256(state)

    weights_path = target_dir / f"{checkpoint_id}_weights.npz"
    manifest_path = target_dir / f"{checkpoint_id}_manifest.json"

    np.savez_compressed(weights_path, **state)

    manifest: Dict[str, Any] = {
        "checkpoint_id": checkpoint_id,
        "agent_name": "Veronica",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights_file": weights_path.name,
        "sha256": sha256_digest,
        "parameter_counts": model.parameter_counts(),
        "tensor_shapes": {k: list(v.shape) for k, v in state.items()},
        "metadata": metadata or {},
    }

    if optimizer is not None:
        opt_state = optimizer.state_dict()
        opt_weights: Dict[str, np.ndarray] = {}
        if "m" in opt_state:
            for k, arr in opt_state["m"].items():
                opt_weights[f"m_{k}"] = arr
        if "v" in opt_state:
            for k, arr in opt_state["v"].items():
                opt_weights[f"v_{k}"] = arr
        if opt_weights:
            opt_path = target_dir / f"{checkpoint_id}_optimizer.npz"
            np.savez_compressed(opt_path, **opt_weights)
            manifest["optimizer_file"] = opt_path.name
        manifest["optimizer_meta"] = {
            k: v for k, v in opt_state.items() if k not in ("m", "v")
        }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_checkpoint(
    model: Module,
    checkpoint_dir: Union[str, Path],
    checkpoint_id: str,
    optimizer: Optional[Optimizer] = None,
    verify_hash: bool = True,
) -> Dict[str, Any]:
    """
    Carrega um checkpoint validando integridade criptográfica SHA-256 antes de aplicar aos pesos.
    """
    target_dir = Path(checkpoint_dir)
    manifest_path = target_dir / f"{checkpoint_id}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifesto de checkpoint não encontrado: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_path = target_dir / manifest["weights_file"]
    if not weights_path.exists():
        raise FileNotFoundError(f"Arquivo de pesos não encontrado: {weights_path}")

    with np.load(weights_path) as data:
        state: Dict[str, np.ndarray] = {k: data[k].copy() for k in data.files}

    if verify_hash:
        actual_hash = _compute_state_sha256(state)
        expected_hash = manifest.get("sha256")
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(
                f"Falha de integridade no checkpoint '{checkpoint_id}': "
                f"SHA-256 esperado={expected_hash}, calculado={actual_hash}"
            )

    model.load_state_dict(state, strict=True)

    if optimizer is not None and "optimizer_meta" in manifest:
        opt_state = dict(manifest["optimizer_meta"])
        opt_file = manifest.get("optimizer_file")
        if opt_file and (target_dir / opt_file).exists():
            with np.load(target_dir / opt_file) as opt_data:
                m_dict: Dict[str, np.ndarray] = {}
                v_dict: Dict[str, np.ndarray] = {}
                for key in opt_data.files:
                    if key.startswith("m_"):
                        m_dict[key[2:]] = opt_data[key].copy()
                    elif key.startswith("v_"):
                        v_dict[key[2:]] = opt_data[key].copy()
                opt_state["m"] = m_dict
                opt_state["v"] = v_dict
        optimizer.load_state_dict(opt_state)

    return manifest
