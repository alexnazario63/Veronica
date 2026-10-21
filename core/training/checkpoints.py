"""Training-facing checkpoint API (implemented by the neural core)."""
from core.neural.checkpoint import load_checkpoint, save_checkpoint

__all__ = ["load_checkpoint", "save_checkpoint"]
