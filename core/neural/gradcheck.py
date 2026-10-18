"""
Verificação Numérica de Gradientes por Diferenças Finitas Centrais (Gradcheck).
Compara os gradientes analíticos calculados pelo motor Autograd da Veronica com:
df/dx_i ≈ (f(x + eps * e_i) - f(x - eps * e_i)) / (2 * eps)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence
import numpy as np
from core.neural.tensor import Tensor


def numerical_gradient(
    func: Callable[[], Tensor],
    target: Tensor,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Calcula o gradiente numérico de `func()` (que deve retornar um Tensor escalar)
    em relação a `target` usando diferenças finitas centrais.
    """
    orig_data = target.data.copy()
    num_grad = np.zeros_like(orig_data, dtype=np.float64)

    it = np.nditer(orig_data, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_val = float(orig_data[idx])

        target.data[idx] = old_val + eps
        loss_plus = float(func().item())

        target.data[idx] = old_val - eps
        loss_minus = float(func().item())

        target.data[idx] = old_val
        num_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()

    return num_grad


def gradcheck(
    func: Callable[[], Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Executa forward + backward analítico e compara com o gradiente numérico por diferenças finitas.
    Lança AssertionError caso algum tensor apresente erro acima da tolerância.
    """
    for inp in inputs:
        inp.zero_grad()

    out = func()
    if out.size != 1:
        raise ValueError("gradcheck requer uma função com saída escalar (size == 1).")
    out.backward()

    report: Dict[str, Any] = {"passed": True, "max_relative_error": 0.0, "details": []}

    for i, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue
        if inp.grad is None:
            raise AssertionError(f"Input #{i} possui grad=None após backward().")

        analytic = inp.grad.copy()
        numeric = numerical_gradient(func, inp, eps=eps)

        diff_norm = float(np.linalg.norm(analytic - numeric))
        denom = max(float(np.linalg.norm(analytic) + np.linalg.norm(numeric)), 1e-12)
        rel_error = diff_norm / denom
        max_abs_diff = float(np.max(np.abs(analytic - numeric)))

        passed_tensor = bool(
            np.allclose(analytic, numeric, atol=atol, rtol=rtol) or rel_error <= rtol
        )
        report["max_relative_error"] = max(report["max_relative_error"], rel_error)
        report["details"].append(
            {
                "index": i,
                "name": inp.name or f"input_{i}",
                "shape": inp.shape,
                "relative_error": rel_error,
                "max_abs_diff": max_abs_diff,
                "passed": passed_tensor,
            }
        )
        if not passed_tensor:
            report["passed"] = False
            raise AssertionError(
                f"Falha no gradcheck para '{inp.name or f'input_{i}'}' (shape={inp.shape}): "
                f"rel_error={rel_error:.3e}, max_abs_diff={max_abs_diff:.3e}"
            )

    return report
