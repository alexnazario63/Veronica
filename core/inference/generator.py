"""
Motor de Inferência Autoregressiva da Veronica com KV-Cache,
amostragem Top-K / Top-P (Nucleus), controle de temperatura,
penalidade de repetição e suporte a cancelamento em tempo real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
import numpy as np

from core.tokenizer.tokenizer import TechnicalTokenizer
from core.transformer.model import VeronicaTransformer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.2
    top_k: int = 20
    top_p: float = 0.9
    greedy: bool = True
    repetition_penalty: float = 1.1
    use_cache: bool = True
    seed: int = 42


class PrefixKVCache:
    """Small exact-prefix cache for model prompt states and final prompt logits."""

    def __init__(self, max_entries: int = 32) -> None:
        from collections import OrderedDict
        self.max_entries = max(1, int(max_entries))
        self._entries: Any = OrderedDict()

    def get(self, token_ids: List[int]) -> Optional[Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]]:
        key = tuple(token_ids)
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        kvs, logits = entry
        return ([(k.copy(), v.copy()) for k, v in kvs], logits.copy())

    def put(self, token_ids: List[int], kvs: List[Tuple[np.ndarray, np.ndarray]], logits: np.ndarray) -> None:
        key = tuple(token_ids)
        self._entries[key] = ([(k.copy(), v.copy()) for k, v in kvs], logits.copy())
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()


class AutoregressiveGenerator:
    """
    Executa decodificação autoregressiva usando o modelo neural próprio VeronicaTransformer.
    """

    def __init__(
        self,
        model: VeronicaTransformer,
        tokenizer: TechnicalTokenizer,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prefix_cache = PrefixKVCache()

    def _active_tool_json_prefix(
        self, prompt_ids: List[int], generated_ids: List[int]
    ) -> Optional[str]:
        """Return the open tool-call body, if the latest call has not closed."""
        call_id = self.tokenizer.token_to_id.get("[TOOL_CALL]")
        close_id = self.tokenizer.token_to_id.get("[/TOOL_CALL]")
        if call_id is None:
            return None
        ids = prompt_ids + generated_ids
        call_pos = next((i for i in range(len(ids) - 1, -1, -1) if ids[i] == call_id), None)
        if call_pos is None:
            return None
        close_pos = next(
            (i for i in range(len(ids) - 1, call_pos, -1) if ids[i] == close_id),
            None,
        )
        if close_pos is not None:
            return None
        return self.tokenizer.decode(ids[call_pos + 1 :], skip_special_tokens=False)

    def _sample_next_token(
        self,
        logits_1d: np.ndarray,
        generated_ids: List[int],
        gen_cfg: GenerationConfig,
        rng: np.random.Generator,
    ) -> Tuple[int, float]:
        scores = logits_1d.astype(np.float64, copy=True)

        # Bloqueia <pad> e <unk> durante a geração
        scores[self.tokenizer.pad_id] = -1e9
        scores[self.tokenizer.unk_id] = -1e9

        # Penalidade de repetição suave
        if gen_cfg.repetition_penalty > 1.0 and generated_ids:
            for prev_id in set(generated_ids[-32:]):
                if scores[prev_id] > 0:
                    scores[prev_id] /= gen_cfg.repetition_penalty
                else:
                    scores[prev_id] *= gen_cfg.repetition_penalty

        if gen_cfg.greedy or gen_cfg.temperature <= 1e-5:
            shifted = scores - np.max(scores)
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            best_id = int(np.argmax(probs))
            return best_id, float(probs[best_id])

        scaled = scores / max(gen_cfg.temperature, 1e-4)

        # Top-K filtering
        if gen_cfg.top_k > 0 and gen_cfg.top_k < scaled.size:
            top_k_indices = np.argpartition(scaled, -gen_cfg.top_k)[-gen_cfg.top_k :]
            mask = np.ones_like(scaled, dtype=bool)
            mask[top_k_indices] = False
            scaled[mask] = -1e9

        shifted = scaled - np.max(scaled)
        probs = np.exp(shifted)
        probs /= np.sum(probs)

        # Top-P (Nucleus) filtering
        if 0.0 < gen_cfg.top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_idx]
            cumulative = np.cumsum(sorted_probs)
            cutoff_mask = cumulative > gen_cfg.top_p
            if np.any(cutoff_mask):
                first_exceed = int(np.argmax(cutoff_mask))
                keep_idx = sorted_idx[: max(first_exceed + 1, 1)]
                filtered = np.zeros_like(probs)
                filtered[keep_idx] = probs[keep_idx]
                probs = filtered / np.sum(filtered)

        chosen_id = int(rng.choice(len(probs), p=probs))
        return chosen_id, float(probs[chosen_id])

    def stream_generate(
        self,
        prompt: str,
        gen_cfg: Optional[GenerationConfig] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        capture_diagnostics: bool = True,
        tool_schema: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, object]]:
        """
        Gera tokens progressivamente, emitindo cada fragmento decodificado e sua probabilidade.
        """
        cfg = gen_cfg or GenerationConfig()
        max_seq_len = max(2, int(self.model.config.max_seq_len))
        # Keep the prompt and generated tokens inside the model's positional
        # window. Oversized profiles used to read past RoPE's cache and crash.
        max_new_tokens = max(1, min(int(cfg.max_new_tokens), max_seq_len - 1))
        rng = np.random.default_rng(cfg.seed)
        self.model.eval()

        prompt_ids = self.tokenizer.encode(
            prompt,
            add_bos=True,
            add_eos=False,
            max_length=max(1, max_seq_len - max_new_tokens),
        )
        if not prompt_ids:
            prompt_ids = [self.tokenizer.bos_id]

        generated_ids: List[int] = []
        past_kvs = None
        cached_prefix = self.prefix_cache.get(prompt_ids) if cfg.use_cache else None
        first_logits: Optional[np.ndarray] = None
        if cached_prefix is not None:
            past_kvs, first_logits = cached_prefix

        for step in range(max_new_tokens):
            if should_cancel is not None and should_cancel():
                break

            if first_logits is not None:
                next_logits = first_logits
                first_logits = None
            else:
                if cfg.use_cache and past_kvs is not None:
                    step_input = np.array([[generated_ids[-1]]], dtype=np.int64)
                else:
                    full_seq = (prompt_ids + generated_ids)[-self.model.config.max_seq_len :]
                    step_input = np.array([full_seq], dtype=np.int64)

                logits, past_kvs = self.model.forward(
                    step_input,
                    past_kvs=past_kvs if cfg.use_cache else None,
                    use_cache=cfg.use_cache,
                    capture_diagnostics=(capture_diagnostics and step == 0),
                )
                next_logits = logits.data[0, -1, :]
                if step == 0 and cfg.use_cache and past_kvs is not None:
                    self.prefix_cache.put(prompt_ids, past_kvs, next_logits)
            # Tool calls switch immediately into schema-constrained JSON mode.
            if tool_schema is not None:
                json_prefix = self._active_tool_json_prefix(prompt_ids, generated_ids)
                if json_prefix is not None:
                    from core.inference.constrained_json import ConstrainedJSONDecoder
                    next_logits = ConstrainedJSONDecoder(
                        self.tokenizer, tool_schema
                    ).mask_logits(next_logits, json_prefix)
            next_id, token_prob = self._sample_next_token(
                next_logits, generated_ids, cfg, rng
            )

            if next_id == self.tokenizer.eos_id:
                break

            generated_ids.append(next_id)
            token_str = self.tokenizer.decode([next_id], skip_special_tokens=True)
            yield {
                "step": step + 1,
                "token_id": next_id,
                "token_text": token_str,
                "confidence": token_prob,
            }
            # A completed constrained call is a complete model action. Stopping
            # here prevents unconstrained tokens from being appended afterward.
            if next_id == self.tokenizer.token_to_id.get("[/TOOL_CALL]"):
                break

    def generate_text(
        self,
        prompt: str,
        gen_cfg: Optional[GenerationConfig] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        tool_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, object]:
        pieces: List[str] = []
        confidences: List[float] = []
        token_ids: List[int] = []

        for item in self.stream_generate(
            prompt,
            gen_cfg=gen_cfg,
            should_cancel=should_cancel,
            tool_schema=tool_schema,
        ):
            pieces.append(str(item["token_text"]))
            confidences.append(float(item["confidence"]))
            token_ids.append(int(item["token_id"]))

        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return {
            "text": "".join(pieces).strip(),
            "token_ids": token_ids,
            "tokens_generated": len(token_ids),
            "mean_token_confidence": mean_conf,
        }
