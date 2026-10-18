"""
Arquitetura Completa VeronicaTransformer:
- Embeddings treináveis
- Positional Encoding / RoPE
- Blocos Transformer (Pre-Norm, Multi-Head Causal Self-Attention, Feed-Forward Network, Residual Connections)
- Cabeça de previsão de tokens (LM Head)
- Exportação de topologia e telemetria interna para o Neural Observatory
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.neural.tensor import Tensor
from core.neural.layers import Embedding, LayerNorm, Linear, Module, RMSNorm
from core.neural.scalability import activation_checkpoint
from core.transformer.config import VeronicaConfig
from core.transformer.positional import (
    LearnedPositionalEncoding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)
from core.transformer.attention import MultiHeadCausalAttention


class FeedForwardNetwork(Module):
    """Rede Feed-Forward de 2 camadas por posição com ativação GELU ou ReLU."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "gelu",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation_name = activation.lower()
        self.fc_in = Linear(d_model, d_ff, bias=True, seed=seed)
        self.fc_out = Linear(d_ff, d_model, bias=True, seed=seed + 1)
        self.last_activation_stats: Dict[str, float] = {}

    def forward(self, x: Tensor, capture_diagnostics: bool = False) -> Tensor:
        h = self.fc_in(x)
        if self.activation_name == "relu":
            act = h.relu()
        else:
            act = h.gelu()

        if capture_diagnostics:
            arr = act.data
            zero_frac = float(np.mean(np.abs(arr) < 1e-4))
            sat_frac = float(np.mean(np.abs(arr) > 3.0))
            self.last_activation_stats = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "max": float(np.max(arr)),
                "min": float(np.min(arr)),
                "dead_or_zero_ratio": zero_frac,
                "saturation_ratio": sat_frac,
            }

        return self.fc_out(act)


class TransformerBlock(Module):
    """Bloco Transformer Pre-Norm com conexões residuais."""

    def __init__(
        self,
        config: VeronicaConfig,
        layer_idx: int,
        rope: Optional[RotaryPositionalEmbedding] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        seed_base = config.seed + layer_idx * 100

        if config.norm_type.lower() == "layernorm":
            self.norm1: Module = LayerNorm(config.d_model)
            self.norm2: Module = LayerNorm(config.d_model)
        else:
            self.norm1 = RMSNorm(config.d_model)
            self.norm2 = RMSNorm(config.d_model)

        self.attn = MultiHeadCausalAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            rope=rope,
            seed=seed_base + 10,
        )
        self.ffn = FeedForwardNetwork(
            d_model=config.d_model,
            d_ff=config.d_ff,
            activation=config.activation,
            seed=seed_base + 50,
        )

    def forward(
        self,
        x: Tensor,
        past_kv: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        use_cache: bool = False,
        capture_diagnostics: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[np.ndarray, np.ndarray]]]:
        attn_out, present_kv = self.attn(
            self.norm1(x),
            past_kv=past_kv,
            use_cache=use_cache,
            capture_diagnostics=capture_diagnostics,
        )
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x), capture_diagnostics=capture_diagnostics)
        x = x + ffn_out
        return x, present_kv


class VeronicaTransformer(Module):
    """
    Modelo Neural Autoregressivo Nativo da Veronica para Engenharia de Redes.
    """

    def __init__(self, config: VeronicaConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = Embedding(
            config.vocab_size, config.d_model, seed=config.seed
        )

        self.rope: Optional[RotaryPositionalEmbedding] = None
        self.pos_encoding: Optional[Module] = None

        if config.positional_type == "rope":
            self.rope = RotaryPositionalEmbedding(
                head_dim=config.head_dim,
                max_seq_len=config.max_seq_len,
            )
        elif config.positional_type == "learned":
            self.pos_encoding = LearnedPositionalEncoding(
                max_seq_len=config.max_seq_len,
                d_model=config.d_model,
                seed=config.seed + 1,
            )
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(
                max_seq_len=config.max_seq_len,
                d_model=config.d_model,
            )

        self.blocks: List[TransformerBlock] = [
            TransformerBlock(config, layer_idx=i, rope=self.rope)
            for i in range(config.n_layers)
        ]

        if config.norm_type.lower() == "layernorm":
            self.final_norm: Module = LayerNorm(config.d_model)
        else:
            self.final_norm = RMSNorm(config.d_model)

        self.lm_head = Linear(
            config.d_model, config.vocab_size, bias=False, seed=config.seed + 999
        )
        self.neurogenesis_circuit: Optional[Any] = None

    def attach_neurogenesis_circuit(self, circuit: Any) -> None:
        """Conecta o circuito de neurogênese dinâmica ao modelo autoregressivo."""
        self.neurogenesis_circuit = circuit


    def forward(
        self,
        input_ids: np.ndarray,
        past_kvs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        use_cache: bool = False,
        capture_diagnostics: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tuple[np.ndarray, np.ndarray]]]]:
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[np.newaxis, :]

        offset = int(past_kvs[0][0].shape[2]) if past_kvs else 0
        x = self.token_embedding(ids)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x, offset=offset)

        new_kvs: List[Tuple[np.ndarray, np.ndarray]] = []
        for idx, block in enumerate(self.blocks):
            layer_past = past_kvs[idx] if past_kvs is not None else None
            if self.config.use_activation_checkpointing and self.training and not use_cache:
                x, present_kv = activation_checkpoint(
                    block,
                    x,
                    past_kv=layer_past,
                    use_cache=use_cache,
                    capture_diagnostics=capture_diagnostics,
                )
            else:
                x, present_kv = block(
                    x,
                    past_kv=layer_past,
                    use_cache=use_cache,
                    capture_diagnostics=capture_diagnostics,
                )
            if use_cache and present_kv is not None:
                new_kvs.append(present_kv)

        if self.neurogenesis_circuit is not None:
            x, _ = self.neurogenesis_circuit.forward_integrated(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, (new_kvs if use_cache else None)

    def encode_sequence_embedding(self, input_ids: List[int]) -> np.ndarray:
        """
        Extrai uma representação vetorial densa normalizada (mean pooling da última camada oculta)
        usando os próprios parâmetros aprendidos da Veronica para recuperação semântica (RAG).
        """
        if not input_ids:
            return np.zeros((self.config.d_model,), dtype=np.float64)
        ids = np.asarray(input_ids[: self.config.max_seq_len], dtype=np.int64)[np.newaxis, :]
        x = self.token_embedding(ids)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x, offset=0)
        for block in self.blocks:
            x, _ = block(x, use_cache=False, capture_diagnostics=False)
        if self.neurogenesis_circuit is not None:
            x, _ = self.neurogenesis_circuit.forward_integrated(x)
        x = self.final_norm(x)
        vec = np.mean(x.data[0], axis=0)

        norm = float(np.linalg.norm(vec))
        return vec / max(norm, 1e-12)

    def export_architecture_graph(self) -> Dict[str, Any]:
        """
        Exporta a estrutura completa da rede neural para o Neural Observatory (Seção 16.3.A),
        incluindo nós, conexões, dimensões de tensores, equações matemáticas, estatísticas
        agregadas de pesos/gradientes e amostra de neurônios individuais.
        """
        counts = self.parameter_counts()
        named = self.named_parameters()

        def param_stats(param_names: List[str]) -> Dict[str, Any]:
            arrays = [named[n].data for n in param_names if n in named]
            grads = [
                named[n].grad
                for n in param_names
                if n in named and named[n].grad is not None
            ]
            if not arrays:
                return {"params": 0, "trainable": False}
            flat = np.concatenate([a.ravel() for a in arrays])
            grad_norm = (
                float(np.sqrt(sum(float(np.sum(g ** 2)) for g in grads)))
                if grads
                else 0.0
            )
            trainable = any(named[n].requires_grad for n in param_names if n in named)
            return {
                "params": int(flat.size),
                "trainable": trainable,
                "status": "trainable" if trainable else "frozen",
                "weight_mean": float(np.mean(flat)),
                "weight_std": float(np.std(flat)),
                "weight_min": float(np.min(flat)),
                "weight_max": float(np.max(flat)),
                "weight_l2_norm": float(np.linalg.norm(flat)),
                "grad_l2_norm": grad_norm,
                "memory_bytes": int(sum(a.nbytes for a in arrays)),
            }

        nodes: List[Dict[str, Any]] = [
            {
                "id": "input_tokens",
                "label": "Camada de Entrada (Token IDs)",
                "type": "input",
                "tensor_shape": ["B", "T"],
                "formula": "x_in ∈ {0, ..., V-1}^{B×T}",
                "stats": {"params": 0, "trainable": False, "status": "fixed"},
            },
            {
                "id": "token_embedding",
                "label": f"Embeddings Técnicos ({self.config.vocab_size}×{self.config.d_model})",
                "type": "embedding",
                "tensor_shape": ["B", "T", self.config.d_model],
                "formula": "E(x_in) = W_E[x_in]",
                "stats": param_stats(["token_embedding.weight"]),
            },
            {
                "id": "positional_encoding",
                "label": f"Codificação Posicional ({self.config.positional_type.upper()})",
                "type": "positional",
                "tensor_shape": ["B", "T", self.config.d_model],
                "formula": "RoPE: R_{Θ,m} x" if self.config.positional_type == "rope" else "x + P_{pos}",
                "stats": param_stats(["pos_encoding.weight"]),
            },
        ]

        edges: List[Dict[str, str]] = [
            {"source": "input_tokens", "target": "token_embedding", "dim": f"(B, T) -> (B, T, {self.config.d_model})"},
            {"source": "token_embedding", "target": "positional_encoding", "dim": f"(B, T, {self.config.d_model})"},
        ]

        prev_node = "positional_encoding"
        for i, block in enumerate(self.blocks):
            attn_id = f"block_{i}_mha"
            ffn_id = f"block_{i}_ffn"

            attn_params = [
                f"blocks.{i}.norm1.weight",
                f"blocks.{i}.norm1.gamma",
                f"blocks.{i}.norm1.beta",
                f"blocks.{i}.attn.q_proj.weight",
                f"blocks.{i}.attn.q_proj.bias",
                f"blocks.{i}.attn.k_proj.weight",
                f"blocks.{i}.attn.k_proj.bias",
                f"blocks.{i}.attn.v_proj.weight",
                f"blocks.{i}.attn.v_proj.bias",
                f"blocks.{i}.attn.out_proj.weight",
                f"blocks.{i}.attn.out_proj.bias",
            ]
            ffn_params = [
                f"blocks.{i}.norm2.weight",
                f"blocks.{i}.norm2.gamma",
                f"blocks.{i}.norm2.beta",
                f"blocks.{i}.ffn.fc_in.weight",
                f"blocks.{i}.ffn.fc_in.bias",
                f"blocks.{i}.ffn.fc_out.weight",
                f"blocks.{i}.ffn.fc_out.bias",
            ]

            nodes.append(
                {
                    "id": attn_id,
                    "label": f"Bloco {i+1}: Multi-Head Attention ({self.config.n_heads} cabeças × {self.config.head_dim}d)",
                    "type": "attention",
                    "layer_index": i,
                    "heads": self.config.n_heads,
                    "head_dim": self.config.head_dim,
                    "tensor_shape": ["B", "T", self.config.d_model],
                    "formula": "h' = x + MHA(Norm(x)), Softmax(QK^T / √d_k + M)V",
                    "stats": param_stats(attn_params),
                    "attention_diagnostics": block.attn.last_attention_stats,
                }
            )
            nodes.append(
                {
                    "id": ffn_id,
                    "label": f"Bloco {i+1}: Feed-Forward ({self.config.d_model} -> {self.config.d_ff} -> {self.config.d_model})",
                    "type": "ffn",
                    "layer_index": i,
                    "tensor_shape": ["B", "T", self.config.d_model],
                    "formula": f"y = h' + W_2({self.config.activation.upper()}(W_1(Norm(h')) + b_1)) + b_2",
                    "stats": param_stats(ffn_params),
                    "activation_diagnostics": block.ffn.last_activation_stats,
                }
            )
            edges.append({"source": prev_node, "target": attn_id, "dim": f"(B, T, {self.config.d_model})"})
            edges.append({"source": attn_id, "target": ffn_id, "dim": f"(B, T, {self.config.d_model})"})
            prev_node = ffn_id

        if self.neurogenesis_circuit is not None and getattr(self.neurogenesis_circuit, "neurons", None):
            circuit_nodes = [n.export_meta() for n in self.neurogenesis_circuit.neurons]
            nodes.append(
                {
                    "id": "neurogenesis_associative_circuit",
                    "label": f"Circuito de Neurogênese Própria ({len(circuit_nodes)} neurônios vivos nascidos)",
                    "type": "neurogenesis",
                    "tensor_shape": ["B", "T", self.config.d_model],
                    "formula": "x_mod = x + 0.15 · Σ (act_i · W_out,i · LTP_i · Mielina_i)",
                    "stats": {
                        "params": len(circuit_nodes) * (self.config.d_model * 2 + 1),
                        "trainable": True,
                        "status": "living_plastic_neurogenesis",
                        "born_neurons_count": len(circuit_nodes),
                    },
                    "born_neurons": circuit_nodes,
                }
            )
            edges.append({"source": prev_node, "target": "neurogenesis_associative_circuit", "dim": f"(B, T, {self.config.d_model})"})
            prev_node = "neurogenesis_associative_circuit"

        nodes.append(
            {
                "id": "final_norm",
                "label": f"Normalização Final ({self.config.norm_type.upper()})",
                "type": "norm",
                "tensor_shape": ["B", "T", self.config.d_model],
                "formula": "RMSNorm(x) = (x / √(mean(x²) + ε)) · γ",
                "stats": param_stats(["final_norm.weight", "final_norm.gamma", "final_norm.beta"]),
            }
        )
        nodes.append(
            {
                "id": "lm_head",
                "label": f"Cabeça de Previsão de Tokens ({self.config.d_model} -> {self.config.vocab_size})",
                "type": "output",
                "tensor_shape": ["B", "T", self.config.vocab_size],
                "formula": "logits = h_final @ W_vocab",
                "stats": param_stats(["lm_head.weight"]),
            }
        )
        edges.append({"source": prev_node, "target": "final_norm", "dim": f"(B, T, {self.config.d_model})"})
        edges.append({"source": "final_norm", "target": "lm_head", "dim": f"(B, T, {self.config.vocab_size})"})

        # Amostra de neurônios individuais (slice 8x8 do primeiro bloco) para visualização micro
        w_sample = self.blocks[0].ffn.fc_in.weight.data[:8, :8]
        b_sample = (
            self.blocks[0].ffn.fc_in.bias.data[:8]
            if self.blocks[0].ffn.fc_in.bias is not None
            else np.zeros((8,))
        )
        micro_neurons = []
        for n_idx in range(w_sample.shape[1]):
            micro_neurons.append(
                {
                    "neuron_id": f"B1_FFN_N{n_idx}",
                    "bias": float(b_sample[n_idx]),
                    "activation": self.config.activation.upper(),
                    "incoming_weights": [float(w) for w in w_sample[:, n_idx]],
                }
            )

        if self.neurogenesis_circuit is not None and getattr(self.neurogenesis_circuit, "neurons", None):
            for n in self.neurogenesis_circuit.neurons[:8]:
                w_in = [float(val) for val in n.weight.data.flatten()[:8]]
                micro_neurons.append(
                    {
                        "neuron_id": f"BORN_{n.neuron_id[:16]}",
                        "bias": float(n.bias.data[0]) if n.bias is not None else 0.0,
                        "activation": f"{n.activation_name.upper()} (LTP={n.synaptic_strength:.2f})",
                        "incoming_weights": w_in,
                        "is_born_neuron": True,
                        "trigger_concept": n.trigger_concept,
                    }
                )

        return {
            "agent_name": "Veronica",
            "config": self.config.to_dict(),
            "parameter_counts": counts,
            "nodes": nodes,
            "edges": edges,
            "micro_neuron_slice": micro_neurons,
        }

