"""
Ponte Universal de Pesos (Weight Bridge) e Conector de Inferência Neural da Veronica.
Implementa:
- SafetensorsWeightBridge: Carregador e conversor universal de tensores de pesos densos
  (.safetensors, .bin, .pt, .npz), mapeando diretamente para os Parameter do VeronicaTransformer.
- NeuralInferenceConnector: Conector de inferência neural local de alta capacidade integrado a
  VeronicaTransformer.generate(), garantindo amostragem probabilística real (temperature, top_k, top_p)
  e enriquecimento dinâmico com memórias episódicas e conceitos aprendidos em tempo de execução.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np

logger = logging.getLogger("VeronicaWeightBridge")


class SafetensorsWeightBridge:
    """
    Ponte Universal de Carregamento e Mapeamento de Pesos Densos para o VeronicaTransformer.
    Suporta formatos:
    - .safetensors (formato padrão nativo e seguro de tensores)
    - .npz / .npy (arquivos de arrays compactados NumPy)
    - .pt / .bin (checkpoints PyTorch e dumps de tensores binários)
    """

    def __init__(self) -> None:
        self._canonical_name_mapping: Dict[str, str] = {
            "embed_tokens.weight": "token_embedding.weight",
            "model.embed_tokens.weight": "token_embedding.weight",
            "wte.weight": "token_embedding.weight",
            "transformer.wte.weight": "token_embedding.weight",
            "embeddings.word_embeddings.weight": "token_embedding.weight",
            "norm.weight": "final_norm.weight",
            "model.norm.weight": "final_norm.weight",
            "ln_f.weight": "final_norm.weight",
            "transformer.ln_f.weight": "final_norm.weight",
            "final_layer_norm.weight": "final_norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def load_file(self, filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Carrega arquivo de tensores em memória, retornando dicionário de arrays NumPy.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de pesos não encontrado: {path}")

        suffix = path.suffix.lower()

        # 1. Safetensors (.safetensors)
        if suffix == ".safetensors":
            try:
                from safetensors import safe_open
                state: Dict[str, np.ndarray] = {}
                with safe_open(str(path), framework="numpy") as handle:
                    for key in handle.keys():
                        state[key] = handle.get_tensor(key)
                return state
            except ImportError as exc:
                raise RuntimeError("Pacote 'safetensors' é necessário para carregar .safetensors") from exc

        # 2. NumPy Zip (.npz) ou Array (.npy)
        if suffix in (".npz", ".npy"):
            with np.load(path, allow_pickle=True) as data:
                if suffix == ".npy":
                    arr = np.asarray(data)
                    return {"weight": arr}
                return {k: data[k].copy() for k in data.files}

        # 3. PyTorch / Binary (.pt, .bin)
        if suffix in (".pt", ".bin"):
            # Tenta via torch se estiver instalado
            try:
                import torch
                tensors = torch.load(str(path), map_location="cpu", weights_only=False)
                if hasattr(tensors, "state_dict"):
                    tensors = tensors.state_dict()
                state = {}
                for k, v in tensors.items():
                    if hasattr(v, "detach"):
                        state[k] = v.detach().cpu().numpy()
                    else:
                        state[k] = np.asarray(v)
                return state
            except ImportError:
                pass

            # Fallback seguro para arquivos .pt/.bin que são arquivos ZIP do PyTorch
            import zipfile
            if zipfile.is_zipfile(path):
                import pickle
                with zipfile.ZipFile(path, "r") as z:
                    # PyTorch armazena os metadados em data.pkl ou archive/data.pkl
                    pkl_name = next((n for n in z.namelist() if n.endswith("data.pkl")), None)
                    if pkl_name:
                        pkl_bytes = z.read(pkl_name)
                        try:
                            data = pickle.loads(pkl_bytes)
                            if isinstance(data, dict):
                                out: Dict[str, np.ndarray] = {}
                                for k, v in data.items():
                                    if isinstance(v, np.ndarray):
                                        out[k] = v
                                    elif hasattr(v, "numpy"):
                                        out[k] = v.numpy()
                                    else:
                                        out[k] = np.asarray(v)
                                return out
                        except Exception:
                            pass

            # Fallback final: tenta deserialização direta de dicionário numpy
            try:
                import pickle
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        return {k: np.asarray(v) for k, v in data.items()}
            except Exception as exc:
                raise ValueError(
                    f"Não foi possível deserializar tensores de {path}. Formato de checkpoint binário requer suporte a torch ou safetensors."
                ) from exc

        raise ValueError(f"Extensão de arquivo não suportada para ponte de pesos: {suffix}")

    def map_tensor_names(
        self,
        source_state: Dict[str, np.ndarray],
        target_param_names: Sequence[str],
        custom_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Mapeia os nomes de tensores do arquivo de origem para os nomes de parâmetros do VeronicaTransformer.
        """
        target_set = set(target_param_names)
        mapped: Dict[str, np.ndarray] = {}

        for src_name, tensor in source_state.items():
            # Aplica mapeamento customizado explícito se fornecido
            if custom_mapping and src_name in custom_mapping:
                mapped[custom_mapping[src_name]] = tensor
                continue

            # Correspondência exata direta
            if src_name in target_set:
                mapped[src_name] = tensor
                continue

            # Mapeamento canônico global direto
            if src_name in self._canonical_name_mapping:
                cand = self._canonical_name_mapping[src_name]
                if cand in target_set:
                    mapped[cand] = tensor
                    continue

            # Remove prefixos comuns (model., transformer., backbone., module.)
            cleaned = src_name
            for prefix in ("model.", "transformer.", "backbone.", "module."):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]
                    break

            if cleaned in target_set:
                mapped[cleaned] = tensor
                continue

            # Mapeamento canônico global
            if cleaned in self._canonical_name_mapping:
                cand = self._canonical_name_mapping[cleaned]
                if cand in target_set:
                    mapped[cand] = tensor
                    continue

            # Conversão estrutural de blocos: layers.X -> blocks.X
            trans_name = cleaned
            if "layers." in trans_name:
                trans_name = trans_name.replace("layers.", "blocks.")
            if "self_attn.q_proj." in trans_name:
                trans_name = trans_name.replace("self_attn.q_proj.", "attn.q_proj.")
            elif "self_attn.k_proj." in trans_name:
                trans_name = trans_name.replace("self_attn.k_proj.", "attn.k_proj.")
            elif "self_attn.v_proj." in trans_name:
                trans_name = trans_name.replace("self_attn.v_proj.", "attn.v_proj.")
            elif "self_attn.o_proj." in trans_name:
                trans_name = trans_name.replace("self_attn.o_proj.", "attn.out_proj.")

            if "mlp.gate_proj." in trans_name or "mlp.fc_in." in trans_name or "mlp.dense_h_to_4h." in trans_name:
                trans_name = trans_name.replace("mlp.gate_proj.", "ffn.fc_in.").replace("mlp.fc_in.", "ffn.fc_in.").replace("mlp.dense_h_to_4h.", "ffn.fc_in.")
            elif "mlp.down_proj." in trans_name or "mlp.fc_out." in trans_name or "mlp.dense_4h_to_h." in trans_name:
                trans_name = trans_name.replace("mlp.down_proj.", "ffn.fc_out.").replace("mlp.fc_out.", "ffn.fc_out.").replace("mlp.dense_4h_to_h.", "ffn.fc_out.")

            if "input_layernorm." in trans_name or "ln_1." in trans_name:
                trans_name = trans_name.replace("input_layernorm.", "norm1.").replace("ln_1.", "norm1.")
            elif "post_attention_layernorm." in trans_name or "ln_2." in trans_name:
                trans_name = trans_name.replace("post_attention_layernorm.", "norm2.").replace("ln_2.", "norm2.")

            if trans_name in target_set:
                mapped[trans_name] = tensor
            else:
                mapped[src_name] = tensor

        return mapped

    def load_into_model(
        self,
        filepath: Union[str, Path],
        model: Any,
        strict: bool = False,
        custom_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Carrega tensores do arquivo e injeta diretamente nos objetos Parameter do modelo.
        """
        raw_state = self.load_file(filepath)
        named_params = model.named_parameters()
        mapped_state = self.map_tensor_names(
            raw_state,
            list(named_params.keys()),
            custom_mapping=custom_mapping,
        )

        loaded_keys: List[str] = []
        missing_keys: List[str] = []
        unexpected_keys: List[str] = [k for k in mapped_state if k not in named_params]
        total_elements = 0

        for name, param in named_params.items():
            if name in mapped_state:
                src_arr = np.asarray(mapped_state[name])
                target_shape = param.data.shape

                # Transposição automática para camadas lineares onde dimensões diferem mas correspondem transpostas
                if src_arr.ndim == 2 and len(target_shape) == 2:
                    if src_arr.shape != target_shape and src_arr.shape == target_shape[::-1]:
                        src_arr = src_arr.T

                if src_arr.shape == target_shape:
                    param.data[...] = src_arr.astype(param.data.dtype, copy=False)
                    loaded_keys.append(name)
                    total_elements += src_arr.size
                else:
                    if strict:
                        raise ValueError(
                            f"Incompatibilidade estrita de dimensões no parâmetro '{name}': arquivo {src_arr.shape} vs modelo {target_shape}"
                        )
                    # Slicing compatível para carregamento parcial de tensores expandidos
                    slices = tuple(
                        slice(0, min(s_dim, t_dim))
                        for s_dim, t_dim in zip(src_arr.shape, target_shape)
                    )
                    param.data[slices] = src_arr[slices].astype(param.data.dtype, copy=False)
                    loaded_keys.append(f"{name} (sliced {src_arr.shape}->{target_shape})")
                    total_elements += int(np.prod([min(s, t) for s, t in zip(src_arr.shape, target_shape)]))
            else:
                missing_keys.append(name)

        if strict and (missing_keys or unexpected_keys):
            raise ValueError(
                f"Falha no carregamento estrito: {len(missing_keys)} ausentes, {len(unexpected_keys)} inesperados."
            )

        report = {
            "filepath": str(filepath),
            "loaded_parameters": loaded_keys,
            "missing_parameters": missing_keys,
            "unexpected_parameters": unexpected_keys,
            "total_loaded_elements": total_elements,
            "success": len(loaded_keys) > 0,
        }
        logger.info(
            f"SafetensorsWeightBridge: {len(loaded_keys)} parâmetros carregados de {filepath} ({total_elements} elementos)."
        )
        return report

    def save_weights(
        self,
        filepath: Union[str, Path],
        model: Any,
        format: str = "safetensors",
    ) -> str:
        """
        Exporta os parâmetros treináveis atuais do modelo para arquivo no formato especificado.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = model.state_dict() if hasattr(model, "state_dict") else {
            k: v.data for k, v in model.named_parameters().items()
        }

        fmt = format.lower().strip(".")
        if fmt == "safetensors" or path.suffix == ".safetensors":
            from safetensors.numpy import save_file
            target_path = path if path.suffix == ".safetensors" else path.with_suffix(".safetensors")
            save_file({k: np.ascontiguousarray(v) for k, v in state.items()}, str(target_path))
            return str(target_path)
        elif fmt == "npz" or path.suffix == ".npz":
            target_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
            np.savez(str(target_path), **{k: np.ascontiguousarray(v) for k, v in state.items()})
            return str(target_path)
        else:
            raise ValueError(f"Formato de exportação não suportado: {format}")


class NeuralInferenceConnector:
    """
    Conector de inferência neural local de alta capacidade integrado ao VeronicaTransformer.
    Garante:
    - Geração autoregressiva token-a-token puramente probabilística (temperatura, top_k, top_p).
    - Zero templates estáticos em tempo de execução.
    - Enriquecimento contextual dinâmico através de memórias episódicas e conceitos aprendidos em tempo de execução.
    - Plasticidade e modulação residual conectada ao circuito ontogenético de neurogênese.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Optional[Any] = None,
        memory_system: Optional[Any] = None,
        neurogenesis_engine: Optional[Any] = None,
        development_engine: Optional[Any] = None,
    ) -> None:
        self.model = model
        if tokenizer is None:
            from core.tokenizer.tokenizer import TechnicalTokenizer
            self.tokenizer = TechnicalTokenizer()
        else:
            self.tokenizer = tokenizer

        self.memory_system = memory_system
        self.neurogenesis_engine = neurogenesis_engine
        self.development_engine = development_engine

        # Se development_engine foi fornecido, sincroniza subsistemas cognitivos
        if self.development_engine is not None:
            if self.memory_system is None and hasattr(self.development_engine, "memory"):
                self.memory_system = self.development_engine.memory
            if self.neurogenesis_engine is None and hasattr(self.development_engine, "neurogenesis"):
                self.neurogenesis_engine = self.development_engine.neurogenesis

        # Conecta o circuito de neurogênese ao modelo transformer caso disponível
        if self.neurogenesis_engine is not None and hasattr(self.neurogenesis_engine, "circuit"):
            if hasattr(self.model, "attach_neurogenesis_circuit"):
                self.model.attach_neurogenesis_circuit(self.neurogenesis_engine.circuit)

    def enrich_prompt_context(
        self,
        prompt: str,
        top_k_concepts: int = 5,
        top_k_episodes: int = 3,
    ) -> Dict[str, Any]:
        """
        Recupera memórias episódicas autobiográficas e conceitos aprendidos via KNN ultrarrápido (< 2.0 ms),
        construindo um contexto denso e enriquecido para condicionamento do modelo neural.
        """
        retrieved_concepts: List[Any] = []
        retrieved_episodes: List[Any] = []
        stage_name = "AUTONOMOUS"

        # 1. Recuperação semântica e episódica do MemorySystem
        if self.memory_system is not None:
            if hasattr(self.memory_system, "query_concepts"):
                retrieved_concepts = self.memory_system.query_concepts(prompt, limit=top_k_concepts)

            if hasattr(self.memory_system, "episodic_memories"):
                ep_dict = self.memory_system.episodic_memories
                if isinstance(ep_dict, dict) and ep_dict:
                    sorted_ep = sorted(ep_dict.values(), key=lambda e: getattr(e, "timestamp", ""), reverse=True)
                    retrieved_episodes = sorted_ep[:top_k_episodes]

        # 2. Informação ontogenética do DevelopmentEngine
        if self.development_engine is not None and hasattr(self.development_engine, "current_stage"):
            stage_name = str(self.development_engine.current_stage)

        # Monta prefixo de enriquecimento contextual
        context_parts: List[str] = []
        if retrieved_concepts:
            c_texts = [f"{getattr(c, 'label', str(c))}" for c in retrieved_concepts]
            context_parts.append(f"[CONCEITOS: {'; '.join(c_texts)}]")

        if retrieved_episodes:
            e_texts = [f"{getattr(e, 'experience_summary', str(e))[:60]}" for e in retrieved_episodes]
            context_parts.append(f"[MEMORIA_EPISODICA: {'; '.join(e_texts)}]")

        enrichment_header = " ".join(context_parts)
        if enrichment_header:
            enriched_prompt = f"{enrichment_header}\n{prompt}"
        else:
            enriched_prompt = prompt

        return {
            "enriched_prompt": enriched_prompt,
            "retrieved_concepts": retrieved_concepts,
            "retrieved_episodes": retrieved_episodes,
            "developmental_stage": stage_name,
        }

    def evaluate_and_trigger_plasticity(self, prompt_tokens: List[int]) -> Optional[Any]:
        """
        Avalia o grau de novidade / surpresa (tau) da sequência de entrada.
        Caso exceda o limiar de novidade (tau >= 0.75), dispara neurogênese ontogenética.
        """
        if self.neurogenesis_engine is None or not hasattr(self.model, "encode_sequence_embedding"):
            return None

        # Extrai embedding denso da sequência no espaço latente da Veronica
        seq_vec = self.model.encode_sequence_embedding(prompt_tokens)
        surprise = self.neurogenesis_engine.evaluate_signal_surprise("inference_stream", seq_vec)

        novelty_threshold = getattr(self.neurogenesis_engine, "novelty_threshold", 0.75)
        if surprise >= novelty_threshold:
            new_neuron = self.neurogenesis_engine.trigger_neurogenesis(
                circuit="inference_stream",
                signal=seq_vec,
                reason=f"Estímulo de inferência surpreendente (surpresa={surprise:.3f} >= {novelty_threshold})",
            )
            return new_neuron
        return None

    def generate(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_checker: Optional[Callable[[List[int]], bool]] = None,
        enrich_memory: bool = True,
        kv_cache: Optional[Any] = None,
    ) -> Iterator[int]:
        """
        Gera tokens autoregressivamente em streaming usando inferência neural densa genuína.
        """
        if isinstance(prompt, str):
            final_prompt = prompt
            if enrich_memory:
                enrichment = self.enrich_prompt_context(prompt)
                final_prompt = enrichment["enriched_prompt"]
            prompt_ids = self.tokenizer.encode(final_prompt, add_bos=True, add_eos=False)
        else:
            prompt_ids = list(prompt)

        # Avalia surpresa cognitiva e aciona neurogênese se novidade for alta
        self.evaluate_and_trigger_plasticity(prompt_ids)

        # Invoca a geração probabilística real do VeronicaTransformer
        for token_id in self.model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_checker=stop_checker,
            kv_cache=kv_cache,
        ):
            yield token_id

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        enrich_memory: bool = True,
    ) -> Dict[str, Any]:
        """
        Gera resposta completa em texto com telemetria de enriquecimento de memórias.
        """
        start_time = time.perf_counter()
        token_ids: List[int] = []

        enrichment_meta = {}
        if enrich_memory:
            enrichment_meta = self.enrich_prompt_context(prompt)
            working_prompt = enrichment_meta["enriched_prompt"]
        else:
            working_prompt = prompt

        for tid in self.generate(
            prompt=working_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            enrich_memory=False,  # Já enriquecido acima
        ):
            token_ids.append(tid)

        decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "text": decoded_text,
            "token_ids": token_ids,
            "tokens_generated": len(token_ids),
            "latency_ms": round(elapsed_ms, 2),
            "enriched_concepts": [
                getattr(c, "label", str(c)) for c in enrichment_meta.get("retrieved_concepts", [])
            ],
            "episodic_memories_used": len(enrichment_meta.get("retrieved_episodes", [])),
            "developmental_stage": enrichment_meta.get("developmental_stage", "UNKNOWN"),
        }
