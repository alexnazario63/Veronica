"""
Sistema de Memórias Humanóides da Veronica (Seção de Desenvolvimento Cognitivo).
Implementa:
- Memória de Trabalho (Working Memory): buffer atencional imediato de capacidade limitada.
- Memória Episódica (Episodic Memory): arquivo autobiográfico de experiências vividas.
- Memória Semântica (Semantic Memory): grafo relacional de conceitos, regras e causalidades.
- Ciclo de Consolidação e Sono Cognitivo: transferência e cristalização de memórias de curto para longo prazo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import io
import math
import json
from pathlib import Path
import struct
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid


_MEMORY_STOP_WORDS = {"qual", "quais", "como", "quando", "onde", "para", "sobre", "com", "uma", "the", "what", "with", "from", "que", "por", "das", "dos", "and", "you", "your", "is", "are"}


@dataclass
class WorkingMemoryItem:
    """Item ativo no buffer de memória de trabalho da Veronica."""
    item_id: str
    concept: str
    modality: str  # "VISION", "READING", "AUDIO", "TEXT"
    summary: str
    attention_weight: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodicMemoryRecord:
    """Registro autobiográfico de uma experiência vivida pela Veronica."""
    memory_id: str
    timestamp: str
    modality: str  # "VISION", "READING", "CONVERSATION", "DIAGNOSTIC", "REFLECTION"
    experience_summary: str
    perceptual_details: Dict[str, Any]
    emotive_valence: str  # "CURIOSIDADE", "SATISFACAO", "ALERTA", "FASCINIO", "RESOLUCAO"
    salience: float  # Relevância cognitiva de 0.0 a 1.0
    consolidation_level: float  # 0.0 (fresca) até 1.0 (totalmente consolidada a longo prazo)
    associated_concepts: List[str]
    created_neuron_ids: List[str] = field(default_factory=list)


@dataclass
class ConceptNode:
    """Nó conceitual na memória semântica da Veronica."""
    concept_id: str
    label: str
    category: str  # "PROTOCOL", "TOPOLOGY", "HARDWARE", "COMMAND", "ANOMALY", "RULE"
    description: str
    activation_count: int = 1
    stability: float = 0.5  # Resistência ao esquecimento
    first_learned_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_accessed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ConceptEdge:
    """Aresta relacional entre conceitos na memória semântica."""
    source_id: str
    target_id: str
    relation: str  # "CAUSES", "RESOLVES", "PART_OF", "REQUIRES", "CONNECTED_TO", "CONFLICTS_WITH"
    strength: float = 0.5  # Peso sináptico relacional


class BinaryMachineMemoryStore:
    """
    Motor de Armazenamento Binário de Alta Performance para Memórias da Veronica.
    Utiliza formato binário proprietário de linguagem de máquina/baixo nível (VRNC_MEM v2.0),
    estruturado via `struct` com alinhamento Big-Endian para máxima velocidade de I/O,
    serialização direta de buffers de atenção, episódios autobiográficos e grafos semânticos.
    """

    MAGIC_HEADER: bytes = b"VRNC_MEM\x02\x00"  # 10 bytes: 'VRNC_MEM' + version 2.0
    HEADER_STRUCT: str = ">10sdIIIII"          # magic (10s), epoch_ts (d), wm (I), ep (I), nodes (I), edges (I), cycles (I)
    HEADER_SIZE: int = struct.calcsize(HEADER_STRUCT)

    def __init__(self, memory_system: Optional[Any] = None) -> None:
        self.memory_system = memory_system

    @staticmethod
    def _pack_bytes(data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + data

    @staticmethod
    def _pack_str(s: str) -> bytes:
        return BinaryMachineMemoryStore._pack_bytes(s.encode("utf-8"))

    @staticmethod
    def _pack_json(obj: Any) -> bytes:
        return BinaryMachineMemoryStore._pack_str(json.dumps(obj, ensure_ascii=False))

    @staticmethod
    def _pack_str_list(items: List[str]) -> bytes:
        buf = io.BytesIO()
        buf.write(struct.pack(">I", len(items)))
        for s in items:
            buf.write(BinaryMachineMemoryStore._pack_str(s))
        return buf.getvalue()

    @staticmethod
    def _unpack_bytes(data: bytes, offset: int) -> Tuple[bytes, int]:
        if offset + 4 > len(data):
            raise ValueError("Offset excede o tamanho dos dados ao ler comprimento de bytes.")
        length = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        if offset + length > len(data):
            raise ValueError("Registro binário corrompido: comprimento do payload excede o buffer.")
        res = data[offset : offset + length]
        return res, offset + length

    @staticmethod
    def _unpack_str(data: bytes, offset: int) -> Tuple[str, int]:
        b, new_offset = BinaryMachineMemoryStore._unpack_bytes(data, offset)
        return b.decode("utf-8", errors="replace"), new_offset

    @staticmethod
    def _unpack_json(data: bytes, offset: int) -> Tuple[Any, int]:
        s, new_offset = BinaryMachineMemoryStore._unpack_str(data, offset)
        try:
            return json.loads(s), new_offset
        except Exception:
            return {}, new_offset

    @staticmethod
    def _unpack_str_list(data: bytes, offset: int) -> Tuple[List[str], int]:
        if offset + 4 > len(data):
            raise ValueError("Offset excede o tamanho dos dados ao ler contagem de lista.")
        count = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        items: List[str] = []
        for _ in range(count):
            s, offset = BinaryMachineMemoryStore._unpack_str(data, offset)
            items.append(s)
        return items, offset

    def serialize_to_bytes(self, memory_system: Optional[Any] = None) -> bytes:
        mem = memory_system or self.memory_system
        if not mem:
            raise ValueError("Uma instância de MemorySystem deve ser fornecida para serialização.")

        buf = io.BytesIO()
        now_ts = datetime.now(timezone.utc).timestamp()

        wm_items = mem.working_memory
        ep_items = list(mem.episodic_memories.values())
        nodes = list(mem.semantic_nodes.values())
        edges = mem.semantic_edges
        cycles = mem.consolidation_history

        # Header
        header_bytes = struct.pack(
            self.HEADER_STRUCT,
            self.MAGIC_HEADER,
            now_ts,
            len(wm_items),
            len(ep_items),
            len(nodes),
            len(edges),
            len(cycles),
        )
        buf.write(header_bytes)

        # 1. Working Memory Items
        for wm in wm_items:
            buf.write(self._pack_str(wm.item_id))
            buf.write(self._pack_str(wm.concept))
            buf.write(self._pack_str(wm.modality))
            buf.write(self._pack_str(wm.summary))
            buf.write(struct.pack(">d", float(wm.attention_weight)))
            buf.write(self._pack_str(wm.timestamp))
            buf.write(self._pack_json(wm.metadata))

        # 2. Episodic Memory Records
        for ep in ep_items:
            buf.write(self._pack_str(ep.memory_id))
            buf.write(self._pack_str(ep.timestamp))
            buf.write(self._pack_str(ep.modality))
            buf.write(self._pack_str(ep.experience_summary))
            buf.write(self._pack_str(ep.emotive_valence))
            buf.write(struct.pack(">dd", float(ep.salience), float(ep.consolidation_level)))
            buf.write(self._pack_str_list(ep.associated_concepts))
            buf.write(self._pack_str_list(ep.created_neuron_ids))
            buf.write(self._pack_json(ep.perceptual_details))

        # 3. Semantic Nodes
        for node in nodes:
            buf.write(self._pack_str(node.concept_id))
            buf.write(self._pack_str(node.label))
            buf.write(self._pack_str(node.category))
            buf.write(self._pack_str(node.description))
            buf.write(struct.pack(">Id", int(node.activation_count), float(node.stability)))
            buf.write(self._pack_str(node.first_learned_at))
            buf.write(self._pack_str(node.last_accessed_at))

        # 4. Semantic Edges
        for edge in edges:
            buf.write(self._pack_str(edge.source_id))
            buf.write(self._pack_str(edge.target_id))
            buf.write(self._pack_str(edge.relation))
            buf.write(struct.pack(">d", float(edge.strength)))

        # 5. Consolidation History
        for cycle in cycles:
            buf.write(self._pack_json(cycle))

        return buf.getvalue()

    def deserialize_from_bytes(
        self, data: bytes, memory_system: Optional[Any] = None
    ) -> Dict[str, Any]:
        if len(data) < self.HEADER_SIZE:
            raise ValueError(f"Dados binários insuficientes (tamanho {len(data)} < {self.HEADER_SIZE}).")

        magic, ts, wm_count, ep_count, node_count, edge_count, cycles_count = struct.unpack_from(
            self.HEADER_STRUCT, data, 0
        )
        if magic != self.MAGIC_HEADER:
            raise ValueError(f"Magic header inválido: esperado {self.MAGIC_HEADER!r}, obtido {magic!r}.")

        offset = self.HEADER_SIZE
        wm_list: List[WorkingMemoryItem] = []
        for _ in range(wm_count):
            item_id, offset = self._unpack_str(data, offset)
            concept, offset = self._unpack_str(data, offset)
            modality, offset = self._unpack_str(data, offset)
            summary, offset = self._unpack_str(data, offset)
            att_weight = struct.unpack_from(">d", data, offset)[0]
            offset += 8
            timestamp, offset = self._unpack_str(data, offset)
            metadata, offset = self._unpack_json(data, offset)
            wm_list.append(
                WorkingMemoryItem(
                    item_id=item_id,
                    concept=concept,
                    modality=modality,
                    summary=summary,
                    attention_weight=att_weight,
                    timestamp=timestamp,
                    metadata=metadata,
                )
            )

        ep_dict: Dict[str, EpisodicMemoryRecord] = {}
        for _ in range(ep_count):
            memory_id, offset = self._unpack_str(data, offset)
            timestamp, offset = self._unpack_str(data, offset)
            modality, offset = self._unpack_str(data, offset)
            summary, offset = self._unpack_str(data, offset)
            emotive_valence, offset = self._unpack_str(data, offset)
            salience, cons_level = struct.unpack_from(">dd", data, offset)
            offset += 16
            associated_concepts, offset = self._unpack_str_list(data, offset)
            created_neuron_ids, offset = self._unpack_str_list(data, offset)
            perceptual_details, offset = self._unpack_json(data, offset)
            ep_dict[memory_id] = EpisodicMemoryRecord(
                memory_id=memory_id,
                timestamp=timestamp,
                modality=modality,
                experience_summary=summary,
                perceptual_details=perceptual_details,
                emotive_valence=emotive_valence,
                salience=salience,
                consolidation_level=cons_level,
                associated_concepts=associated_concepts,
                created_neuron_ids=created_neuron_ids,
            )

        nodes_dict: Dict[str, ConceptNode] = {}
        for _ in range(node_count):
            concept_id, offset = self._unpack_str(data, offset)
            label, offset = self._unpack_str(data, offset)
            category, offset = self._unpack_str(data, offset)
            description, offset = self._unpack_str(data, offset)
            act_count, stability = struct.unpack_from(">Id", data, offset)
            offset += 12
            first_learned, offset = self._unpack_str(data, offset)
            last_accessed, offset = self._unpack_str(data, offset)
            nodes_dict[concept_id] = ConceptNode(
                concept_id=concept_id,
                label=label,
                category=category,
                description=description,
                activation_count=act_count,
                stability=stability,
                first_learned_at=first_learned,
                last_accessed_at=last_accessed,
            )

        edges_list: List[ConceptEdge] = []
        for _ in range(edge_count):
            source_id, offset = self._unpack_str(data, offset)
            target_id, offset = self._unpack_str(data, offset)
            relation, offset = self._unpack_str(data, offset)
            strength = struct.unpack_from(">d", data, offset)[0]
            offset += 8
            edges_list.append(
                ConceptEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation,
                    strength=strength,
                )
            )

        cycles_list: List[Dict[str, Any]] = []
        for _ in range(cycles_count):
            cycle_data, offset = self._unpack_json(data, offset)
            cycles_list.append(cycle_data)

        mem = memory_system or self.memory_system
        if mem:
            mem.working_memory = wm_list
            mem.episodic_memories = ep_dict
            mem.semantic_nodes.update(nodes_dict)
            mem.semantic_edges = edges_list
            mem.consolidation_history = cycles_list

        return {
            "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "working_memory_count": len(wm_list),
            "episodic_count": len(ep_dict),
            "semantic_nodes_count": len(nodes_dict),
            "semantic_edges_count": len(edges_list),
            "consolidation_cycles_count": len(cycles_list),
            "working_memory": wm_list,
            "episodic_memories": ep_dict,
            "semantic_nodes": nodes_dict,
            "semantic_edges": edges_list,
            "consolidation_history": cycles_list,
        }

    def save_binary_file(
        self, filepath: Union[str, Path], memory_system: Optional[Any] = None
    ) -> int:
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        raw_bytes = self.serialize_to_bytes(memory_system)
        # Escrita atômica via arquivo temporário para evitar race conditions em multi-threading
        tmp_target = target_path.with_suffix(f".tmp_{uuid.uuid4().hex[:6]}")
        tmp_target.write_bytes(raw_bytes)
        tmp_target.replace(target_path)
        return len(raw_bytes)

    def load_binary_file(
        self, filepath: Union[str, Path], memory_system: Optional[Any] = None
    ) -> Dict[str, Any]:
        target_path = Path(filepath)
        if not target_path.exists():
            raise FileNotFoundError(f"Arquivo binário de memória não encontrado: {target_path}")
        raw_bytes = target_path.read_bytes()
        return self.deserialize_from_bytes(raw_bytes, memory_system)

    @classmethod
    def inspect_binary_file(cls, filepath: Union[str, Path]) -> Dict[str, Any]:
        target_path = Path(filepath)
        if not target_path.exists():
            return {
                "file_path": str(target_path),
                "file_exists": False,
                "file_size_bytes": 0,
                "is_valid_binary": False,
                "magic_header": None,
                "error": "Arquivo não existe.",
            }
        file_size = target_path.stat().st_size
        if file_size < cls.HEADER_SIZE:
            return {
                "file_path": str(target_path),
                "file_exists": True,
                "file_size_bytes": file_size,
                "is_valid_binary": False,
                "magic_header": None,
                "error": f"Arquivo muito pequeno para conter cabeçalho ({file_size} < {cls.HEADER_SIZE}).",
            }
        with target_path.open("rb") as f:
            header_bytes = f.read(cls.HEADER_SIZE)
        if len(header_bytes) < cls.HEADER_SIZE:
            return {
                "file_path": str(target_path),
                "file_exists": True,
                "file_size_bytes": file_size,
                "is_valid_binary": False,
                "magic_header": None,
                "error": f"Leitura parcial do cabeçalho ({len(header_bytes)} < {cls.HEADER_SIZE}).",
            }
        try:
            magic, ts, wm_count, ep_count, node_count, edge_count, cycles_count = struct.unpack(
                cls.HEADER_STRUCT, header_bytes
            )
        except struct.error as e:
            return {
                "file_path": str(target_path),
                "file_exists": True,
                "file_size_bytes": file_size,
                "is_valid_binary": False,
                "magic_header": None,
                "error": f"Cabeçalho inválido: {e}",
            }
        is_valid = (magic == cls.MAGIC_HEADER)
        return {
            "file_path": str(target_path),
            "file_exists": True,
            "file_size_bytes": file_size,
            "is_valid_binary": is_valid,
            "magic_header": magic.decode("latin1", errors="replace"),
            "magic_header_hex": magic.hex(),
            "format_version": "2.0",
            "last_saved_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if is_valid else None,
            "working_memory_count": wm_count if is_valid else 0,
            "episodic_count": ep_count if is_valid else 0,
            "semantic_nodes_count": node_count if is_valid else 0,
            "semantic_edges_count": edge_count if is_valid else 0,
            "consolidation_cycles_count": cycles_count if is_valid else 0,
            "total_records": (wm_count + ep_count + node_count) if is_valid else 0,
        }


class MemorySystem:
    """
    Sistema integrado de memórias humanas da Veronica.
    """

    def __init__(self, storage_dir: Optional[Path] = None, max_working_memory: int = 7) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.max_working_memory = max_working_memory
        self.working_memory: List[WorkingMemoryItem] = []
        self.episodic_memories: Dict[str, EpisodicMemoryRecord] = {}
        self.semantic_nodes: Dict[str, ConceptNode] = {}
        self.semantic_edges: List[ConceptEdge] = []
        self.consolidation_history: List[Dict[str, Any]] = []

        # Armazenamento e persistência em arquivo binário próprio
        self.binary_store = BinaryMachineMemoryStore(memory_system=self)
        self.binary_filepath = (self.storage_dir / "veronica_machine_memory.bin") if self.storage_dir else None

        if self.binary_filepath and self.binary_filepath.exists():
            try:
                self.load_binary_file(self.binary_filepath)
            except Exception:
                self._bootstrap_semantic_base()
        else:
            # Inicializa alguns esquemas semânticos fundamentais
            self._bootstrap_semantic_base()
            if self.binary_filepath:
                try:
                    self.save_binary_file(self.binary_filepath)
                except Exception:
                    pass

    def _bootstrap_semantic_base(self) -> None:
        """Inicializa esquemas semânticos inatos de engenharia e cognição."""
        base_concepts = [
            ("BGP", "PROTOCOL", "Border Gateway Protocol — protocolo de roteamento entre Autonomous Systems"),
            ("OSPF", "PROTOCOL", "Open Shortest Path First — protocolo IGP Link-State"),
            ("MTU", "RULE", "Maximum Transmission Unit — tamanho máximo do pacote na camada 2/3"),
            ("OPTICAL_FIBER", "HARDWARE", "Fibra Óptica para enlaces de alta velocidade e transmissão DWDM"),
            ("TOPOLOGY_RING", "TOPOLOGY", "Topologia em anel com proteção e redundância de enlace"),
            ("ROLLBACK_SAFETY", "RULE", "Garantia determinística de reversão segura de configurações"),
        ]
        for name, cat, desc in base_concepts:
            cid = name.lower()
            self.semantic_nodes[cid] = ConceptNode(
                concept_id=cid,
                label=name,
                category=cat,
                description=desc,
                activation_count=3,
                stability=0.9,
            )

    # ==================== Memória de Trabalho ====================

    def push_working_memory(
        self,
        concept: str,
        modality: str,
        summary: str,
        attention_weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkingMemoryItem:
        """Adiciona um item ao foco atencional imediato (capacidade 7 ± 2)."""
        item_id = f"wm-{uuid.uuid4().hex[:6]}"
        now = datetime.now(timezone.utc).isoformat()
        item = WorkingMemoryItem(
            item_id=item_id,
            concept=concept,
            modality=modality,
            summary=summary,
            attention_weight=attention_weight,
            timestamp=now,
            metadata=metadata or {},
        )
        self.working_memory.append(item)
        if len(self.working_memory) > self.max_working_memory:
            # Desloca os mais antigos, que poderão ser consolidados
            self.working_memory.pop(0)
        return item

    def get_working_memory_snapshot(self) -> List[Dict[str, Any]]:
        return [asdict(item) for item in self.working_memory]

    def clear_working_memory(self) -> None:
        self.working_memory.clear()

    # ==================== Memória Episódica ====================

    def record_episodic_memory(
        self,
        modality: str,
        experience_summary: str,
        perceptual_details: Optional[Dict[str, Any]] = None,
        emotive_valence: str = "CURIOSIDADE",
        salience: float = 0.75,
        associated_concepts: Optional[List[str]] = None,
        created_neuron_ids: Optional[List[str]] = None,
    ) -> EpisodicMemoryRecord:
        """
        Grava um episódio autobiográfico na linha do tempo de vida da Veronica.
        """
        mem_id = f"EPM-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:6].upper()}"
        now = datetime.now(timezone.utc).isoformat()
        concepts = associated_concepts or []

        rec = EpisodicMemoryRecord(
            memory_id=mem_id,
            timestamp=now,
            modality=modality,
            experience_summary=experience_summary,
            perceptual_details=perceptual_details or {},
            emotive_valence=emotive_valence,
            salience=salience,
            consolidation_level=0.1,  # Memória recente, pronta para ser consolidada no ciclo de sono
            associated_concepts=concepts,
            created_neuron_ids=created_neuron_ids or [],
        )
        self.episodic_memories[mem_id] = rec

        # Adiciona também à memória de trabalho imediata
        self.push_working_memory(
            concept="; ".join(concepts) if concepts else modality,
            modality=modality,
            summary=experience_summary,
            attention_weight=salience,
            metadata={"memory_id": mem_id},
        )

        # Se houver conceitos associados, ativa/cria na memória semântica
        for c in concepts:
            self._touch_or_create_concept(c, category="EXPERIENCE", description="Concept associated with a conversation episode.")

        self.auto_save_binary()
        return rec

    def search_episodic_memories(
        self,
        query: str,
        modality: Optional[str] = None,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Rank relevant episodes with normalized lexical and phrase overlap, scoped by owner."""
        import re

        def normalized_tokens(text: str) -> List[str]:
            normalized = "".join(c for c in unicodedata.normalize("NFD", text.casefold()) if unicodedata.category(c) != "Mn")
            return [token for token in re.findall(r"[\w-]{3,}", normalized) if token not in _MEMORY_STOP_WORDS]

        query_tokens = normalized_tokens(query)
        if not query_tokens:
            recent = [
                rec for rec in self.episodic_memories.values()
                if (not modality or rec.modality == modality)
                and (not user_id or rec.perceptual_details.get("user_id") == user_id)
            ]
            recent.sort(key=lambda rec: rec.timestamp, reverse=True)
            return [asdict(rec) for rec in recent[:limit]]

        eligible = [
            rec for rec in self.episodic_memories.values()
            if (not modality or rec.modality == modality)
            and (not user_id or rec.perceptual_details.get("user_id") == user_id)
        ]
        tokenized = {
            rec.memory_id: normalized_tokens(
                rec.experience_summary + " " + json.dumps(rec.perceptual_details, ensure_ascii=False)
                + " " + " ".join(rec.associated_concepts)
            ) for rec in eligible
        }
        document_frequency: Dict[str, int] = {}
        for tokens in tokenized.values():
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        query_set = set(query_tokens)
        query_bigrams = set(zip(query_tokens, query_tokens[1:]))
        scored: List[Tuple[float, EpisodicMemoryRecord]] = []
        for rec in eligible:
            tokens = tokenized[rec.memory_id]
            words = set(tokens)
            overlap = query_set & words
            if not overlap:
                continue
            weighted_overlap = sum(
                math.log1p(len(eligible) / (1 + document_frequency.get(token, 0)))
                for token in overlap
            )
            total_query_weight = sum(
                math.log1p(len(eligible) / (1 + document_frequency.get(token, 0)))
                for token in query_set
            ) or 1.0
            token_score = weighted_overlap / total_query_weight
            episode_bigrams = set(zip(tokens, tokens[1:]))
            phrase_score = len(query_bigrams & episode_bigrams) / max(1, len(query_bigrams))
            exact_phrase = 0.2 if " ".join(query_tokens) in " ".join(tokens) else 0.0
            score = min(1.0, 0.72 * token_score + 0.2 * phrase_score + exact_phrase + rec.salience * 0.02)
            scored.append((score, rec))

        scored.sort(key=lambda pair: (pair[0], pair[1].timestamp), reverse=True)
        output = []
        for score, record in scored[:limit]:
            item = asdict(record)
            item["_relevance"] = round(score, 4)
            output.append(item)
        return output

    def forget_conversation(self, conversation_id: str, user_id: str) -> int:
        """Remove one owner's episodic records for a deleted conversation."""
        removed = [
            memory_id for memory_id, record in self.episodic_memories.items()
            if record.perceptual_details.get("conversation_id") == conversation_id
            and record.perceptual_details.get("user_id") == user_id
        ]
        concepts = {
            concept.strip().lower().replace(" ", "_")
            for memory_id in removed
            for concept in self.episodic_memories[memory_id].associated_concepts
        }
        for memory_id in removed:
            self.episodic_memories.pop(memory_id, None)
        for concept_id in concepts:
            node = self.semantic_nodes.get(concept_id)
            still_used = any(
                concept_id == concept.strip().lower().replace(" ", "_")
                for record in self.episodic_memories.values()
                for concept in record.associated_concepts
            )
            if node and node.category == "EXPERIENCE" and not still_used:
                self.semantic_nodes.pop(concept_id, None)
                self.semantic_edges = [edge for edge in self.semantic_edges if edge.source_id != concept_id and edge.target_id != concept_id]
        self.auto_save_binary()
        return len(removed)

    # ==================== Memória Semântica ====================

    def _touch_or_create_concept(
        self,
        concept_name: str,
        category: str = "GENERAL",
        description: str = "",
    ) -> ConceptNode:
        cid = concept_name.strip().lower().replace(" ", "_")
        now = datetime.now(timezone.utc).isoformat()
        if cid in self.semantic_nodes:
            node = self.semantic_nodes[cid]
            node.activation_count += 1
            node.stability = min(1.0, node.stability + 0.05)
            node.last_accessed_at = now
            return node

        node = ConceptNode(
            concept_id=cid,
            label=concept_name.strip(),
            category=category,
            description=description or f"Conceito aprendido autonomamente: {concept_name}",
            activation_count=1,
            stability=0.4,
            first_learned_at=now,
            last_accessed_at=now,
        )
        self.semantic_nodes[cid] = node
        return node

    def add_semantic_relation(
        self,
        source_concept: str,
        target_concept: str,
        relation: str = "ASSOCIATED_WITH",
        strength: float = 0.5,
    ) -> ConceptEdge:
        """Conecta dois conceitos na rede semântica com reforço sináptico."""
        s_node = self._touch_or_create_concept(source_concept)
        t_node = self._touch_or_create_concept(target_concept)

        for edge in self.semantic_edges:
            if edge.source_id == s_node.concept_id and edge.target_id == t_node.concept_id and edge.relation == relation:
                edge.strength = min(1.0, edge.strength + 0.1)
                return edge

        edge = ConceptEdge(
            source_id=s_node.concept_id,
            target_id=t_node.concept_id,
            relation=relation,
            strength=strength,
        )
        self.semantic_edges.append(edge)
        return edge

    # ==================== Ciclo de Consolidação e Sono Cognitivo ====================

    def run_sleep_and_consolidation_cycle(self) -> Dict[str, Any]:
        """
        Executa um ciclo de sono cognitivo e consolidação de memórias:
        1. Processa memórias episódicas não consolidadas (< 0.95).
        2. Eleva o nível de consolidação sináptica (LTP de longo prazo).
        3. Identifica associações conceituais cruzadas entre o que viu, leu e conversou.
        4. Fortalece o grafo semântico.
        5. Gera um relato de reflexão onírica da Veronica.
        """
        unconsolidated = [m for m in self.episodic_memories.values() if m.consolidation_level < 0.95]
        consolidated_count = 0
        strengthened_edges = 0
        new_cross_associations: List[str] = []

        now = datetime.now(timezone.utc).isoformat()

        # Eleva a consolidação das memórias episódicas recentes
        for mem in unconsolidated:
            mem.consolidation_level = min(1.0, mem.consolidation_level + 0.45)
            consolidated_count += 1

            # Cria conexões entre conceitos que ocorreram no mesmo episódio
            concepts = mem.associated_concepts
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    edge = self.add_semantic_relation(
                        source_concept=concepts[i],
                        target_concept=concepts[j],
                        relation="CO_OCCURRED_IN_EXPERIENCE",
                        strength=0.6,
                    )
                    strengthened_edges += 1
                    new_cross_associations.append(f"{concepts[i]} <-> {concepts[j]}")

        # Limpa itens antigos da memória de trabalho agora que foram consolidados
        self.clear_working_memory()

        # Elabora o relato de reflexão/sono
        experience_summary_sample = "; ".join(
            [m.experience_summary[:60] for m in unconsolidated[:3]]
        ) or "experiências e diálogos operacionais recentes"

        reflection_text = (
            f"Durante meu ciclo de reflexão e sono cognitivo, consolidei {consolidated_count} memórias episódicas. "
            f"Processei vivências como: '{experience_summary_sample}'. "
            f"Fortaleci {strengthened_edges} conexões sinápticas no meu grafo de conceitos, "
            f"integrando o que vi e li à minha sabedoria de longo prazo."
        )

        cycle_result = {
            "timestamp": now,
            "memories_consolidated": consolidated_count,
            "synaptic_edges_strengthened": strengthened_edges,
            "total_episodic_memories": len(self.episodic_memories),
            "total_semantic_concepts": len(self.semantic_nodes),
            "total_semantic_edges": len(self.semantic_edges),
            "cross_associations": new_cross_associations[:8],
            "reflection_dream_report": reflection_text,
        }
        self.consolidation_history.append(cycle_result)
        self.auto_save_binary()
        return cycle_result

    # ==================== Métodos de Salvamento e Carga Binária ====================

    def auto_save_binary(self) -> None:
        """Salva automaticamente no arquivo binário próprio se storage_dir estiver configurado."""
        if self.binary_filepath:
            try:
                self.save_binary_file(self.binary_filepath)
            except Exception:
                pass

    def save_binary_file(self, filepath: Optional[Union[str, Path]] = None) -> int:
        """Salva todas as memórias em arquivo binário próprio (VRNC_MEM v2.0)."""
        target = Path(filepath) if filepath else self.binary_filepath
        if not target:
            raise ValueError("Nenhum caminho de arquivo binário especificado para salvamento.")
        return self.binary_store.save_binary_file(target, self)

    def load_binary_file(self, filepath: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Carrega e restaura todas as memórias a partir de arquivo binário próprio."""
        target = Path(filepath) if filepath else self.binary_filepath
        if not target:
            raise ValueError("Nenhum caminho de arquivo binário especificado para carregamento.")
        return self.binary_store.load_binary_file(target, self)

    def get_binary_stats(self) -> Dict[str, Any]:
        """Inspeciona e retorna estatísticas em tempo real do arquivo de memória binária."""
        if not self.binary_filepath or not self.binary_filepath.exists():
            return {
                "file_path": str(self.binary_filepath) if self.binary_filepath else None,
                "file_exists": False,
                "file_size_bytes": 0,
                "magic_header": "VRNC_MEM\x02\x00",
                "magic_header_hex": BinaryMachineMemoryStore.MAGIC_HEADER.hex(),
                "format_version": "2.0",
                "is_valid_binary": False,
                "last_saved_at": None,
                "working_memory_count": len(self.working_memory),
                "episodic_count": len(self.episodic_memories),
                "semantic_nodes_count": len(self.semantic_nodes),
                "semantic_edges_count": len(self.semantic_edges),
                "consolidation_cycles_count": len(self.consolidation_history),
                "total_records": len(self.working_memory) + len(self.episodic_memories) + len(self.semantic_nodes),
            }
        return self.binary_store.inspect_binary_file(self.binary_filepath)

    def export_full_memory_map(self) -> Dict[str, Any]:
        """Exporta o mapa completo de memórias para a interface e observatório."""
        recent_episodes = sorted(
            self.episodic_memories.values(),
            key=lambda x: x.timestamp,
            reverse=True,
        )[:25]

        nodes = [asdict(n) for n in self.semantic_nodes.values()]
        edges = [asdict(e) for e in self.semantic_edges]

        return {
            "working_memory_active_count": len(self.working_memory),
            "working_memory": [asdict(w) for w in self.working_memory],
            "total_episodic_memories": len(self.episodic_memories),
            "recent_episodic_timeline": [asdict(e) for e in recent_episodes],
            "total_semantic_concepts": len(self.semantic_nodes),
            "semantic_nodes": nodes,
            "semantic_edges": edges,
            "total_consolidation_cycles": len(self.consolidation_history),
            "last_consolidation": self.consolidation_history[-1] if self.consolidation_history else None,
        }

