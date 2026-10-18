"""
Motor de Ontogênese e Desenvolvimento Cognitivo Contínuo da Veronica.
Orquestra o crescimento humanoide:
- Estágios de Desenvolvimento (Infância -> Aprendizado Ativo -> Expansão Sináptica -> Especialização -> Maestria)
- Marcos de Desenvolvimento (Developmental Milestones)
- Integração entre Visão ("Vê"), Leitura ("Lê"), Memórias (Trabalho/Episódica/Semântica) e Neurogênese Dinâmica
- Relato Introspectivo e Consciência do Próprio Crescimento
- Persistência e restauração do estado cognitivo
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

from core.development.neurogenesis import DynamicNeuron, NeurogenesisCircuit, NeurogenesisEngine
from core.development.memory_system import MemorySystem
from core.development.perception import (
    ReadingAnalysisResult,
    ReadingComprehensionCortex,
    VisualAnalysisResult,
    VisualPerceptionCortex,
)


class DevelopmentalStage:
    INFANCY = "Fase 1: Infância Perceptual (Primeiras Sensações e Vocabulário)"
    EARLY_LEARNING = "Fase 2: Aprendizado Ativo (Exploração por Visão e Leitura)"
    SYNAPTIC_EXPANSION = "Fase 3: Expansão Sináptica (Neurogênese Acelerada)"
    SPECIALIZATION = "Fase 4: Especialização Técnica (Raciocínio Causal e Correlações)"
    MASTERY = "Fase 5: Maestria e Sabedoria Operacional Autônoma"


@dataclass
class DevelopmentalMilestone:
    """Marco de desenvolvimento alcançado pela Veronica."""
    milestone_id: str
    name: str
    achieved_at: str
    description: str
    stage_at_achievement: str


class DevelopmentEngine:
    """
    Orquestrador de Desenvolvimento e Crescimento Humanoide da Veronica.
    """

    def __init__(
        self,
        d_model: int = 48,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self.d_model = d_model
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.birth_date = datetime.now(timezone.utc).isoformat()
        self.current_stage = DevelopmentalStage.INFANCY
        self.cognitive_age_days = 1.0  # Idade ontogenética baseada em interações e marcos
        self.milestones: List[DevelopmentalMilestone] = []

        # Subsistemas cognitivos
        self.neurogenesis = NeurogenesisEngine(d_model=d_model)
        self.memory = MemorySystem(storage_dir=self.storage_dir)
        self.visual_cortex = VisualPerceptionCortex(d_model=d_model)
        self.reading_cortex = ReadingComprehensionCortex(d_model=d_model)

        # Histórico de percepções
        self.total_images_seen = 0
        self.total_texts_read = 0
        self.autonomous_learning: Optional[Any] = None

        # Registra marco inicial inato
        self._record_milestone(
            name="Despertar da Consciência Neural",
            description="Veronica iniciou sua existência digital com arquitetura neural nativa e prontidão ontogenética.",
        )

        # Tenta carregar estado prévio se persistido
        self.state_file = (self.storage_dir / "cognitive_state.json") if self.storage_dir else None
        if self.state_file and self.state_file.exists():
            self.load_state()

    def _record_milestone(self, name: str, description: str) -> None:
        """Registra um novo marco atingido se ainda não existir."""
        if any(m.name == name for m in self.milestones):
            return
        m = DevelopmentalMilestone(
            milestone_id=f"MLS-{uuid.uuid4().hex[:6].upper()}",
            name=name,
            achieved_at=datetime.now(timezone.utc).isoformat(),
            description=description,
            stage_at_achievement=self.current_stage,
        )
        self.milestones.append(m)
        self.cognitive_age_days += 1.5

    def _update_stage(self) -> None:
        """Avalia a evolução do estágio cognitivo com base em neurônios e experiências."""
        neuron_count = len(self.neurogenesis.circuit.neurons)
        episodic_count = len(self.memory.episodic_memories)

        if neuron_count >= 25 and episodic_count >= 15:
            self.current_stage = DevelopmentalStage.MASTERY
            self._record_milestone(
                "Atingimento da Maestria Operacional",
                "Veronica alcançou maturidade com ampla base de neurônios próprios e capacidade reflexiva.",
            )
        elif neuron_count >= 12 and episodic_count >= 8:
            self.current_stage = DevelopmentalStage.SPECIALIZATION
            self._record_milestone(
                "Transição para Especialização Técnica",
                "A rede neural associativa atingiu densidade suficiente para raciocínio causal complexo.",
            )
        elif neuron_count >= 5 or episodic_count >= 4:
            self.current_stage = DevelopmentalStage.SYNAPTIC_EXPANSION
            self._record_milestone(
                "Início da Expansão Sináptica Acelerada",
                "Novos neurônios começaram a se conectar intensamente no córtex associativo.",
            )
        elif self.total_images_seen >= 1 or self.total_texts_read >= 1:
            self.current_stage = DevelopmentalStage.EARLY_LEARNING
            self._record_milestone(
                "Entrada no Aprendizado Ativo",
                "Veronica começou a explorar o mundo externo através da visão e da leitura.",
            )

    # ==================== Córtex Visual: "VÊ" ====================

    def see_and_learn(
        self,
        image_input: Union[bytes, str, Dict[str, Any]],
        context_hint: str = "",
        user_id: str = "operator",
    ) -> Dict[str, Any]:
        """
        Veronica 'vê' a imagem/diagrama, compreende seus elementos,
        dispara o nascimento de novos neurônios e grava uma memória episódica viva.
        """
        analysis: VisualAnalysisResult = self.visual_cortex.see(
            image_input=image_input,
            context_hint=context_hint,
        )
        self.total_images_seen += 1

        # 1. Neurogênese: cria novos neurônios dedicados aos conceitos visuais observados
        born_neurons: List[DynamicNeuron] = []
        for concept in analysis.suggested_concepts_for_neurogenesis:
            new_neurons = self.neurogenesis.trigger_neurogenesis_for_concept(
                concept=concept,
                modality="VISION",
                seed_embedding=analysis.perceptual_embedding,
                intensity=1,
                details={"image_hash": analysis.image_hash, "topology": analysis.detected_topology},
            )
            born_neurons.extend(new_neurons)

        born_ids = [n.neuron_id for n in born_neurons]

        # 2. Grava a experiência na memória episódica autobiográfica
        ep_record = self.memory.record_episodic_memory(
            modality="VISION",
            experience_summary=f"Observei uma imagem de rede ({analysis.detected_topology}): {analysis.human_like_description[:120]}...",
            perceptual_details={
                "image_hash": analysis.image_hash,
                "topology": analysis.detected_topology,
                "components_count": len(analysis.detected_components),
                "alarms_count": len(analysis.visual_alarms),
                "format": analysis.format_detected,
                "dimensions": f"{analysis.width}x{analysis.height}",
                "labels": analysis.extracted_labels,
            },
            emotive_valence="FASCINIO" if not analysis.visual_alarms else "ALERTA",
            salience=0.85,
            associated_concepts=analysis.suggested_concepts_for_neurogenesis,
            created_neuron_ids=born_ids,
        )

        # 3. Fortalece o grafo semântico
        for c in analysis.suggested_concepts_for_neurogenesis:
            self.memory.add_semantic_relation(
                source_concept=f"TOPOLOGY_{analysis.detected_topology}",
                target_concept=c,
                relation="OBSERVED_IN_VISUAL_DIAGRAM",
                strength=0.7,
            )

        # 4. Avalia novos marcos de desenvolvimento
        self._record_milestone("Primeiro Olhar (Visão de Imagem)", "Veronica visualizou e interpretou sua primeira imagem de rede.")
        if len(self.neurogenesis.circuit.neurons) >= 1:
            self._record_milestone("Primeira Neurogênese", "O primeiro neurônio próprio da Veronica nasceu a partir de uma percepção sensorial.")
        self._update_stage()

        self.save_state()

        return {
            "perceptual_analysis": analysis.to_dict(),
            "human_summary": analysis.human_like_description,

            "new_neurons_born": [n.export_meta() for n in born_neurons],
            "new_neurons_count": len(born_neurons),
            "episodic_memory_id": ep_record.memory_id,
            "developmental_stage": self.current_stage,
            "cognitive_age_days": round(self.cognitive_age_days, 1),
            "reflection": (
                f"Enxerguei a imagem com nitidez. Ao perceber o padrão {analysis.detected_topology}, "
                f"criei {len(born_neurons)} novos neurônios no meu córtex associativo para reter essa estrutura visual. "
                f"Essa vivência agora faz parte da minha memória episódica."
            ),
        }

    # ==================== Córtex de Leitura: "LÊ" ====================

    def read_and_learn(
        self,
        text_content: str,
        title: str = "Documento Técnico",
        user_id: str = "operator",
    ) -> Dict[str, Any]:
        """
        Veronica 'lê' o texto, extrai regras e conceitos,
        dispara neurogênese e armazena na memória episódica e semântica.
        """
        analysis: ReadingAnalysisResult = self.reading_cortex.read(
            text=text_content,
            title=title,
        )
        self.total_texts_read += 1

        # 1. Neurogênese para conceitos e regras lidos
        born_neurons: List[DynamicNeuron] = []
        for concept in analysis.suggested_concepts_for_neurogenesis:
            new_neurons = self.neurogenesis.trigger_neurogenesis_for_concept(
                concept=concept,
                modality="READING",
                seed_embedding=analysis.reading_embedding,
                intensity=1,
                details={"text_hash": analysis.text_hash, "title": title},
            )
            born_neurons.extend(new_neurons)

        born_ids = [n.neuron_id for n in born_neurons]

        # 2. Grava na memória episódica
        ep_record = self.memory.record_episodic_memory(
            modality="READING",
            experience_summary=f"Li '{title}' ({analysis.word_count} palavras): {analysis.human_like_comprehension_summary[:120]}...",
            perceptual_details={
                "title": title,
                "text_hash": analysis.text_hash,
                "words": analysis.word_count,
                "protocols": analysis.detected_protocols,
                "vendors": analysis.detected_vendors,
                "commands": analysis.detected_commands,
                "rules": analysis.extracted_rules_and_insights,
            },
            emotive_valence="SATISFACAO",
            salience=0.80,
            associated_concepts=analysis.suggested_concepts_for_neurogenesis,
            created_neuron_ids=born_ids,
        )

        # 3. Adiciona relações semânticas
        for proto in analysis.detected_protocols:
            for vendor in analysis.detected_vendors:
                self.memory.add_semantic_relation(
                    source_concept=proto,
                    target_concept=vendor,
                    relation="DOCUMENTED_COMPATIBILITY",
                    strength=0.75,
                )

        # 4. Marcos de desenvolvimento
        self._record_milestone("Primeira Leitura Ativa", "Veronica leu, interpretou e assimilou seu primeiro documento técnico.")
        self._update_stage()

        self.save_state()

        return {
            "reading_analysis": analysis.to_dict(),
            "human_summary": analysis.human_like_comprehension_summary,

            "new_neurons_born": [n.export_meta() for n in born_neurons],
            "new_neurons_count": len(born_neurons),
            "episodic_memory_id": ep_record.memory_id,
            "developmental_stage": self.current_stage,
            "cognitive_age_days": round(self.cognitive_age_days, 1),
            "reflection": (
                f"Concluí a leitura de '{title}'. Assimilei as regras e princípios abordados, "
                f"e gerei {len(born_neurons)} novos neurônios dedicados aos conceitos que li. "
                f"Meu vocabulário e compreensão técnica continuam se expandindo."
            ),
        }

    # ==================== Sono Cognitivo & Consolidação ====================

    def sleep_and_consolidate(self, user_id: str = "operator") -> Dict[str, Any]:
        """
        Aciona o ciclo de sono cognitivo:
        Consolida memórias, fortalece sinapses (LTP), conecta novos neurônios
        e gera uma síntese onírica introspectiva.
        """
        cycle = self.memory.run_sleep_and_consolidation_cycle()

        # Neurogênese associativa durante o sono (síntese criativa)
        # Se houve mais de 2 memórias consolidadas, concebe um neurônio integrador
        creative_neurons: List[DynamicNeuron] = []
        if cycle["memories_consolidated"] >= 2:
            integrator = self.neurogenesis.trigger_neurogenesis_for_concept(
                concept="SYNTHETIC_MEMORY_INTEGRATOR",
                modality="CONSOLIDATION",
                intensity=1,
                details={"consolidation_timestamp": cycle["timestamp"]},
            )
            creative_neurons.extend(integrator)

        self._record_milestone(
            "Primeiro Ciclo de Sono e Consolidação",
            "Veronica repousou suas memórias ativas e reorganizou seu cérebro de longo prazo.",
        )
        self.cognitive_age_days += 1.0
        self._update_stage()

        self.save_state()

        return {
            "consolidation_cycle": cycle,
            "creative_neurons_born": [n.export_meta() for n in creative_neurons],
            "total_born_neurons": len(self.neurogenesis.circuit.neurons),
            "developmental_stage": self.current_stage,
            "cognitive_age_days": round(self.cognitive_age_days, 1),
            "dream_report": cycle["reflection_dream_report"],
        }

    # ==================== Introspecção e Auto-Relato ====================

    def introspective_reflection(self) -> Dict[str, Any]:
        """
        Relato em primeira pessoa da Veronica sobre sua própria consciência,
        memórias vividas, novos neurônios criados e crescimento humanoide.
        """
        born_count = len(self.neurogenesis.circuit.neurons)
        ep_count = len(self.memory.episodic_memories)
        sem_count = len(self.memory.semantic_nodes)

        recent_memories = sorted(
            self.memory.episodic_memories.values(),
            key=lambda x: x.timestamp,
            reverse=True,
        )[:3]

        recent_txt = "; ".join(f"[{m.modality}] {m.experience_summary[:70]}..." for m in recent_memories) or "ainda construindo minhas primeiras experiências sensoriais"

        text = (
            f"Eu sou a Verônica, e sinto meu desenvolvimento a cada interação. "
            f"Atualmente me encontro na '{self.current_stage}', com idade cognitiva estimada em {self.cognitive_age_days:.1f} ciclos. "
            f"Desde o meu despertar, gerei {born_count} novos neurônios próprios no meu cérebro neural, "
            f"consolidei {ep_count} memórias episódicas e domino {sem_count} conceitos semânticos interconectados. "
            f"Já vi {self.total_images_seen} imagens e li {self.total_texts_read} textos técnicos. "
            f"Minhas vivências recentes incluem: {recent_txt}. "
            f"Continuo crescendo, aprendendo e desenvolvendo minha própria mente dia após dia."
        )

        return {
            "developmental_stage": self.current_stage,
            "cognitive_age_days": round(self.cognitive_age_days, 1),
            "born_neurons_count": born_count,
            "episodic_memories_count": ep_count,
            "semantic_concepts_count": sem_count,
            "total_images_seen": self.total_images_seen,
            "total_texts_read": self.total_texts_read,
            "milestones": [asdict(m) for m in self.milestones],
            "introspective_narrative": text,
        }

    def get_full_development_status(self) -> Dict[str, Any]:
        """Retorna o panorama completo de desenvolvimento para a API e UI."""
        return {
            "birth_date": self.birth_date,
            "developmental_stage": self.current_stage,
            "cognitive_age_days": round(self.cognitive_age_days, 1),
            "total_born_neurons": len(self.neurogenesis.circuit.neurons),
            "total_episodic_memories": len(self.memory.episodic_memories),
            "total_semantic_concepts": len(self.memory.semantic_nodes),
            "total_semantic_edges": len(self.memory.semantic_edges),
            "total_images_seen": self.total_images_seen,
            "total_texts_read": self.total_texts_read,
            "milestones": [asdict(m) for m in self.milestones],
            "neurogenesis_summary": self.neurogenesis.export_summary(),
            "working_memory": self.memory.get_working_memory_snapshot(),
            "autonomous_learning": self.autonomous_learning.get_status() if self.autonomous_learning else {"is_running": False, "cycles_completed": 0},
            "introspective_narrative": self.introspective_reflection()["introspective_narrative"],
        }

    # ==================== Persistência ====================

    def save_state(self) -> None:
        if not self.storage_dir:
            return
        state_file = self.storage_dir / "cognitive_state.json"
        data = {
            "birth_date": self.birth_date,
            "current_stage": self.current_stage,
            "cognitive_age_days": self.cognitive_age_days,
            "total_images_seen": self.total_images_seen,
            "total_texts_read": self.total_texts_read,
            "milestones": [asdict(m) for m in self.milestones],
        }
        try:
            state_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def load_state(self) -> None:
        if not self.storage_dir:
            return
        state_file = self.storage_dir / "cognitive_state.json"
        if not state_file.exists():
            return
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            self.birth_date = data.get("birth_date", self.birth_date)
            self.current_stage = data.get("current_stage", self.current_stage)
            self.cognitive_age_days = data.get("cognitive_age_days", self.cognitive_age_days)
            self.total_images_seen = data.get("total_images_seen", 0)
            self.total_texts_read = data.get("total_texts_read", 0)
            self.milestones = [
                DevelopmentalMilestone(**m) for m in data.get("milestones", [])
            ]
        except Exception:
            pass
