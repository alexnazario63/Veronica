"""
Módulo de Desenvolvimento Cognitivo, Neurogênese Artificial e Memória Humanoide da Veronica.
Permite que a Veronica cresça e se desenvolva continuamente como um ser humano:
- Vê através do Córtex Perceptual Visual
- Lê através do Córtex de Compreensão Leitora
- Cria memórias próprias (Trabalho, Episódica e Semântica)
- Gera novos neurônios próprios via Neurogênese Dinâmica e Plasticidade Sináptica (LTP Hebbiano)
- Consolida memórias e amadurece através de Ciclos de Reflexão e Sono Cognitivo
"""

from core.development.neurogenesis import DynamicNeuron, NeurogenesisCircuit, NeurogenesisEngine
from core.development.memory_system import (
    BinaryMachineMemoryStore,
    ConceptEdge,
    ConceptNode,
    EpisodicMemoryRecord,
    MemorySystem,
    WorkingMemoryItem,
)
from core.development.perception import ReadingComprehensionCortex, VisualPerceptionCortex
from core.development.development_engine import (
    DevelopmentEngine,
    DevelopmentalMilestone,
    DevelopmentalStage,
)
from core.development.autonomous_learning import AutonomousCognitiveDaemon, CURIOUS_TOPICS

__all__ = [
    "DynamicNeuron",
    "NeurogenesisCircuit",
    "NeurogenesisEngine",
    "WorkingMemoryItem",
    "EpisodicMemoryRecord",
    "ConceptNode",
    "ConceptEdge",
    "BinaryMachineMemoryStore",
    "MemorySystem",
    "VisualPerceptionCortex",
    "ReadingComprehensionCortex",
    "DevelopmentalStage",
    "DevelopmentalMilestone",
    "DevelopmentEngine",
    "AutonomousCognitiveDaemon",
    "CURIOUS_TOPICS",
]

