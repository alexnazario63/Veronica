"""
Mecanismo de Neurogênese Artificial e Plasticidade Estrutural da Veronica.
Implementa:
- DynamicNeuron: Neurônio artificial vivo parametrizado com ciclo de vida, mielinização e LTP hebbiano.
- NeurogenesisCircuit: Circuito neural expansível que integra neurônios nascidos ao espaço latente da rede.
- NeurogenesisEngine: Motor ontogenético de nascimento, conexão sináptica e consolidação de neurônios.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple
import uuid
import numpy as np

from core.neural.layers import Module, Neuron, Parameter
from core.neural.tensor import Tensor


@dataclass
class NeuronOntogenyMeta:
    """Metadados de nascimento e desenvolvimento de um neurônio artificial."""
    neuron_id: str
    birth_timestamp: str
    trigger_type: str  # "VISION", "READING", "CONSOLIDATION", "COGNITIVE_EXPERIENCE"
    trigger_concept: str
    developmental_stage: str  # "NEONATAL", "SYNAPTICALLY_INTEGRATED", "MATURE", "SPECIALIZED"
    activation_fn: str
    firing_count: int
    synaptic_strength: float
    myelination_score: float  # 0.1 (imaturo) a 1.0 (totalmente mielinizado/veloz)
    coords_3d: Tuple[float, float, float]
    layer_affinity: str = "associative_cortex"


class DynamicNeuron(Neuron):
    """
    Neurônio artificial vivo criado pela própria Veronica durante seu desenvolvimento.
    Herda do Neurônio parametrizado nativo (z = x @ w + b, a = act(z)),
    adicionando projeção sináptica de saída (outgoing projection),
    reforço hebbiano (LTP) e amadurecimento ontogenético (mielinização).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        trigger_concept: str,
        trigger_type: str = "COGNITIVE_EXPERIENCE",
        activation: str = "gelu",
        seed_vector: Optional[np.ndarray] = None,
        coords_3d: Optional[Tuple[float, float, float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(in_features=in_features, activation=activation, bias=True, seed=seed)
        self.out_features = out_features
        self.trigger_concept = trigger_concept
        self.trigger_type = trigger_type
        self.neuron_id = f"NEURON-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:6].upper()}"
        self.birth_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Estágios ontogenéticos do neurônio
        self.developmental_stage = "NEONATAL"
        self.firing_count = 0
        self.synaptic_strength = 1.0
        self.myelination_score = 0.15  # Começa jovem/imaturo
        self.last_activated_at: Optional[str] = None
        self.co_activated_concepts: List[str] = []

        # Se houver vetor semente (ex: embedding do conceito visto ou lido),
        # alinha o peso de entrada com afinidade a esse padrão (inicialização hebbiana)
        rng = np.random.default_rng(seed)
        if seed_vector is not None:
            norm_seed = seed_vector.flatten()[:in_features]
            if len(norm_seed) < in_features:
                norm_seed = np.pad(norm_seed, (0, in_features - len(norm_seed)))
            norm_val = np.linalg.norm(norm_seed)
            if norm_val > 1e-9:
                norm_seed = norm_seed / norm_val
            # Combina semente com pequena perturbação estocástica para diferenciação
            bound = 0.15 / math.sqrt(max(in_features, 1))
            noise = rng.uniform(-bound, bound, size=(in_features,))
            combined = (norm_seed * 0.85 + noise * 0.15).reshape(in_features, 1)
            self.weight.data[...] = combined

        # Projeção sináptica de saída (conecta a ativação desse neurônio de volta ao espaço latente d_model)
        # Inicializa com amplitude conservadora para evitar choque nas representações prévias
        init_out_bound = 0.05 / math.sqrt(max(out_features, 1))
        out_weights = rng.uniform(-init_out_bound, init_out_bound, size=(1, out_features))
        self.outgoing_weights = Parameter(out_weights, name=f"{self.neuron_id}.outgoing")

        # Coordenadas 3D para renderização no Neural Observatory (Three.js)
        if coords_3d is not None:
            self.coords_3d = coords_3d
        else:
            # Distribui em uma coroa esférica que orbita as camadas do transformer
            phi = rng.uniform(0, 2 * math.pi)
            costheta = rng.uniform(-1, 1)
            u = rng.uniform(0, 1)
            theta = math.acos(costheta)
            r = 35.0 + 15.0 * (u ** (1 / 3))  # Raio de 35 a 50 unidades
            x_pos = r * math.sin(theta) * math.cos(phi)
            y_pos = r * math.sin(theta) * math.sin(phi)
            z_pos = r * math.cos(theta)
            self.coords_3d = (float(round(x_pos, 2)), float(round(y_pos, 2)), float(round(z_pos, 2)))

    def project_outgoing(self, act: Tensor) -> Tensor:
        """
        Calcula a projeção sináptica do impulso deste neurônio de volta para a rede:
        proj = (act @ outgoing_weights) * (synaptic_strength * myelination_score)
        """
        scale = float(self.synaptic_strength * self.myelination_score)
        return (act @ self.outgoing_weights) * scale

    def fire_and_adapt(self, activation_val: float, context_concept: Optional[str] = None) -> None:
        """
        Registra disparo do neurônio e aplica potenciação de longo prazo (LTP) hebbiana.
        'Neurons that fire together wire together':
        Conforme dispara mais vezes e de forma relevante, aumenta sua mielinização e força sináptica.
        """
        self.firing_count += 1
        self.last_activated_at = datetime.now(timezone.utc).isoformat()
        if context_concept and context_concept not in self.co_activated_concepts:
            self.co_activated_concepts.append(context_concept)
            if len(self.co_activated_concepts) > 8:
                self.co_activated_concepts.pop(0)

        # Regra de Potenciação de Longo Prazo (LTP Hebbiano)
        # Mais ativação -> maior reforço sináptico assintótico
        ltp_gain = 0.04 * math.tanh(abs(activation_val))
        self.synaptic_strength = min(3.5, self.synaptic_strength + ltp_gain)

        # Mielinização ontogenética (amadurecimento biológico do axônio)
        # De imaturo (0.15) até totalmente integrado (1.0)
        self.myelination_score = min(1.0, self.myelination_score + 0.02)

        # Atualização do estágio de desenvolvimento do neurônio
        if self.firing_count >= 30 and self.myelination_score >= 0.85:
            self.developmental_stage = "SPECIALIZED"
        elif self.firing_count >= 12 and self.myelination_score >= 0.55:
            self.developmental_stage = "MATURE"
        elif self.firing_count >= 3:
            self.developmental_stage = "SYNAPTICALLY_INTEGRATED"

    def export_meta(self) -> Dict[str, Any]:
        """Exporta metadados completos para o observatório neural e monitor de crescimento."""
        w_in_flat = self.weight.data.flatten()
        w_out_flat = self.outgoing_weights.data.flatten()
        return {
            "neuron_id": self.neuron_id,
            "birth_timestamp": self.birth_timestamp,
            "trigger_type": self.trigger_type,
            "trigger_concept": self.trigger_concept,
            "developmental_stage": self.developmental_stage,
            "activation_fn": self.activation_name,
            "firing_count": self.firing_count,
            "synaptic_strength": round(float(self.synaptic_strength), 4),
            "myelination_score": round(float(self.myelination_score), 4),
            "coords_3d": self.coords_3d,
            "last_activated_at": self.last_activated_at,
            "co_activated_concepts": list(self.co_activated_concepts),
            "weight_in_norm": round(float(np.linalg.norm(w_in_flat)), 4),
            "weight_out_norm": round(float(np.linalg.norm(w_out_flat)), 4),
        }


class NeurogenesisCircuit(Module):
    """
    Circuito neural expansível que gerencia o crescimento de novos neurônios da Veronica.
    Conecta-se ao modelo principal (VeronicaTransformer) como uma rede associativa
    de expansão contínua.
    """

    def __init__(self, d_model: int = 48, seed: int = 42) -> None:
        super().__init__()
        self.d_model = d_model
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.neurons: List[DynamicNeuron] = []
        self._neuron_lookup: Dict[str, DynamicNeuron] = {}

    def birth_neuron(
        self,
        trigger_concept: str,
        trigger_type: str = "COGNITIVE_EXPERIENCE",
        seed_vector: Optional[np.ndarray] = None,
        activation_fn: str = "gelu",
        coords_3d: Optional[Tuple[float, float, float]] = None,
    ) -> DynamicNeuron:
        """
        Realiza a neurogênese: concebe e integra um novo neurônio próprio à Veronica.
        """
        neuron = DynamicNeuron(
            in_features=self.d_model,
            out_features=self.d_model,
            trigger_concept=trigger_concept,
            trigger_type=trigger_type,
            activation=activation_fn,
            seed_vector=seed_vector,
            coords_3d=coords_3d,
            seed=int(self.rng.integers(0, 1000000)),
        )
        self.neurons.append(neuron)
        self._neuron_lookup[neuron.neuron_id] = neuron
        return neuron

    def forward_integrated(
        self,
        x: Tensor,
        context_concept: Optional[str] = None,
        activation_threshold: float = 0.05,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Executa a passagem de ativação pelos neurônios nascidos.
        Retorna a modulação residual associativa combinada e telemetria de disparos.
        """
        if not self.neurons:
            return x, {"firing_neurons": [], "active_count": 0, "total_neurons": 0}

        batch_size = x.shape[0] if x.ndim >= 2 else 1
        seq_len = x.shape[1] if x.ndim == 3 else (x.shape[0] if x.ndim == 2 else 1)
        
        # Flatten para avaliação de ativação posicional ou média sobre tokens
        # x pode ter shape (B, T, D) ou (T, D) ou (1, D)
        accum_projection = np.zeros(x.shape, dtype=np.float64)
        firing_telemetry: List[Dict[str, Any]] = []

        for neuron in self.neurons:
            act = neuron(x)  # Shape (..., 1)
            act_data = act.data
            max_act = float(np.max(np.abs(act_data)))
            mean_act = float(np.mean(act_data))

            if max_act > activation_threshold:
                # O neurônio disparou para esta entrada!
                neuron.fire_and_adapt(activation_val=max_act, context_concept=context_concept)
                proj = neuron.project_outgoing(act)
                accum_projection += proj.data

                firing_telemetry.append(
                    {
                        "neuron_id": neuron.neuron_id,
                        "concept": neuron.trigger_concept,
                        "stage": neuron.developmental_stage,
                        "max_activation": round(max_act, 4),
                        "mean_activation": round(mean_act, 4),
                        "synaptic_strength": round(float(neuron.synaptic_strength), 4),
                        "myelination": round(float(neuron.myelination_score), 4),
                    }
                )

        # Modulação residual suave com fator de acoplamento conservador
        coupling_factor = 0.15
        modulated_x = x + Tensor(accum_projection * coupling_factor, requires_grad=x.requires_grad)

        stats = {
            "firing_neurons": firing_telemetry,
            "active_count": len(firing_telemetry),
            "total_neurons": len(self.neurons),
            "max_firing_activation": max((f["max_activation"] for f in firing_telemetry), default=0.0),
        }
        return modulated_x, stats

    def get_neuron(self, neuron_id: str) -> Optional[DynamicNeuron]:
        return self._neuron_lookup.get(neuron_id)

    def export_observatory_graph(self) -> Dict[str, Any]:
        """Exporta os neurônios nascidos para o Neural Observatory 3D."""
        nodes = [n.export_meta() for n in self.neurons]
        # Conecta sinapticamente neurônios que compartilham conceitos co-ativados
        synaptic_edges: List[Dict[str, Any]] = []
        for i in range(len(self.neurons)):
            for j in range(i + 1, len(self.neurons)):
                n1 = self.neurons[i]
                n2 = self.neurons[j]
                common = set(n1.co_activated_concepts).intersection(set(n2.co_activated_concepts))
                if common or (n1.trigger_concept.lower() in n2.trigger_concept.lower() and n1.trigger_concept != n2.trigger_concept):
                    synaptic_edges.append(
                        {
                            "source_id": n1.neuron_id,
                            "target_id": n2.neuron_id,
                            "synapse_type": "ASSOCIATIVE_LONG_TERM",
                            "strength": round(float(min(n1.synaptic_strength, n2.synaptic_strength)), 3),
                            "shared_concepts": list(common) or [n1.trigger_concept],
                        }
                    )
        return {
            "total_born_neurons": len(self.neurons),
            "nodes": nodes,
            "synaptic_edges": synaptic_edges,
        }


class NeurogenesisEngine:
    """
    Motor central que orquestra a neurogênese da Veronica:
    - Avalia necessidade de criação de neurônios perante novos estímulos
    - Aloca e conecta neurônios
    - Mantém a plasticidade e conservação da rede
    """

    def __init__(self, circuit: Optional[NeurogenesisCircuit] = None, d_model: int = 48) -> None:
        self.circuit = circuit if circuit is not None else NeurogenesisCircuit(d_model=d_model)
        self.novelty_threshold: float = 0.45
        self.birth_history: List[Dict[str, Any]] = []

    def trigger_neurogenesis_for_concept(
        self,
        concept: str,
        modality: str,
        seed_embedding: Optional[np.ndarray] = None,
        intensity: int = 1,
        details: Optional[Dict[str, Any]] = None,
    ) -> List[DynamicNeuron]:
        """
        Cria um ou mais novos neurônios dedicados ao novo conceito ou experiência aprendida.
        """
        born: List[DynamicNeuron] = []
        for i in range(max(1, intensity)):
            sub_concept = f"{concept} (sub-{i+1})" if intensity > 1 else concept
            neuron = self.circuit.birth_neuron(
                trigger_concept=sub_concept,
                trigger_type=modality,
                seed_vector=seed_embedding,
                activation_fn="gelu" if i % 2 == 0 else "tanh",
            )
            born.append(neuron)
            log_entry = {
                "neuron_id": neuron.neuron_id,
                "timestamp": neuron.birth_timestamp,
                "modality": modality,
                "concept": sub_concept,
                "coords_3d": neuron.coords_3d,
                "details": details or {},
            }
            self.birth_history.append(log_entry)
        return born

    def export_summary(self) -> Dict[str, Any]:
        """Retorna resumo estatístico da neurogênese para a interface e APIs."""
        neurons = self.circuit.neurons
        if not neurons:
            return {
                "total_neurons_created": 0,
                "average_synaptic_strength": 1.0,
                "average_myelination": 0.0,
                "stages_distribution": {},
                "top_active_neurons": [],
                "birth_history": [],
            }

        stages: Dict[str, int] = {}
        for n in neurons:
            stages[n.developmental_stage] = stages.get(n.developmental_stage, 0) + 1

        top_active = sorted(neurons, key=lambda x: x.firing_count, reverse=True)[:5]

        return {
            "total_neurons_created": len(neurons),
            "average_synaptic_strength": round(float(np.mean([n.synaptic_strength for n in neurons])), 4),
            "average_myelination": round(float(np.mean([n.myelination_score for n in neurons])), 4),
            "stages_distribution": stages,
            "top_active_neurons": [n.export_meta() for n in top_active],
            "birth_history": self.birth_history[-20:],
        }

