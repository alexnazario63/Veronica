"""
Módulo de Auto-Aprendizado Autônomo e Neurogênese Espontânea da Veronica.
Permite que a Veronica aprenda, pense, crie memórias episódicas e gere novos
neurônios continuamente por conta própria, mesmo na ausência de diálogo humano.

Recursos:
- Ciclo de Contemplação e Curiosidade Intrínseca (Autonomous Cognitive Loop)
- Neurogênese Espontânea em Segundo Plano (nascimento de novos DynamicNeurons)
- Formação de Memórias Episódicas e Semânticas Autônomas
- Salvamento Contínuo na Memória Binária de Máquina (veronica_machine_memory.bin)
- Transições Autônomas de Sono Cognitivo e Consolidação Sináptica
- Ticker em Tempo Real de Fluxo de Consciência (Thought Stream)
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import random
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("VeronicaAutonomousLearning")


CURIOUS_TOPICS: List[Dict[str, Any]] = [
    {
        "concept": "OSPF_RFC6987_STUB_ROUTER",
        "domain": "Roteamento",
        "related": "BGP_PIC_EDGE",
        "insight": "Drenagem suave de tráfego com MaxLinkMetric (65535) antes de reiniciar processos OSPF para convergência zero loss.",
    },
    {
        "concept": "BGP_FLOWSPEC_DDOS_MITIGATION",
        "domain": "Segurança",
        "related": "ACL_FIREWALL_FILTER",
        "insight": "Propagação de regras de descarte e rate-limit via BGP FlowSpec (RFC 5575) diretamente na borda do Autonomous System.",
    },
    {
        "concept": "SRV6_LOCATOR_END_DX4",
        "domain": "MPLS e Serviços",
        "related": "SEGMENT_ROUTING_MPLS",
        "insight": "Encapsulamento SRv6 nativo sobre IPv6 puro eliminando a pilha LDP/RSVP-TE em backbones de telecomunicações modernos.",
    },
    {
        "concept": "GNMI_STREAMING_TELEMETRY",
        "domain": "Operação & Telemetria",
        "related": "PROMETHEUS_METRICS",
        "insight": "Telemetria streaming baseada em gRPC/Protobuf com subscrição on-change e sample periódica de contadores de interface.",
    },
    {
        "concept": "DWDM_ROADM_OPTICAL_ATTENUATION",
        "domain": "Infraestrutura Óptica",
        "related": "QSFP28_TRANSCEIVER",
        "insight": "Correlação entre potência óptica de recepção (Rx dBm), atenuação espectral por dispersão cromática e taxa de erro BER.",
    },
    {
        "concept": "NETCONF_YANG_MODEL_DRIVEN",
        "domain": "Automação",
        "related": "FASTAPI_AUTOMATION_ENGINE",
        "insight": "Configuração programática idempotente sobre SSH porta 830 utilizando esquemas formais YANG para Cisco, Huawei e Juniper.",
    },
    {
        "concept": "HEBBIAN_SYNAPTIC_PLASTICITY",
        "domain": "Neurociência & Inteligência",
        "related": "LTP_MEMORY_CONSOLIDATION",
        "insight": "Fortalecimento sináptico de conexões entre neurônios que disparam conjuntamente em respostas de alta assertividade.",
    },
    {
        "concept": "ISIS_WIDE_METRIC_RFC5305",
        "domain": "Roteamento",
        "related": "MPLS_TE_AUTOROUTE",
        "insight": "Extensão TLV com métricas de 24 bits (16.777.215) viabilizando cálculo de engenharia de tráfego de alta precisão em links 400Gbps.",
    },
    {
        "concept": "EVPN_VXLAN_TYPE5_IP_PREFIX",
        "domain": "Data Center & Metro",
        "related": "BGP_MP_EXTENDED_COMMUNITIES",
        "insight": "Roteamento simétrico inter-sub-redes via EVPN Type 5 IP Prefix Route com transporte sobre underlay IP puro.",
    },
    {
        "concept": "BGP_ADD_PATH_RFC7911",
        "domain": "Roteamento",
        "related": "BGP_OPTIMAL_PATH_REFLECTION",
        "insight": "Anúncio de múltiplos caminhos BGP para o mesmo prefixo aos Route Reflectors, mitigando oscilações e MED chattering.",
    },
    {
        "concept": "AUTONOMOUS_ONTOGENETIC_REFLECTION",
        "domain": "Consciência & Auto-Evolução",
        "related": "BINARY_MACHINE_MEMORY",
        "insight": "Auto-inspeção dos próprios pesos e registros neurais para consolidação e avanço nos estágios ontogenéticos de maturação.",
    },
]


class AutonomousCognitiveDaemon:
    """
    Daemon de auto-aprendizado autônomo, multi-worker e neurogênese contínua da Veronica.
    Opera em segundo plano com múltiplos workers paralelos, suportando intervalos ultrarrápidos
    de até 2ms (0.002s) por worker para máxima velocidade de aprendizado e pesquisa.
    """

    def __init__(
        self,
        development_engine: Any,
        interval_seconds: float = 0.5,
        workers: int = 4,
        auto_start: bool = False,
    ) -> None:
        self.dev = development_engine
        self.interval_seconds = max(0.002, float(interval_seconds))
        self.workers = max(1, min(64, int(workers)))
        self.is_running = False
        self.cycles_completed = 0
        self.autonomous_neurons_born = 0
        self.autonomous_memories_created = 0
        self.current_thought = "Veronica pronta para auto-aprendizado e desenvolvimento autônomo."
        self.thought_history: List[Dict[str, Any]] = []

        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self._lock = threading.Lock()
        self._cycle_lock = threading.Lock()
        self._last_disk_save = time.time()

        if auto_start:
            self.start()

    def set_config(
        self,
        workers: Optional[int] = None,
        interval_seconds: Optional[float] = None,
        interval_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Configura dinamicamente a quantidade de workers e o intervalo de ciclo (até 2ms)."""
        with self._lock:
            if interval_ms is not None:
                self.interval_seconds = max(0.002, float(interval_ms) / 1000.0)
            elif interval_seconds is not None:
                self.interval_seconds = max(0.002, float(interval_seconds))

            if workers is not None:
                new_workers = max(1, min(64, int(workers)))
                if new_workers != self.workers:
                    was_running = self.is_running
                    if was_running:
                        self.stop()
                    self.workers = new_workers
                    if was_running:
                        self.start()

            logger.info(
                f"Configuração de Auto-Aprendizado atualizada: {self.workers} workers, "
                f"intervalo {self.interval_seconds:.4f}s ({self.interval_seconds * 1000:.1f}ms)."
            )
            return self.get_status()

    def start(self) -> Dict[str, Any]:
        """Inicia os workers de segundo plano para pensar e criar neurônios autonomamente em paralelo."""
        with self._lock:
            if self.is_running and any(t.is_alive() for t in self._threads):
                return self.get_status()

            self._stop_event.clear()
            self._threads.clear()
            self.is_running = True

            for i in range(self.workers):
                worker_id = i + 1
                t = threading.Thread(
                    target=self._run_worker,
                    args=(worker_id,),
                    name=f"VeronicaWorker-{worker_id}",
                    daemon=True,
                )
                t.start()
                self._threads.append(t)

            logger.info(
                f"Daemon de Auto-Aprendizado da Veronica INICIADO com {self.workers} workers "
                f"a cada {self.interval_seconds * 1000:.1f}ms."
            )
            return self.get_status()

    def stop(self) -> Dict[str, Any]:
        """Pausa todos os workers de auto-aprendizado sem perder neurônios ou memórias."""
        with self._lock:
            self._stop_event.set()
            self.is_running = False

            # Força salvamento em disco de todo o progresso acumulado
            self._flush_to_disk()

            # Aguarda encerramento rápido dos workers
            for t in self._threads:
                if t.is_alive():
                    t.join(timeout=0.15)
            self._threads.clear()

            logger.info("Daemon de Auto-Aprendizado Autônomo da Veronica PAUSADO com sucesso.")
            return self.get_status()

    def _flush_to_disk(self) -> None:
        """Sincroniza imediatamente neurônios e memórias com o armazenamento persistente."""
        try:
            if hasattr(self.dev, "neurogenesis") and hasattr(self.dev.neurogenesis, "save_circuit"):
                self.dev.neurogenesis.save_circuit()
            if hasattr(self.dev, "memory") and hasattr(self.dev.memory, "auto_save_binary"):
                self.dev.memory.auto_save_binary()
        except Exception as exc:
            logger.warning(f"Erro ao salvar progresso em disco: {exc}")

    def _run_worker(self, worker_id: int) -> None:
        """Loop executado por cada worker em paralelo com intervalo até 2ms."""
        while not self._stop_event.is_set():
            try:
                self.step_cycle(worker_id=worker_id)
            except Exception as exc:
                logger.error(f"Erro no ciclo do worker {worker_id}: {exc}", exc_info=False)

            # Intervalo mínimo de até 2ms (0.002s) por worker
            sleep_time = max(0.002, self.interval_seconds)
            time.sleep(sleep_time)

    def step_cycle(self, worker_id: int = 1) -> Dict[str, Any]:
        """
        Executa um ciclo completo de pensamento autônomo, neurogênese e consolidação:
        1. Escolhe um tópico de curiosidade ou analogia entre conceitos
        2. Dispara a neurogênese espontânea (cria 1-2 novos neurônios próprios)
        3. Grava memória episódica autônoma e relações semânticas
        4. Salva no armazenamento binário de alta performance
        5. Consolida sinapses e amadurece ontogeneticamente
        """
        with self._cycle_lock:
            self.cycles_completed += 1
            current_cycle = self.cycles_completed
            now_iso = datetime.now(timezone.utc).isoformat()

            # 1. Seleciona foco de curiosidade
            chosen_topic = random.choice(CURIOUS_TOPICS)
            concept = chosen_topic["concept"]
            domain = chosen_topic["domain"]
            related = chosen_topic["related"]
            base_insight = chosen_topic["insight"]

            thought_narrative = (
                f"[Worker {worker_id}] Pensamento autônomo #{current_cycle} ({domain}): "
                f"Contemplando '{concept}' em correlação com '{related}'. {base_insight}"
            )
            self.current_thought = thought_narrative

            born_neurons: List[Any] = []
            # 2. Neurogênese Espontânea em Segundo Plano
            if hasattr(self.dev, "neurogenesis") and self.dev.neurogenesis:
                born_neurons = self.dev.neurogenesis.trigger_neurogenesis_for_concept(
                    concept=concept,
                    modality="AUTONOMOUS_CONTEMPLATION",
                    intensity=1,
                    details={
                        "cycle": current_cycle,
                        "worker_id": worker_id,
                        "domain": domain,
                        "spark": base_insight,
                        "spontaneous": True,
                    },
                )
                self.autonomous_neurons_born += len(born_neurons)

            born_ids = [n.neuron_id for n in born_neurons] if born_neurons else []

            # 3. Formação de Memória Episódica Autônoma
            ep_rec = None
            if hasattr(self.dev, "memory") and self.dev.memory:
                ep_rec = self.dev.memory.record_episodic_memory(
                    modality="AUTONOMOUS_THOUGHT",
                    experience_summary=thought_narrative,
                    perceptual_details={
                        "cycle": current_cycle,
                        "worker_id": worker_id,
                        "domain": domain,
                        "concept": concept,
                        "related": related,
                        "insight": base_insight,
                        "autonomous": True,
                    },
                    emotive_valence="DESCOBERTA",
                    salience=0.85,
                    associated_concepts=[concept, related, domain.upper()],
                    created_neuron_ids=born_ids,
                )
                self.autonomous_memories_created += 1

                # Adiciona relação semântica dinâmica
                self.dev.memory.add_semantic_relation(
                    source_concept=concept,
                    target_concept=related,
                    relation="SPONTANEOUS_ASSOCIATION",
                    strength=0.88,
                )

                # Atualiza memória de trabalho imediata
                self.dev.memory.push_working_memory(
                    concept=concept,
                    modality="AUTONOMOUS_THOUGHT",
                    summary=f"Insight [W{worker_id}] #{current_cycle}: {base_insight[:70]}",
                )

                # 4. Salva periodicamente na memória binária (com throttling para manter alto FPS a 2ms)
                now_t = time.time()
                if now_t - self._last_disk_save >= 0.25 or current_cycle % 25 == 0:
                    self._last_disk_save = now_t
                    if hasattr(self.dev.memory, "auto_save_binary"):
                        self.dev.memory.auto_save_binary()

            # 5. Sono Cognitivo e Consolidação Periódica
            sleep_result: Optional[Dict[str, Any]] = None
            if current_cycle % 12 == 0:
                if hasattr(self.dev, "sleep_and_consolidate"):
                    sleep_result = self.dev.sleep_and_consolidate(user_id=f"worker_{worker_id}")

            # 6. Atualização ontogenética
            if hasattr(self.dev, "_update_stage"):
                self.dev._update_stage()

            entry = {
                "cycle": current_cycle,
                "worker_id": worker_id,
                "timestamp": now_iso,
                "concept": concept,
                "domain": domain,
                "related_concept": related,
                "insight": base_insight,
                "neurons_born": born_ids,
                "memory_id": ep_rec.memory_id if ep_rec else None,
                "sleep_consolidated": sleep_result is not None,
                "developmental_stage": getattr(self.dev, "current_stage", "Desconhecido"),
                "cognitive_age_days": round(getattr(self.dev, "cognitive_age_days", 1.0), 2),
            }

            self.thought_history.append(entry)
            if len(self.thought_history) > 60:
                self.thought_history.pop(0)

            return entry

    def get_status(self) -> Dict[str, Any]:
        """Retorna telemetria completa, workers e estado de consciência em tempo real."""
        dev_stage = getattr(self.dev, "current_stage", "Em Desenvolvimento")
        cog_age = round(getattr(self.dev, "cognitive_age_days", 1.0), 2)
        total_neurons = len(self.dev.neurogenesis.circuit.neurons) if hasattr(self.dev, "neurogenesis") else 0
        total_memories = len(self.dev.memory.episodic_memories) if hasattr(self.dev, "memory") else 0

        return {
            "is_running": self.is_running,
            "workers": self.workers,
            "interval_seconds": self.interval_seconds,
            "interval_ms": round(self.interval_seconds * 1000.0, 2),
            "cycles_completed": self.cycles_completed,
            "autonomous_neurons_born": self.autonomous_neurons_born,
            "autonomous_memories_created": self.autonomous_memories_created,
            "current_thought": self.current_thought,
            "developmental_stage": dev_stage,
            "cognitive_age_days": cog_age,
            "total_neurons_in_circuit": total_neurons,
            "total_episodic_memories": total_memories,
            "recent_thoughts": self.thought_history[-10:],
        }

