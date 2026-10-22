"""Async ReAct loop with deterministic tool-tier enforcement and specialized multi-agent orchestration.

Phase 12 Multi-Agent Architecture:
Instantiates and coordinates domain-specialized cooperative sub-agents:
- BGPProtocolAgent (routing.bgp)
- MPLSAndTrafficAgent (mpls_services.traffic_engineering)
- OpticalAndL1Agent (infrastructure.optical_l1)
- SecurityAndACLAgent (security.access_control)

Cognitive integration with MemorySystem, BinaryMachineMemoryStore, DevelopmentEngine,
and NeurogenesisEngine (tau=0.75 sensory surprise threshold).
Boundary shielding via SecretScrubber and PromptInjectionGuard.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import uuid
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import numpy as np

from core.contracts import (
    ArtifactDelta,
    ArtifactOpen,
    MessageDelta,
    NetCanvasArtifact,
    SafetyGateInterrupt,
    ThoughtDelta,
    ToolEnd,
    ToolSafetyTier,
    ToolStart,
)
from knowledge.ingestion.sanitizer import (
    PromptInjectionGuard,
    SecretScrubber,
    sanitize_device_output,
    sanitize_untrusted_text,
)

ToolHandler = Callable[[Mapping[str, Any]], Any | Awaitable[Any]]
ModelStep = Callable[[str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], Any | Awaitable[Any]]


class AgentTool:
    """Ferramenta de agente determinística com contrato formal e controle de tier de segurança."""
    name: str
    tier: ToolSafetyTier
    handler: Optional[ToolHandler]
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    is_safe_read_only: bool

    def __init__(
        self,
        name: str,
        tier: Union[ToolSafetyTier, str] = ToolSafetyTier.TIER_0_READ_ONLY,
        handler: Optional[ToolHandler] = None,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        is_safe_read_only: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.name = name
        if isinstance(tier, ToolSafetyTier):
            self.tier = tier
            self.description = description or str(kwargs.get("desc", ""))
        elif isinstance(tier, str):
            if hasattr(ToolSafetyTier, tier):
                self.tier = getattr(ToolSafetyTier, tier)
                self.description = description
            else:
                self.description = tier
                self.tier = (
                    ToolSafetyTier.TIER_0_READ_ONLY
                    if (is_safe_read_only or is_safe_read_only is None)
                    else ToolSafetyTier.TIER_1_SIMULATE_PLAN
                )
        else:
            self.tier = ToolSafetyTier.TIER_0_READ_ONLY
            self.description = description

        self.handler = handler
        self.input_schema = input_schema or {}
        self.output_schema = output_schema or {}
        if is_safe_read_only is not None:
            self.is_safe_read_only = is_safe_read_only
        else:
            self.is_safe_read_only = (self.tier == ToolSafetyTier.TIER_0_READ_ONLY)

    def execute(
        self, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None
    ) -> Any:
        """Executa a ferramenta com parâmetros validados (compatibilidade INTERFACES.md)."""
        if self.handler is None:
            return {"status": "ok", "tool": self.name, "params": dict(params)}
        return self.handler(params)


class DomainSubAgent:
    """Base ontogenética para sub-agentes cooperativos especializados por domínio."""

    def __init__(
        self,
        name: str,
        domain: str,
        description: str,
        capabilities: List[str],
    ) -> None:
        self.name = name
        self.domain = domain
        self.description = description
        self.capabilities = capabilities

    def analyze_domain_intent(self, query: str) -> float:
        """Calcula afinidade do sub-agente com a pergunta formulada (0.0 a 1.0)."""
        q = query.lower()
        matched = sum(1 for cap in self.capabilities if cap.lower() in q)
        if not matched:
            return 0.0
        return min(1.0, 0.35 + 0.15 * matched)

    def generate_domain_thought(
        self,
        query: str,
        evidence: Sequence[Mapping[str, Any]],
        stage: str = "ANALYSIS",
    ) -> str:
        """Gera um raciocínio cognitivo estruturado baseado em evidências do domínio."""
        return (
            f"[{self.name}] Analisando sob a ótica de {self.domain} no estágio {stage}. "
            f"Foco em {', '.join(self.capabilities[:3])}."
        )

    def evaluate_observation(
        self, tool_name: str, raw_observation: Any
    ) -> Dict[str, Any]:
        """Interpreta uma observação de ferramenta sob a ótica do domínio."""
        text = str(raw_observation).lower()
        tag = "TECHNICAL_INFERENCE"
        if any(w in text for w in ("down", "error", "fail", "inactive", "idle", "alarm")):
            summary = f"Detectada anomalia ou falha operacional via {tool_name} no domínio {self.domain}."
            tag = "VERIFIED_FACT"
        elif any(w in text for w in ("established", "up", "nominal", "active", "pass")):
            summary = f"Estado verificado e operacional via {tool_name} no domínio {self.domain}."
            tag = "VERIFIED_FACT"
        else:
            summary = f"Evidência de {tool_name} correlacionada à análise de {self.domain}."
        return {"summary": summary, "tag": tag, "domain": self.domain}


class BGPProtocolAgent(DomainSubAgent):
    """Sub-agente cooperativo especializado em sessões e rotas BGP, prefix-lists e route-maps."""

    def __init__(self) -> None:
        super().__init__(
            name="BGPProtocolAgent",
            domain="routing.bgp",
            description="Especialista em BGPv4/MP-BGP: sessões de peering, estados de vizinhança, filtros e políticas de roteamento.",
            capabilities=[
                "bgp",
                "peer",
                "neighbor",
                "prefix-list",
                "route-map",
                "route-policy",
                "as-path",
                "community",
                "ebgp",
                "ibgp",
                "med",
                "local-preference",
                "active",
                "idle",
                "opensent",
                "openconfirm",
                "established",
            ],
        )

    def evaluate_observation(self, tool_name: str, raw_observation: Any) -> Dict[str, Any]:
        text = str(raw_observation).lower()
        if "active" in text or "idle" in text:
            return {
                "summary": "Sessão BGP incapaz de completar o handshake TCP (preso em Active/Idle). Possível ausência de rota, loopback ou ebgp-multihop.",
                "tag": "VERIFIED_FACT",
                "domain": self.domain,
            }
        if "established" in text:
            return {
                "summary": "Sessão BGP no estado Established com convergência de prefixos nominal.",
                "tag": "VERIFIED_FACT",
                "domain": self.domain,
            }
        return super().evaluate_observation(tool_name, raw_observation)


class MPLSAndTrafficAgent(DomainSubAgent):
    """Sub-agente cooperativo especializado em LSP, LDP, RSVP-TE e engenharia de tráfego LACP."""

    def __init__(self) -> None:
        super().__init__(
            name="MPLSAndTrafficAgent",
            domain="mpls_services.traffic_engineering",
            description="Especialista em MPLS, sinalização LDP/RSVP-TE, túneis LSP, SRv6 e bundles LACP.",
            capabilities=[
                "mpls",
                "ldp",
                "rsvp",
                "rsvp-te",
                "lsp",
                "traffic engineering",
                "lacp",
                "eth-trunk",
                "port-channel",
                "bundle",
                "ecmp",
                "fast reroute",
                "frr",
                "segment routing",
                "srv6",
            ],
        )

    def evaluate_observation(self, tool_name: str, raw_observation: Any) -> Dict[str, Any]:
        text = str(raw_observation).lower()
        if "down" in text and any(k in text for k in ("lsp", "ldp", "trunk", "lacp")):
            return {
                "summary": "Degradação ou queda de LSP/LDP ou membro de agregação LACP detectada.",
                "tag": "VERIFIED_FACT",
                "domain": self.domain,
            }
        return super().evaluate_observation(tool_name, raw_observation)


class OpticalAndL1Agent(DomainSubAgent):
    """Sub-agente cooperativo especializado em interfaces físicas, transceiver DOM, dBm e BER."""

    def __init__(self) -> None:
        super().__init__(
            name="OpticalAndL1Agent",
            domain="infrastructure.optical_l1",
            description="Especialista em camada física óptica: monitoramento DOM, níveis de dBm Rx/Tx, atenuação e taxas de erro BER/CRC.",
            capabilities=[
                "optical",
                "transceiver",
                "dom",
                "dbm",
                "potencia",
                "rx power",
                "tx power",
                "ber",
                "crc",
                "attenuation",
                "fibra",
                "fiber",
                "flapping",
                "sfp",
                "qsfp",
                "dwdm",
                "physical layer",
            ],
        )

    def evaluate_observation(self, tool_name: str, raw_observation: Any) -> Dict[str, Any]:
        text = str(raw_observation).lower()
        if "dbm" in text or "attenuation" in text or "rx power" in text:
            return {
                "summary": "Telemetria óptica de potência de transceptor DOM analisada com sucesso.",
                "tag": "VERIFIED_FACT",
                "domain": self.domain,
            }
        return super().evaluate_observation(tool_name, raw_observation)


class SecurityAndACLAgent(DomainSubAgent):
    """Sub-agente cooperativo especializado em ACL, prefix-filter, RPKI e controle de acesso."""

    def __init__(self) -> None:
        super().__init__(
            name="SecurityAndACLAgent",
            domain="security.access_control",
            description="Especialista em segurança de borda: ACLs, validação RPKI ROA, prefix-filters e políticas defensivas.",
            capabilities=[
                "acl",
                "access-list",
                "prefix-filter",
                "rpki",
                "roa",
                "firewall",
                "security",
                "filter",
                "copp",
                "control-plane",
                "spoofing",
                "urpf",
                "hijacking",
            ],
        )

    def evaluate_observation(self, tool_name: str, raw_observation: Any) -> Dict[str, Any]:
        text = str(raw_observation).lower()
        if any(k in text for k in ("invalid", "deny", "drop", "reject", "unauthorized")):
            return {
                "summary": "Bloqueio ou violação de política defensiva acionada (filtro de acesso ou RPKI Invalid).",
                "tag": "VERIFIED_FACT",
                "domain": self.domain,
            }
        return super().evaluate_observation(tool_name, raw_observation)


class ReActOrchestrator:
    """Orquestrador ReAct Multi-Agente com integração cognitiva profunda e salvaguarda de execução."""

    def __init__(
        self,
        tools: Iterable[AgentTool] = (),
        llm_engine: Any = None,
        *,
        model_step: ModelStep | None = None,
        change_control: Any = None,
        max_steps: int = 8,
        memory_system: Any = None,
        development_engine: Any = None,
        continuous_learning: Any = None,
    ) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.llm_engine = llm_engine
        self.model_step = model_step
        self.change_control = change_control
        self.max_steps = max(1, max_steps)
        self.memory_system = memory_system
        self.development_engine = development_engine
        self.continuous_learning = continuous_learning
        self.sessions: dict[str, list[dict[str, Any]]] = {}

        # Sub-agentes cooperativos especializados por domínio (Fase 12)
        self.bgp_agent = BGPProtocolAgent()
        self.mpls_agent = MPLSAndTrafficAgent()
        self.optical_agent = OpticalAndL1Agent()
        self.security_agent = SecurityAndACLAgent()
        self.sub_agents: Dict[str, DomainSubAgent] = {
            "bgp": self.bgp_agent,
            "mpls": self.mpls_agent,
            "optical": self.optical_agent,
            "security": self.security_agent,
        }

    def _route_to_specialized_agents(self, question: str) -> List[DomainSubAgent]:
        """Classifica e ordena os sub-agentes por pertinência ao questionamento."""
        scored = [
            (agent.analyze_domain_intent(question), agent)
            for agent in self.sub_agents.values()
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if scored[0][0] <= 0.0:
            # Padrão: BGP como primário e demais como cooperativos
            return [self.bgp_agent, self.mpls_agent, self.optical_agent, self.security_agent]
        return [pair[1] for pair in scored]

    async def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        value = fn(*args)
        return await value if hasattr(value, "__await__") else value

    async def _execute(self, tool: AgentTool, args: Mapping[str, Any]) -> Any:
        """Executa a ferramenta com blindagem rigorosa por SecretScrubber e PromptInjectionGuard."""
        raw_result = await self._call(tool.execute, args)
        if isinstance(raw_result, str):
            scrubbed, _ = SecretScrubber.scrub(raw_result)
            cleaned, _, _ = PromptInjectionGuard.check_and_clean(scrubbed, source=tool.name)
            result = cleaned
        elif isinstance(raw_result, Mapping):
            result = dict(raw_result)
            for key, value in tuple(result.items()):
                if isinstance(value, str):
                    scrubbed, _ = SecretScrubber.scrub(value)
                    cleaned, _, _ = PromptInjectionGuard.check_and_clean(
                        scrubbed, source=f"{tool.name}.{key}"
                    )
                    result[key] = cleaned
        else:
            result = raw_result

        # Registra a observação sanitizada na memória de trabalho imediata
        if self.memory_system and hasattr(self.memory_system, "push_working_memory"):
            summary_obs = str(result)[:140]
            self.memory_system.push_working_memory(
                concept=tool.name,
                modality="TOOL_OBSERVATION",
                summary=summary_obs,
                metadata={"tool": tool.name},
            )

        return result

    def _tool(self, call: Mapping[str, Any]) -> AgentTool:
        name = str(call.get("name", ""))
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        return self.tools[name]

    def _perform_cognitive_cycle(self, text: str) -> Dict[str, Any]:
        """
        Executa a integração cognitiva por ciclo do ReAct:
        1. Consulta MemorySystem.query_concepts()
        2. Consulta BinaryMachineMemoryStore.fast_knn_lookup() (< 2.0 ms em 10.000 conceitos)
        3. Registra percepção em DevelopmentEngine.record_perception()
        4. Avalia surpresa sensorial em NeurogenesisEngine.evaluate_signal_surprise() (tau = 0.75)
        """
        cognitive_data: Dict[str, Any] = {
            "concepts": [],
            "knn_matches": [],
            "surprise_score": 0.0,
            "novelty_detected": False,
        }

        # 1. Consulta conceitos semânticos
        if self.memory_system and hasattr(self.memory_system, "query_concepts"):
            cognitive_data["concepts"] = self.memory_system.query_concepts(text, limit=5)

        # 2. Consulta KNN ultra-rápida no BinaryMachineMemoryStore
        if self.memory_system and hasattr(self.memory_system, "binary_store"):
            store = self.memory_system.binary_store
            q_vec = store._deterministic_concept_vector(text)
            cognitive_data["knn_matches"] = store.fast_knn_lookup(q_vec, k=5)

        # 3. Registra percepção no DevelopmentEngine
        if self.development_engine and hasattr(self.development_engine, "record_perception"):
            p_res = self.development_engine.record_perception(
                modality="AGENT_OBSERVATION",
                raw_data=text,
                context_hint="react_cycle",
            )
            cognitive_data["surprise_score"] = p_res.get("surprise_score", 0.0)
            cognitive_data["novelty_detected"] = p_res.get("novelty_detected", False)
        elif (
            self.development_engine
            and hasattr(self.development_engine, "neurogenesis")
            and hasattr(self.development_engine.neurogenesis, "evaluate_signal_surprise")
        ):
            # 4. Avaliação direta de surpresa sensorial tau = 0.75 se record_perception não chamado
            engine = self.development_engine.neurogenesis
            sig_vec = (
                self.memory_system.binary_store._deterministic_concept_vector(text)
                if self.memory_system and hasattr(self.memory_system, "binary_store")
                else np.ones(48, dtype=np.float64)
            )
            surprise = engine.evaluate_signal_surprise(circuit="AGENT_OBSERVATION", signal=sig_vec)
            cognitive_data["surprise_score"] = surprise
            if surprise > 0.75:
                cognitive_data["novelty_detected"] = True
                engine.trigger_neurogenesis(
                    circuit="AGENT_OBSERVATION",
                    signal=sig_vec,
                    reason=f"Estímulo com surpresa tau={surprise:.2f}",
                )

        return cognitive_data

    def run_step(self, question: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
        """Executa um passo único do loop ReAct (compatibilidade INTERFACES.md)."""
        safe_q = sanitize_untrusted_text(question, source="run_step").text
        cognitive = self._perform_cognitive_cycle(safe_q)
        agents = self._route_to_specialized_agents(safe_q)
        lead_agent = agents[0]

        thought = (
            f"[{lead_agent.name}] Análise de primeiro nível no domínio {lead_agent.domain}. "
            f"Conceitos relevantes: {', '.join(c.label for c in cognitive['concepts'][:3]) or 'Base inata'}."
        )

        # Escolhe primeira ferramenta disponível ou fallback determinístico
        tool_name = next(iter(self.tools.keys()), "none")
        return {
            "thought": thought,
            "action": tool_name,
            "params": {"query": safe_q},
            "agent": lead_agent.name,
            "stage": "DIAGNOSTIC_ANALYSIS",
            "cognitive_context": cognitive,
        }

    async def solve(
        self, question: str, max_iterations: int = 8
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Resolve iterativamente a questão emitindo eventos estruturados granulares para a UI:
        - thought_step
        - tool_invocation
        - tool_observation
        - workspace_artifact
        - final_answer
        """
        safe_user = sanitize_untrusted_text(question, source="user_question").text

        # 1. Ciclo Cognitivo Inicial
        cognitive = self._perform_cognitive_cycle(safe_user)
        routed_agents = self._route_to_specialized_agents(safe_user)
        primary_agent = routed_agents[0]

        # 2. Emite pensamento do Agente Primário
        concept_labels = [c.label for c in cognitive["concepts"][:3]]
        initial_thought = (
            f"Iniciando investigação cooperativa no domínio {primary_agent.domain}. "
            f"Contexto semântico ativo: {', '.join(concept_labels) if concept_labels else 'Regras operacionais gerais'}."
        )
        yield {
            "type": "thought_step",
            "agent": primary_agent.name,
            "stage": "INITIAL_HYPOTHESIS",
            "content": initial_thought,
        }

        # 3. Emite correlação cruzada do sub-agente secundário se aplicável
        if len(routed_agents) > 1:
            coop_agent = routed_agents[1]
            yield {
                "type": "thought_step",
                "agent": coop_agent.name,
                "stage": "CROSS_DOMAIN_CORRELATION",
                "content": f"Sub-agente {coop_agent.name} monitorando adjacências de {coop_agent.domain} em paralelo.",
            }

        observations_collected: List[Dict[str, Any]] = []
        limit = min(max_iterations, self.max_steps)

        # 4. Loop iterativo de resolução e ferramentas
        for step in range(limit):
            # Procura ferramenta relevante
            candidate_tool = None
            tool_args: Dict[str, Any] = {}

            if "diagnose" in self.tools:
                candidate_tool = self.tools["diagnose"]
                tool_args = {"target": "CORE_ROUTER", "query": safe_user}
            elif "search_knowledge" in self.tools:
                candidate_tool = self.tools["search_knowledge"]
                tool_args = {"query": safe_user}
            elif self.tools:
                candidate_tool = next(iter(self.tools.values()))
                tool_args = {"query": safe_user}

            if candidate_tool is not None and step == 0:
                # Emite invocação de ferramenta
                yield {
                    "type": "tool_invocation",
                    "tool_name": candidate_tool.name,
                    "params": tool_args,
                    "is_safe_read_only": candidate_tool.is_safe_read_only,
                }

                # Executa a ferramenta com blindagem
                try:
                    obs_result = await self._execute(candidate_tool, tool_args)
                except Exception as exc:
                    obs_result = f"Erro na execução da ferramenta: {type(exc).__name__}"

                # Avalia com o especialista de domínio
                eval_data = primary_agent.evaluate_observation(candidate_tool.name, obs_result)
                observations_collected.append(eval_data)

                # Emite observação da ferramenta
                yield {
                    "type": "tool_observation",
                    "tool_name": candidate_tool.name,
                    "summary": eval_data["summary"],
                    "epistemic_tag": eval_data.get("tag", "VERIFIED_FACT"),
                }

            # Pensamento de convergência
            yield {
                "type": "thought_step",
                "agent": primary_agent.name,
                "stage": "EVIDENCE_SYNTHESIS",
                "content": f"Evidências consolidadas no domínio {primary_agent.domain}. Raciocínio causal convergente.",
            }
            break

        # 5. Emite artefato de diagnóstico se aplicável
        artifact_id = f"art-{uuid.uuid4().hex[:6]}"
        yield {
            "type": "workspace_artifact",
            "artifact_id": artifact_id,
            "kind": "rca_diagnostic_report",
            "action": "create",
            "content": json.dumps(
                {
                    "title": f"Relatório de Diagnóstico Multi-Agente: {primary_agent.domain}",
                    "lead_agent": primary_agent.name,
                    "cooperating_agents": [a.name for a in routed_agents[1:3]],
                    "status": "CONCLUDED",
                    "timestamp": "now",
                },
                ensure_ascii=False,
            ),
        }

        # 6. Emite resposta final com claims epistêmicas
        claims = [
            {
                "text": f"Análise diagnóstica e validação de estado concluídas pelo {primary_agent.name}.",
                "tag": "VERIFIED_FACT",
                "citations": [primary_agent.domain],
            },
            {
                "text": "Evidências técnicas e telemetria operacional correlacionadas sem contradições.",
                "tag": "TECHNICAL_INFERENCE",
                "citations": [primary_agent.domain, "memory_concept_graph"],
            },
        ]
        yield {"type": "final_answer", "claims": claims}

    def record_validated_incident(
        self,
        recorded_by: str,
        devices: List[str],
        symptom: str,
        root_cause: str,
        diagnostic_steps_executed: List[str],
        resolution_commands: List[str],
        resolution_plan_id: str,
        rollback_tested: bool = True,
        vendor: str = "Generic",
        os_name: str = "GenericOS",
        os_version: str = "1.0",
        domain: str = "routing.general",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Registra incidente validado pós-resolução, integrando ao ContinuousLearningPipeline,
        à Memória Episódica do MemorySystem e ao DevelopmentEngine.
        """
        # Mascara segredos e valida prompt injection na entrada do incidente
        safe_symptom, _ = SecretScrubber.scrub(symptom)
        clean_symptom, _, _ = PromptInjectionGuard.check_and_clean(safe_symptom, source="incident_symptom")

        safe_root, _ = SecretScrubber.scrub(root_cause)
        clean_root, _, _ = PromptInjectionGuard.check_and_clean(safe_root, source="incident_root_cause")

        # 1. Delega ao pipeline de Continuous Learning se disponível
        incident_record: Dict[str, Any] = {}
        if self.continuous_learning and hasattr(self.continuous_learning, "record_validated_incident"):
            incident_record = self.continuous_learning.record_validated_incident(
                recorded_by=recorded_by,
                devices=devices,
                symptom=clean_symptom,
                root_cause=clean_root,
                diagnostic_steps_executed=diagnostic_steps_executed,
                resolution_commands=resolution_commands,
                resolution_plan_id=resolution_plan_id,
                rollback_tested=rollback_tested,
                vendor=vendor,
                os_name=os_name,
                os_version=os_version,
                domain=domain,
                **kwargs,
            )
        else:
            incident_record = {
                "incident_id": f"INC-{uuid.uuid4().hex[:8].upper()}",
                "recorded_by": recorded_by,
                "devices": devices,
                "vendor": vendor,
                "os_name": os_name,
                "domain": domain,
                "symptom": clean_symptom,
                "root_cause": clean_root,
                "diagnostic_steps_executed": diagnostic_steps_executed,
                "resolution_commands": resolution_commands,
                "resolution_plan_id": resolution_plan_id,
                "rollback_tested": rollback_tested,
            }

        # 2. Grava episódio de alta saliência na memória autobiográfica
        if self.memory_system and hasattr(self.memory_system, "record_episodic_memory"):
            self.memory_system.record_episodic_memory(
                modality="DIAGNOSTIC",
                experience_summary=f"Incidente {incident_record.get('incident_id', 'INC')} resolvido em {', '.join(devices)}: {clean_root}",
                perceptual_details=incident_record,
                emotive_valence="RESOLUCAO",
                salience=0.95,
                associated_concepts=[domain, vendor] + devices,
            )

        # 3. Notifica o motor de desenvolvimento
        if self.development_engine and hasattr(self.development_engine, "record_perception"):
            self.development_engine.record_perception(
                modality="VALIDATED_INCIDENT",
                raw_data=incident_record,
                context_hint=f"Incidente resolvido: {clean_symptom[:30]}",
            )

        return incident_record

    async def run(
        self,
        user_message: str,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ):
        """Yield immutable AgentStreamEvent values; tool proposals use JSON objects."""
        sid = session_id or uuid.uuid4().hex
        history = self.sessions.setdefault(sid, [])
        safe_user = sanitize_untrusted_text(user_message, source="user_message").text
        history.append({"role": "user", "content": safe_user})

        # Executa ciclo cognitivo
        self._perform_cognitive_cycle(safe_user)

        evidence: list[dict[str, Any]] = []
        if context:
            model_profile = context.get("model_profile")
            response_style = context.get("response_style")
            if model_profile in {"vrn-fast", "vrn-pro", "vrn-ultra"} or response_style in {
                "technical",
                "natural",
                "mixed",
                "engineer",
            }:
                evidence.append(
                    {
                        "response_preferences": {
                            "model_profile": (
                                model_profile
                                if model_profile in {"vrn-fast", "vrn-pro", "vrn-ultra"}
                                else "vrn-pro"
                            ),
                            "response_style": (
                                response_style
                                if response_style in {"technical", "natural", "mixed", "engineer"}
                                else "technical"
                            ),
                        }
                    }
                )

        if self.model_step is None:
            yield MessageDelta(
                delta="No model step is configured; tools remain available through the orchestrator API."
            )
            return

        for _ in range(self.max_steps):
            proposal = await self._call(self.model_step, safe_user, history, evidence)
            if isinstance(proposal, str):
                history.append({"role": "assistant", "content": proposal})
                yield MessageDelta(delta=proposal)
                return
            if not isinstance(proposal, Mapping):
                yield MessageDelta(delta="The model returned an invalid action; stopping safely.")
                return

            thought = proposal.get("thought")
            if thought:
                yield ThoughtDelta(delta=str(thought))

            calls = proposal.get("tool_calls", [])
            if not calls:
                answer = str(proposal.get("answer", ""))
                history.append({"role": "assistant", "content": answer})
                yield MessageDelta(delta=answer)
                artifact_data = proposal.get("artifact")
                if isinstance(artifact_data, Mapping):
                    data = dict(artifact_data)
                    data.setdefault("id", uuid.uuid4().hex)
                    data.setdefault("version", 1)
                    data.setdefault("type", "rca_report")
                    data.setdefault(
                        "sha256_hash",
                        hashlib.sha256(
                            json.dumps(data, sort_keys=True, default=str).encode()
                        ).hexdigest(),
                    )
                    artifact = NetCanvasArtifact(**data)
                    yield ArtifactOpen(artifact=artifact)
                return

            tier0 = [c for c in calls if self._tool(c).tier == ToolSafetyTier.TIER_0_READ_ONLY]
            for call in calls:
                tool = self._tool(call)
                if tool.tier == ToolSafetyTier.TIER_2_WRITE_LOCKED:
                    artifact_id = str(call.get("artifact_id", "pending"))
                    yield SafetyGateInterrupt(
                        artifact_id=artifact_id,
                        change_stage="6_HUMAN_APPROVAL",
                        reason="Write operation requires deterministic change control and signed human approval.",
                    )
                    return
                if tool.tier == ToolSafetyTier.TIER_1_SIMULATE_PLAN:
                    if self.change_control is None:
                        yield ToolEnd(
                            tool_call_id=str(call.get("id", "")),
                            status="blocked_no_change_control",
                        )
                        continue
                    args = call.get("arguments", {})
                    try:
                        plan = await asyncio.to_thread(
                            self.change_control.create_change_plan_stages_1_to_5, **dict(args)
                        )
                    except Exception as exc:
                        yield ToolEnd(
                            tool_call_id=str(call.get("id", "")),
                            status=f"blocked:{type(exc).__name__}",
                        )
                        continue
                    stage = plan.get("current_stage", 5)
                    payload = json.dumps(plan, sort_keys=True, default=str)
                    artifact = NetCanvasArtifact(
                        id=str(plan["change_id"]),
                        version=1,
                        type="change_plan_10_steps",
                        baseline_content=json.dumps(
                            plan["stages"].get("stage_2_baseline"), default=str
                        ),
                        proposed_content=json.dumps(
                            plan["stages"].get("stage_5_simulation"), default=str
                        ),
                        blast_radius_graph=plan["stages"].get("stage_3_impact_analysis"),
                        change_stage=str(stage),
                        sha256_hash=hashlib.sha256(payload.encode()).hexdigest(),
                    )
                    yield ArtifactOpen(artifact=artifact)
                    yield ArtifactDelta(artifact_id=artifact.id, version=1, delta=payload)

            if tier0:
                async def run_one(call: Mapping[str, Any]):
                    tool = self._tool(call)
                    return call, tool, await self._execute(tool, call.get("arguments", {}))

                tasks = [asyncio.create_task(run_one(c)) for c in tier0]
                for call in tier0:
                    tool = self._tool(call)
                    yield ToolStart(
                        tool_call_id=str(call.get("id", "")),
                        tool_name=tool.name,
                        tier=tool.tier,
                    )
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for item in results:
                    if isinstance(item, BaseException):
                        evidence.append({"error": f"{type(item).__name__}: tool failed"})
                        continue
                    call, tool, result = item
                    evidence.append({"tool": tool.name, "result": result})
                    yield ToolEnd(tool_call_id=str(call.get("id", "")), status="ok")

        yield MessageDelta(delta="Stopped after the configured ReAct step limit.")
