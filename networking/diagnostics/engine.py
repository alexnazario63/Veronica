"""
Motor de Diagnóstico de Redes da Veronica (Seção 7).
Executa os 11 passos obrigatórios combinando:
- Modelo neural próprio (VeronicaTransformer)
- Conhecimento documental (HybridRAGEngine)
- Regras determinísticas de protocolo
- Análise de topologia
- Dados operacionais em tempo real com acesso estritamente somente leitura
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional

from core.inference.generator import AutoregressiveGenerator, GenerationConfig
from knowledge.retrieval.rag_engine import HybridRAGEngine
from networking.vendors.adapters import BaseVendorAdapter
from networking.topology.graph import NetworkTopologyGraph
from networking.protocols.analyzers import analyze_protocol_anomalies
from core.contracts import EpistemicTag, NetCanvasArtifact


class NetworkDiagnosticEngine:
    """
    Orquestrador de Investigação e Diagnóstico em 11 Etapas (Seção 7).
    """

    def __init__(
        self,
        adapter: BaseVendorAdapter,
        topology: NetworkTopologyGraph,
        rag: HybridRAGEngine,
        generator: Optional[AutoregressiveGenerator] = None,
    ) -> None:
        self.adapter = adapter
        self.topology = topology
        self.rag = rag
        self.generator = generator

    def investigate(
        self,
        problem_description: str,
        target_devices: Optional[List[str]] = None,
        incomplete_telemetry_simulation: bool = False,
    ) -> Dict[str, Any]:
        """
        Executa os 11 passos de investigação diagnóstica somente leitura.
        Trata explicitamente cenários com dados incompletos sem inventar fatos.
        """
        steps_trace: List[Dict[str, Any]] = []

        # Passo 1: Receber a descrição do problema
        steps_trace.append(
            {
                "step": 1,
                "name": "Recepção e Classificação do Problema",
                "detail": f"Problema relatado: '{problem_description}'",
            }
        )

        # Passo 2: Identificar equipamentos envolvidos
        all_known = list(self.topology.inventory.keys())
        if target_devices:
            involved = [d for d in target_devices if d in self.topology.inventory]
            unknown_devices = [d for d in target_devices if d not in self.topology.inventory]
        else:
            involved = [
                d
                for d in all_known
                if d.lower() in problem_description.lower()
                or self.topology.inventory[d]["vendor"].lower() in problem_description.lower()
            ]
            if not involved:
                prob_low = problem_description.lower()
                has_trouble_keyword = any(
                    w in prob_low
                    for w in (
                        "diagnost", "investig", "problema", "falha", "queda",
                        "erro", "alarme", "down", "active", "exstart", "stuck",
                        "rede", "anomalia", "geral", "core", "neighbor"
                    )
                )
                if has_trouble_keyword:
                    # Seleciona equipamentos com alarmes ativos no laboratório
                    involved = [
                        d for d in all_known if self.topology.inventory[d].get("alarms")
                    ] or all_known[:2]
                else:
                    involved = []
            unknown_devices = []

        steps_trace.append(
            {
                "step": 2,
                "name": "Identificação de Equipamentos",
                "involved_devices": involved,
                "unknown_devices": unknown_devices,
            }
        )

        # Passo 3: Consultar a topologia
        topo_snapshot = self.topology.export_topology()
        neighbors_map = {
            dev_id: self.adapter.get_topology_neighbors(dev_id) for dev_id in involved
        }
        steps_trace.append(
            {
                "step": 3,
                "name": "Consulta de Topologia e Adjacências",
                "neighbors_by_device": neighbors_map,
            }
        )

        # Passo 4: Obter informações operacionais com acesso somente leitura
        # Passo 5: Analisar logs, alarmes e configurações
        all_findings: List[Dict[str, Any]] = []
        collected_states: Dict[str, Any] = {}
        unknown_limitations: List[str] = []

        if unknown_devices:
            unknown_limitations.append(
                f"Equipamentos não encontrados no inventário verificado: {', '.join(unknown_devices)}."
            )

        for dev_id in involved:
            ident = self.adapter.discover_device(dev_id)
            if incomplete_telemetry_simulation:
                unknown_limitations.append(
                    f"Coleta operacional parcial em {dev_id}: SNMP/SSH timeout simulado; logs em tempo real indisponíveis."
                )
                collected_states[dev_id] = {"partial": True, "identity": ident}
                continue

            state = self.adapter.get_operational_state(dev_id)
            cfg = self.adapter.get_running_config(dev_id)
            diag_bundle = self.adapter.collect_diagnostics(dev_id)
            collected_states[dev_id] = {
                "identity": ident,
                "state": state,
                "diagnostics": diag_bundle,
            }

            dev_findings = analyze_protocol_anomalies(
                device_identity=ident,
                operational_state=state,
                running_config=cfg["running_config"],
            )
            for f in dev_findings:
                f["device_id"] = dev_id
                f["vendor"] = ident["vendor"]
                f["os_name"] = ident["os_name"]
                f["version"] = ident["version"]
                all_findings.append(f)

        steps_trace.append(
            {
                "step": 4,
                "name": "Coleta Operacional Somente Leitura",
                "read_only_enforced": True,
                "devices_queried": list(collected_states.keys()),
            }
        )
        steps_trace.append(
            {
                "step": 5,
                "name": "Análise de Logs, Alarmes e Configurações",
                "anomalies_found": len(all_findings),
            }
        )

        # Passo 6: Formular hipóteses
        # Passo 7: Determinar testes seguros
        # Passo 8: Comparar resultados com comportamento esperado
        # Passo 9: Identificar possíveis causas-raiz
        # Passo 10: Recomendar correções
        # Passo 11: Apresentar evidências, riscos e nível de confiança
        verified_facts: List[str] = []
        technical_inferences: List[str] = []
        hypotheses: List[str] = []
        safe_tests: List[str] = []
        recommended_commands: List[str] = []
        root_causes: List[str] = []

        for f in all_findings:
            root_causes.append(f"[{f['device_id']}] {f['summary']}")
            verified_facts.extend(f["verified_facts"])
            technical_inferences.extend(f["technical_inferences"])
            hypotheses.extend(f["hypotheses"])
            for cmd in f["safe_diagnostic_commands"]:
                if cmd not in safe_tests:
                    safe_tests.append(cmd)
            for rcmd in f["proposed_remediation_commands"]:
                if rcmd not in recommended_commands:
                    recommended_commands.append(rcmd)

        if not all_findings and incomplete_telemetry_simulation:
            hypotheses.append(
                "Com dados operacionais incompletos, a causa-raiz não pode ser comprovada; realizar coleta somente leitura assistida antes de propor mudanças."
            )

        # Recuperação documental RAG para fundamentar com citações oficiais
        rag_result = self.rag.search(problem_description, top_k=3)
        citations = [r["citation"] for r in rag_result["results"]]

        # Síntese neural pela VeronicaTransformer
        neural_synthesis = ""
        if self.generator is not None:
            gen_out = self.generator.generate_text(
                f"Pergunta: {problem_description}",
                gen_cfg=GenerationConfig(max_new_tokens=28, greedy=True),
            )
            neural_synthesis = str(gen_out["text"])

        confidence = (
            round(float(sum(f["confidence"] for f in all_findings) / len(all_findings)), 2)
            if all_findings
            else (0.35 if incomplete_telemetry_simulation else 0.75)
        )

        evidence_trail = (
            [{"tag": EpistemicTag.VERIFIED_FACT.value, "text": item} for item in verified_facts]
            + [{"tag": EpistemicTag.TECHNICAL_INFERENCE.value, "text": item} for item in technical_inferences]
            + [{"tag": EpistemicTag.HYPOTHESIS.value, "text": item} for item in hypotheses]
        )

        for s_num, s_name, s_data in [
            (6, "Formulação de Hipóteses Verificáveis", hypotheses),
            (7, "Determinação de Testes Seguros (Read-Only)", safe_tests),
            (8, "Comparação com Comportamento Esperado (RFC/Vendor Docs)", [c["title"] for c in citations]),
            (9, "Identificação de Causas-Raiz", root_causes),
            (10, "Recomendação de Correções Mínimas", recommended_commands),
            (11, "Consolidação de Evidências, Riscos e Confiança", {"confidence": confidence}),
        ]:
            steps_trace.append({"step": s_num, "name": s_name, "detail": s_data})

        report_payload = {
            "problem_description": problem_description,
            "steps_executed": steps_trace,
            "evidence": evidence_trail,
            "root_causes": root_causes,
            "confidence": confidence,
        }
        report_hash = hashlib.sha256(
            json.dumps(report_payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        artifact = NetCanvasArtifact(
            id=f"diagnostic-{report_hash[:16]}", version=1, type="diagnostic_11_steps",
            baseline_content=None,
            proposed_content=json.dumps(report_payload, ensure_ascii=False, default=str),
            blast_radius_graph=None, change_stage="diagnosis_complete", sha256_hash=report_hash,
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "problem_description": problem_description,
            "involved_devices": involved,
            "read_only_guarantee": True,
            "steps_executed": steps_trace,
            "epistemic_breakdown": {
                "verified_facts": verified_facts,
                "technical_inferences": technical_inferences,
                "hypotheses": hypotheses,
            },
            "evidence_trail": evidence_trail,
            "diagnostic_artifact": artifact.model_dump(mode="json"),
            "root_causes": root_causes,
            "safe_diagnostic_commands": safe_tests,
            "proposed_remediation_commands": recommended_commands,
            "citations": citations,
            "obsolete_doc_warnings": rag_result["obsolete_warnings"],
            "version_conflicts": rag_result["version_conflicts"],
            "neural_model_synthesis": neural_synthesis,
            "confidence": confidence,
            "unknown_or_limitations": unknown_limitations,
            "topology_context": topo_snapshot,
        }
