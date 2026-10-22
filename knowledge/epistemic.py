"""
Motor de Classificação Epistêmica Dinâmica da Veronica.
Avalia empiricamente declarações, evidências, telemetrias e saídas de ferramentas,
calculando dinamicamente etiquetas epistêmicas com base em critérios de prova:
- VERIFIED_FACT: Comprovado por execução real de código no sandbox, saída de ferramenta real
  ou documento indexado e consultado no catálogo com alta confiabilidade (score >= 0.85).
- TECHNICAL_INFERENCE: Dedução lógica/matemática baseada em premissas ou correlação de telemetria.
- HYPOTHESIS: Hipótese aberta ou proposta teórica sujeita a teste empírico.
"""

from __future__ import annotations

from enum import StrEnum
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from knowledge.sources.catalog import DynamicKnowledgeCatalog


class EpistemicTag(StrEnum):
    VERIFIED_FACT = "VERIFIED_FACT"
    TECHNICAL_INFERENCE = "TECHNICAL_INFERENCE"
    HYPOTHESIS = "HYPOTHESIS"


_EXPLICIT_DEDUCTION_MARKERS = re.compile(
    r"(?:^\s*(?:portanto|consequentemente|logo|deduz-se|implica-se|por\s+conseguinte)\b|"
    r"\bportanto\b|\bcausa\s+prov[aá]vel\b|\bconsequentemente\b)",
    re.IGNORECASE,
)

_INFERENCE_MARKERS = re.compile(
    r"(?:\bportanto\b|\bconsequentemente\b|\blogo\b|\bdevido\s+a\b|\bimplica(?:\s+que)?\b|"
    r"\bdeduz-se(?:\s+que)?\b|\bresulta\s+(?:em|de)\b|\bpor\s+causa\s+de\b|\bindica\s+que\b|"
    r"\bcausa\s+prov[aá]vel\b|\bcorrelaciona-se\b|\bcorrela[cç][aã]o\b|\bc[aá]lculo\b|"
    r"\btaxa\s+calculada\b|\btherefore\b|\bhence\b|\bconsequently\b|\bimplies\b|\bbecause\b)",
    re.IGNORECASE,
)

_HYPOTHESIS_MARKERS = re.compile(
    r"(?:\bhip[oó]tese\b|\bposs[ií]vel\b|\bpode\s+ser\b|\bsuspeita-se\b|\bsugere-se\b|"
    r"\bprovavelmente\b|\bsujeit[oa]\s+a\s+teste\b|\bproposta\s+te[oó]rica\b|\bteoricamente\b|"
    r"\bnecessita\s+(?:de\s+)?confirma[cç][aã]o\b|\bvalida[cç][aã]o\s+emp[ií]rica\b|"
    r"\bcandidat[oa]\b|\bn[aã]o\s+comprovad[oa]\b|\bdados\s+incompletos\b|"
    r"\bperhaps\b|\bmaybe\b|\bpossibly\b|\bhypothesis\b|\bconjecture\b)",
    re.IGNORECASE,
)

_FACT_ACTION_MARKERS = re.compile(
    r"(?:\bexecu[cç][aã]o\s+confirmada\b|\bverificad[oa]\s+em\s+tempo\s+real\b|"
    r"\bsa[ií]da\s+do\s+comando\b|\bresultado\s+do\s+ping\b|\bstatus\s+ativo\b|"
    r"\btransceiver\s+retornou\b|\btelemetria\s+coletada\b|\brunning-config\b|"
    r"\bRFC\s+\d+\b|\bmanual\s+oficial\b|\bexit\s+code\s+0\b|\bup\s+for\b)",
    re.IGNORECASE,
)


class DynamicEpistemicClassifier:
    """
    Classificador Epistêmico Dinâmico e Empírico.
    Substitui heurísticas estáticas por avaliação dinâmica baseada em evidências.
    """

    def __init__(self, catalog: Optional[DynamicKnowledgeCatalog] = None) -> None:
        self.catalog = catalog or DynamicKnowledgeCatalog.get_default()

    def classify(
        self,
        statement: str,
        context: Optional[Dict[str, Any]] = None,
        evidence: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Calcula dinamicamente a etiqueta epistêmica para uma declaração ou achado.
        """
        ctx = context or {}
        raw_text = str(statement).strip()

        # 1. Hipótese teórica aberta ou telemetria incompleta
        if ctx.get("incomplete_telemetry_simulation") or ctx.get("incomplete_telemetry"):
            return {
                "tag": EpistemicTag.HYPOTHESIS,
                "confidence": 0.40,
                "basis": "INCOMPLETE_TELEMETRY",
                "reasoning": "Hipótese aberta: telemetria operacional incompleta; requer validação empírica.",
            }

        if _HYPOTHESIS_MARKERS.search(raw_text):
            return {
                "tag": EpistemicTag.HYPOTHESIS,
                "confidence": 0.55,
                "basis": "HYPOTHETICAL_FORMULATION",
                "reasoning": "Hipótese teórica aberta sujeita a teste empírico ou coleta confirmatória.",
            }

        # 2. Se for uma dedução lógica explícita ("Portanto...", "Causa provável é...")
        if _EXPLICIT_DEDUCTION_MARKERS.search(raw_text):
            return {
                "tag": EpistemicTag.TECHNICAL_INFERENCE,
                "confidence": 0.88,
                "basis": "LOGICAL_MATHEMATICAL_DEDUCTION",
                "reasoning": "Dedução lógica/matemática inferida a partir de premissas ou correlação de telemetria.",
            }

        # 3. VERIFIED_FACT: Comprovado por execução real de código no sandbox
        sandbox_data = ctx.get("sandbox_execution") or ctx.get("sandbox_result")
        if sandbox_data:
            exit_code = (
                sandbox_data.get("exit_code")
                if isinstance(sandbox_data, dict)
                else getattr(sandbox_data, "exit_code", 0)
            )
            if exit_code == 0:
                return {
                    "tag": EpistemicTag.VERIFIED_FACT,
                    "confidence": 1.0,
                    "basis": "SANDBOX_CODE_EXECUTION",
                    "reasoning": "Comprovado por execução real de código no sandbox com sucesso (exit_code 0).",
                }

        # 4. VERIFIED_FACT: Comprovado por saída operacional real de ferramenta
        tool_output = ctx.get("tool_output") or ctx.get("command_output") or ctx.get("telemetry_raw")
        if tool_output and (isinstance(tool_output, (str, list, dict)) and len(tool_output) > 0):
            return {
                "tag": EpistemicTag.VERIFIED_FACT,
                "confidence": 0.98,
                "basis": "REAL_TOOL_OUTPUT",
                "reasoning": "Comprovado por saída operacional de ferramenta/coleta de comando em equipamento real.",
            }

        # 5. VERIFIED_FACT: Documento indexado e consultado com alta confiabilidade (score >= 0.85)
        citation = ctx.get("citation") or (evidence if isinstance(evidence, dict) and "reliability_score" in evidence else None)
        if citation and isinstance(citation, dict):
            rel_score = float(citation.get("reliability_score", 0.0))
            is_off = bool(citation.get("is_official", True))
            if rel_score >= 0.85 and is_off:
                return {
                    "tag": EpistemicTag.VERIFIED_FACT,
                    "confidence": round(rel_score, 2),
                    "basis": "OFFICIAL_CATALOG_DOCUMENT",
                    "reasoning": f"Comprovado por documento oficial indexado no catálogo ({citation.get('doc_id', 'N/A')}) com confiabilidade {rel_score:.2f}.",
                }

        # 6. VERIFIED_FACT: Busca dinâmica no catálogo por referências de alta confiabilidade
        if self.catalog is not None and len(raw_text) > 10:
            words = [w for w in re.findall(r"[a-zA-Z0-9_\-\.]{3,}", raw_text) if len(w) > 3]
            if words:
                sample_query = " ".join(words[:4])
                matches = self.catalog.search_knn(sample_query, k=2)
                for doc, score in matches:
                    if doc.reliability_score >= 0.85 and doc.is_official and score > 0.45:
                        doc_content_low = doc.content.lower()
                        if any(w.lower() in doc_content_low for w in words[:3]):
                            return {
                                "tag": EpistemicTag.VERIFIED_FACT,
                                "confidence": round(float(doc.reliability_score), 2),
                                "basis": "MATCHED_CATALOG_HIGH_RELIABILITY",
                                "reasoning": f"Comprovado por referência documental oficial do catálogo ({doc.doc_id}) com confiabilidade {doc.reliability_score:.2f}.",
                            }

        # 7. VERIFIED_FACT: Ações e evidências factuais declaradas diretamente
        if _FACT_ACTION_MARKERS.search(raw_text):
            return {
                "tag": EpistemicTag.VERIFIED_FACT,
                "confidence": 0.95,
                "basis": "FACTUAL_EVIDENCE_IN_STATEMENT",
                "reasoning": "Afirmação factual baseada em evidência operacional ou saída direta documentada.",
            }

        # 8. TECHNICAL_INFERENCE: Correlação de telemetria ou dedução geral
        correlations = ctx.get("correlations") or ctx.get("telemetry_correlations")
        if correlations and len(correlations) > 0:
            return {
                "tag": EpistemicTag.TECHNICAL_INFERENCE,
                "confidence": 0.88,
                "basis": "TELEMETRY_CORRELATION",
                "reasoning": "Dedução baseada em correlação de telemetria e métricas operacionais.",
            }

        if _INFERENCE_MARKERS.search(raw_text):
            return {
                "tag": EpistemicTag.TECHNICAL_INFERENCE,
                "confidence": 0.85,
                "basis": "LOGICAL_MATHEMATICAL_DEDUCTION",
                "reasoning": "Dedução lógica/matemática inferida a partir de premissas ou estado de rede.",
            }

        # Padrão contextual numérico
        if re.search(r"\d+(?:\.\d+)?\s*(?:dBm|ms|pps|bps|Gbps|Mbps|%|hops|AS)", raw_text):
            return {
                "tag": EpistemicTag.TECHNICAL_INFERENCE,
                "confidence": 0.80,
                "basis": "NUMERICAL_ANALYSIS",
                "reasoning": "Inferência técnica derivada de cálculo ou medição quantitativa.",
            }

        # Fallback conservador: hipótese aberta
        return {
            "tag": EpistemicTag.HYPOTHESIS,
            "confidence": 0.50,
            "basis": "OPEN_PROPOSITION",
            "reasoning": "Proposição teórica pendente de correlação empírica ou comprovação por ferramenta.",
        }

    def categorize_epistemic_breakdown(
        self,
        statements: Sequence[Union[str, Dict[str, Any]]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """
        Divide uma lista de declarações/achados em três grupos epistêmicos:
        `verified_facts`, `technical_inferences`, e `hypotheses`.
        """
        verified_facts: List[str] = []
        technical_inferences: List[str] = []
        hypotheses: List[str] = []

        for item in statements:
            if isinstance(item, dict):
                text = item.get("text") or item.get("statement") or str(item)
                item_ctx = {**(context or {}), **item}
            else:
                text = str(item)
                item_ctx = context or {}

            classified = self.classify(text, context=item_ctx)
            tag = classified["tag"]

            if tag == EpistemicTag.VERIFIED_FACT:
                verified_facts.append(text)
            elif tag == EpistemicTag.TECHNICAL_INFERENCE:
                technical_inferences.append(text)
            else:
                hypotheses.append(text)

        return {
            "verified_facts": verified_facts,
            "technical_inferences": technical_inferences,
            "hypotheses": hypotheses,
        }


def classify_epistemic_statement(
    statement: str,
    context: Optional[Dict[str, Any]] = None,
    evidence: Optional[Any] = None,
    catalog: Optional[DynamicKnowledgeCatalog] = None,
) -> EpistemicTag:
    """Classifica rapidamente uma declaração retornando o EpistemicTag apropriado."""
    classifier = DynamicEpistemicClassifier(catalog=catalog)
    res = classifier.classify(statement, context=context, evidence=evidence)
    return res["tag"]
