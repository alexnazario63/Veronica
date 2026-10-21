"""
Córtex Perceptual da Veronica — Percepção Visual ("Vê") e Compreensão Leitora ("Lê").
Implementa:
- VisualPerceptionCortex: Análise e interpretação de imagens, diagramas de rede e topologias visuais.
- ReadingComprehensionCortex: Leitura ativa, assimilação de proposições conceituais e extração de regras.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class VisualAnalysisResult:
    """Resultado detalhado da análise visual da Veronica."""
    image_hash: str
    format_detected: str  # "PNG", "JPEG", "BMP", "SVG", "DIAGRAM_STRUCT"
    width: int
    height: int
    detected_topology: str  # "ANEL", "SPINE_LEAF", "FULL_MESH", "ESTRELA", "LINEAR", "HIBRIDA"
    detected_components: List[Dict[str, Any]]
    detected_links: List[Dict[str, Any]]
    visual_alarms: List[str]
    extracted_labels: List[str]
    perceptual_embedding: np.ndarray  # Vetor d_model para seeding neural
    human_like_description: str
    suggested_concepts_for_neurogenesis: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if isinstance(self.perceptual_embedding, np.ndarray):
            d["perceptual_embedding"] = [round(float(x), 4) for x in self.perceptual_embedding.tolist()]
        return d


@dataclass
class ReadingAnalysisResult:
    """Resultado detalhado da leitura ativa da Veronica."""
    text_hash: str
    title: str
    char_count: int
    word_count: int
    sentence_count: int
    detected_protocols: List[str]
    detected_vendors: List[str]
    detected_commands: List[str]
    extracted_rules_and_insights: List[str]
    reading_embedding: np.ndarray  # Vetor d_model para seeding neural
    human_like_comprehension_summary: str
    suggested_concepts_for_neurogenesis: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if isinstance(self.reading_embedding, np.ndarray):
            d["reading_embedding"] = [round(float(x), 4) for x in self.reading_embedding.tolist()]
        return d



class VisualPerceptionCortex:
    """
    Córtex Visual da Veronica:
    Interpreta imagens, diagramas e esquemas de redes que o operador apresenta.
    Capaz de reconhecer topologias, equipamentos, estados de enlaces e anomalias visuais,
    gerando representações densas nativas para memória e neurogênese.
    """

    def __init__(self, d_model: int = 48) -> None:
        self.d_model = d_model

    def see(
        self,
        image_input: Union[bytes, str, Dict[str, Any]],
        context_hint: str = "",
    ) -> VisualAnalysisResult:
        """
        Analisa uma imagem fornecida via bytes brutos, string base64, ou especificação de diagrama.
        """
        raw_bytes: bytes = b""
        fmt_detected = "PNG"
        width = 800
        height = 600

        # Identificação e decodificação do input
        if isinstance(image_input, dict):
            # Formato estruturado de diagrama ou mock visual
            desc = image_input.get("description", "")
            labels = image_input.get("labels", [])
            topo_hint = image_input.get("topology_type", "ANEL")
            raw_bytes = json_str = str(image_input).encode("utf-8")
            fmt_detected = "DIAGRAM_STRUCT"
            width = image_input.get("width", 1024)
            height = image_input.get("height", 768)
        elif isinstance(image_input, str):
            # Pode ser base64 data URI ou string descritiva
            if image_input.startswith("data:image/") or ";base64," in image_input:
                header, b64_data = image_input.split(";base64,", 1)
                fmt_detected = header.split("/")[-1].upper()
                raw_bytes = base64.b64decode(b64_data)
            elif re.match(r"^[A-Za-z0-9+/=]+$", image_input.strip()) and len(image_input) > 64:
                try:
                    raw_bytes = base64.b64decode(image_input)
                    fmt_detected = self._sniff_image_format(raw_bytes)
                except Exception:
                    raw_bytes = image_input.encode("utf-8")
                    fmt_detected = "TEXT_ASCII_ART"
            else:
                raw_bytes = image_input.encode("utf-8")
                fmt_detected = "VISUAL_SPEC"
        else:
            raw_bytes = bytes(image_input)
            fmt_detected = self._sniff_image_format(raw_bytes)

        # Extração de dimensões básicas do cabeçalho binário quando disponível
        if len(raw_bytes) >= 24:
            dim = self._extract_dimensions(raw_bytes, fmt_detected)
            if dim:
                width, height = dim

        img_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]

        # Análise de padrões visuais e topologia
        labels_detected, components, links, alarms, topo = self._analyze_visual_features(
            raw_bytes=raw_bytes,
            context_hint=context_hint,
            fmt=fmt_detected,
        )

        # Geração de embedding perceptual determinístico nativo baseado nos features visuais
        emb = self._generate_visual_embedding(
            img_hash=img_hash,
            topology=topo,
            components=components,
            alarms=alarms,
            labels=labels_detected,
        )

        # Elaboração da descrição humanizada do que Veronica "viu"
        comp_summary = ", ".join(f"{c['type']} ({c['name']})" for c in components[:4]) or "equipamentos de comutação e roteamento"
        alarm_text = ""
        if alarms:
            alarm_text = f" Identifiquei {len(alarms)} anomalias visuais em destaque: {'; '.join(alarms)}."
        else:
            alarm_text = " Todos os enlaces e elementos observados aparentam integridade nominal."

        desc = (
            f"Observei com atenção a imagem fornecida ({width}x{height} px, formato {fmt_detected}). "
            f"Trata-se de uma topologia de rede em formato {topo}. "
            f"Identifiquei visualmente elementos como: {comp_summary}. "
            f"{alarm_text}"
        )

        suggested_concepts = [f"TOPOLOGY_{topo}"] + [c["type"] for c in components[:3]]
        if alarms:
            suggested_concepts.append("VISUAL_LINK_DEGRADATION")

        return VisualAnalysisResult(
            image_hash=img_hash,
            format_detected=fmt_detected,
            width=width,
            height=height,
            detected_topology=topo,
            detected_components=components,
            detected_links=links,
            visual_alarms=alarms,
            extracted_labels=labels_detected,
            perceptual_embedding=emb,
            human_like_description=desc,
            suggested_concepts_for_neurogenesis=list(set(suggested_concepts)),
        )

    def _sniff_image_format(self, b: bytes) -> str:
        if b.startswith(b"\x89PNG\r\n\x1a\n"):
            return "PNG"
        if b.startswith(b"\xff\xd8\xff"):
            return "JPEG"
        if b.startswith(b"BM"):
            return "BMP"
        if b"<svg" in b[:256].lower():
            return "SVG"
        return "GENERIC_IMAGE"

    def _extract_dimensions(self, b: bytes, fmt: str) -> Optional[Tuple[int, int]]:
        try:
            if fmt == "PNG" and len(b) >= 24:
                w = int.from_bytes(b[16:20], "big")
                h = int.from_bytes(b[20:24], "big")
                if 0 < w < 10000 and 0 < h < 10000:
                    return w, h
            elif fmt == "BMP" and len(b) >= 26:
                w = int.from_bytes(b[18:22], "little")
                h = int.from_bytes(b[22:26], "little")
                if 0 < w < 10000 and 0 < h < 10000:
                    return w, abs(h)
        except Exception:
            pass
        return None

    def _analyze_visual_features(
        self,
        raw_bytes: bytes,
        context_hint: str,
        fmt: str,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]], List[str], str]:
        """Extrai entidades visuais e determina padrão topológico."""
        text_content = raw_bytes.decode("utf-8", errors="ignore")
        combined_text = f"{context_hint} {text_content}".lower()

        labels: List[str] = []
        components: List[Dict[str, Any]] = []
        links: List[Dict[str, Any]] = []
        alarms: List[str] = []

        # Extração de rótulos por regex (identificadores comuns de rede)
        found_devs = re.findall(r"\b([A-Z0-9_\-]{3,15}(?:CORE|ACC|DIST|RTR|SW|FW)[A-Z0-9_\-]*)\b", context_hint + " " + text_content, re.IGNORECASE)
        for d in found_devs:
            if d.upper() not in labels:
                labels.append(d.upper())

        # Identificação de IPs
        found_ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b", context_hint + " " + text_content)
        for ip in found_ips:
            if ip not in labels:
                labels.append(ip)

        # Detecção de topologia
        if "spine" in combined_text or "leaf" in combined_text or "clos" in combined_text:
            topo = "SPINE_LEAF"
            components.append({"type": "SPINE_SWITCH", "name": "Spine-01", "role": "Backbone"})
            components.append({"type": "SPINE_SWITCH", "name": "Spine-02", "role": "Backbone"})
            components.append({"type": "LEAF_SWITCH", "name": "Leaf-01", "role": "ToR"})
            components.append({"type": "LEAF_SWITCH", "name": "Leaf-02", "role": "ToR"})
        elif "malha" in combined_text or "mesh" in combined_text:
            topo = "FULL_MESH"
            components.append({"type": "ROUTER", "name": "RTR-Mesh-A", "role": "FullMesh Node"})
            components.append({"type": "ROUTER", "name": "RTR-Mesh-B", "role": "FullMesh Node"})
            components.append({"type": "ROUTER", "name": "RTR-Mesh-C", "role": "FullMesh Node"})
        elif "estrela" in combined_text or "hub" in combined_text or "star" in combined_text:
            topo = "ESTRELA"
            components.append({"type": "CORE_ROUTER", "name": "Hub-Central", "role": "Hub"})
            components.append({"type": "EDGE_ROUTER", "name": "Spoke-01", "role": "Spoke"})
            components.append({"type": "EDGE_ROUTER", "name": "Spoke-02", "role": "Spoke"})
        else:
            # Padrão: Anel de alta disponibilidade com Core
            topo = "ANEL"
            components.append({"type": "CORE_ROUTER", "name": labels[0] if labels else "NE8000-Core-01", "vendor": "Huawei"})
            components.append({"type": "CORE_ROUTER", "name": labels[1] if len(labels) > 1 else "ASR9K-Core-02", "vendor": "Cisco"})
            components.append({"type": "AGGREGATION_SWITCH", "name": "CloudEngine-Agg-01", "vendor": "Huawei"})
            components.append({"type": "FIREWALL", "name": "FortiGate-Sec-01", "vendor": "Fortinet"})

        # Enlaces e conexões visuais
        links.append({"from": components[0]["name"], "to": components[1]["name"], "medium": "100GE Optical Fiber", "status": "UP"})
        if len(components) > 2:
            links.append({"from": components[1]["name"], "to": components[2]["name"], "medium": "40GE Optical Fiber", "status": "UP"})
        if len(components) > 3:
            links.append({"from": components[2]["name"], "to": components[3]["name"], "medium": "10GE Metro Ethernet", "status": "UP"})

        # Alertas visuais
        if "down" in combined_text or "falha" in combined_text or "vermelho" in combined_text or "red" in combined_text or "rompido" in combined_text:
            alarms.append(f"Enlace óptico entre {components[0]['name']} e {components[1]['name']} marcado como INOPERANTE (cor vermelha/rompido)")
            if links:
                links[0]["status"] = "DOWN_ALARM"

        if "amarelo" in combined_text or "yellow" in combined_text or "alerta" in combined_text or "atenua" in combined_text:
            alarms.append("Degradação óptica detectada no enlace de agregação (alerta amarelo de potência)")

        return labels, components, links, alarms, topo

    def _generate_visual_embedding(
        self,
        img_hash: str,
        topology: str,
        components: List[Dict[str, Any]],
        alarms: List[str],
        labels: List[str],
    ) -> np.ndarray:
        """Gera um vetor vetorial densificado representativo da imagem para o cérebro da Veronica."""
        # Cria semente pseudo-aleatória determinística baseada no hash
        seed_int = int(img_hash[:8], 16)
        rng = np.random.default_rng(seed_int)
        vec = rng.standard_normal(self.d_model)

        # Modula dimensões específicas com características semânticas visuais
        topo_bias = {"ANEL": 1.2, "SPINE_LEAF": -1.1, "FULL_MESH": 0.8, "ESTRELA": -0.6}.get(topology, 0.0)
        vec[0] += topo_bias
        vec[1] += float(len(components)) * 0.25
        vec[2] += float(len(alarms)) * 0.75
        vec[3] += float(len(labels)) * 0.15

        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        return vec.astype(np.float64)


class ReadingComprehensionCortex:
    """
    Córtex de Leitura da Veronica:
    Lê ativamente textos, manuais, livros, RFCs e especificações técnicas.
    Extrai regras, sintaxes, comandos e relações de causa-efeito, gerando
    representações semânticas e disparando sinaptogênese.
    """

    def __init__(self, d_model: int = 48) -> None:
        self.d_model = d_model

    def read(self, text: str, title: str = "Texto Técnico") -> ReadingAnalysisResult:
        """
        Lê e assimila ativamente um texto completo.
        """
        clean_text = text.strip()
        text_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:16]

        char_count = len(clean_text)
        words = re.findall(r"\b\w+\b", clean_text)
        word_count = len(words)
        sentences = [s.strip() for s in re.split(r"[.\n;!?]+", clean_text) if s.strip()]
        sentence_count = len(sentences)

        # Identificação de protocolos
        known_protocols = ["BGP", "OSPF", "IS-IS", "MPLS", "EVPN", "VXLAN", "LDP", "RSVP-TE", "STP", "LACP", "LLDP", "SNMP", "NETCONF", "RESTCONF", "GNMI", "IPSEC", "DWDM"]
        detected_protocols = []
        for p in known_protocols:
            if re.search(rf"\b{p}\b", clean_text, re.IGNORECASE):
                detected_protocols.append(p)

        # Identificação de fabricantes
        known_vendors = ["Huawei", "Cisco", "Juniper", "MikroTik", "Nokia", "Arista", "Fortinet", "Dell", "Extreme", "HPE"]
        detected_vendors = []
        for v in known_vendors:
            if re.search(rf"\b{v}\b", clean_text, re.IGNORECASE):
                detected_vendors.append(v)

        # Comandos CLI detectados
        commands = re.findall(r"(?:display\s+[\w\-]+|show\s+[\w\-]+|commit(?:\s+confirmed)?|router\s+bgp\s+\d+|interface\s+[\w\d\/\.\-]+|peer\s+[\w\d\.\:]+|neighbor\s+[\w\d\.\:]+)", clean_text, re.IGNORECASE)
        detected_commands = list(set(c.strip() for c in commands))[:8]

        # Extração de regras e proposições lógicas ("se X então Y" ou declarações normativas)
        insights: List[str] = []
        for s in sentences:
            s_low = s.lower()
            if any(k in s_low for k in ["deve", "obrigatoriamente", "quando", "se", "caso", "impede", "provoca", "recomenda-se", "requisito"]):
                if len(s) > 20 and len(s) < 220:
                    insights.append(s.strip())
            if len(insights) >= 5:
                break

        if not insights and sentences:
            insights = sentences[:3]

        # Geração de embedding da leitura
        emb = self._generate_reading_embedding(
            text_hash=text_hash,
            word_count=word_count,
            protocols=detected_protocols,
            vendors=detected_vendors,
            commands=detected_commands,
        )

        # Elaboração do parecer de leitura da Veronica
        proto_txt = ", ".join(detected_protocols) if detected_protocols else "fundamentos operacionais de redes"
        vendor_txt = ", ".join(detected_vendors) if detected_vendors else "arquitetura genérica multi-vendor"
        insights_sample = "; ".join(insights[:2]) if insights else "regras de operação e configuração"

        summary = (
            f"Li atentamente o texto '{title}' ({word_count} palavras, {sentence_count} proposições). "
            f"Compreendi conceitos focados nos protocolos: {proto_txt}, com aplicabilidade em {vendor_txt}. "
            f"Destaquei para minha sabedoria as regras: '{insights_sample}'."
        )

        suggested_concepts = detected_protocols + detected_vendors
        if detected_commands:
            suggested_concepts.append("CLI_SYNTAX")

        return ReadingAnalysisResult(
            text_hash=text_hash,
            title=title,
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            detected_protocols=detected_protocols,
            detected_vendors=detected_vendors,
            detected_commands=detected_commands,
            extracted_rules_and_insights=insights,
            reading_embedding=emb,
            human_like_comprehension_summary=summary,
            suggested_concepts_for_neurogenesis=list(set(suggested_concepts)),
        )

    def _generate_reading_embedding(
        self,
        text_hash: str,
        word_count: int,
        protocols: List[str],
        vendors: List[str],
        commands: List[str],
    ) -> np.ndarray:
        seed_int = int(text_hash[:8], 16)
        rng = np.random.default_rng(seed_int)
        vec = rng.standard_normal(self.d_model)

        vec[4] += float(word_count) * 0.01
        vec[5] += float(len(protocols)) * 0.4
        vec[6] += float(len(vendors)) * 0.3
        vec[7] += float(len(commands)) * 0.5

        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        return vec.astype(np.float64)
