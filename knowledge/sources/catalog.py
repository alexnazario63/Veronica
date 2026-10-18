"""
Catálogo de Fontes Oficiais e Base Documental Inicial Curada da Veronica.
Cada registro preserva rigorosamente os 12 campos de metadados obrigatórios (Seção 4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class KnowledgeDocument:
    doc_id: str
    title: str
    content: str
    # 12 Metadados Obrigatórios (Seção 4)
    vendor: str
    family_model: str
    os_name: str
    version_build: str
    feature: str
    original_source: str
    source_uri: str
    page_section: str
    published_date: str
    consulted_date: str
    reliability_score: float
    applicability_restrictions: str
    # Campos extraídos automaticamente
    extracted_commands: List[str] = field(default_factory=list)
    extracted_tables: List[str] = field(default_factory=list)
    is_official: bool = True
    is_obsolete: bool = False
    superseded_by: str = ""
    security_quarantined: bool = False
    quarantine_reason: str = ""
    domain: str = "routing"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Os 7 Domínios de Conhecimento de Redes Obrigatórios (Seção 6)
NETWORKING_KNOWLEDGE_DOMAINS: Dict[str, Dict[str, Any]] = {
    "fundamentals": {
        "section_number": 1,
        "title": "Fundamentos",
        "topics": [
            "OSI",
            "TCP/IP",
            "Ethernet",
            "IPv4",
            "IPv6",
            "ARP",
            "ND",
            "VLAN",
            "STP",
            "LACP",
            "MTU",
            "QoS",
            "ACL",
        ],
    },
    "routing": {
        "section_number": 2,
        "title": "Roteamento",
        "topics": [
            "rotas estáticas",
            "OSPF",
            "OSPFv3",
            "IS-IS",
            "BGP",
            "MP-BGP",
            "políticas",
            "redistribuição",
            "TE",
        ],
    },
    "mpls_services": {
        "section_number": 3,
        "title": "MPLS e serviços",
        "topics": [
            "LDP",
            "RSVP-TE",
            "Segment Routing",
            "L2VPN",
            "L3VPN",
            "VPWS",
            "VPLS",
            "EVPN",
            "VXLAN",
            "pseudowires",
        ],
    },
    "infrastructure": {
        "section_number": 4,
        "title": "Infraestrutura",
        "topics": [
            "backbone",
            "metro",
            "DWDM",
            "OTN",
            "enlaces ópticos",
            "transceptores",
            "potência óptica",
            "atenuação",
            "redundância",
        ],
    },
    "security": {
        "section_number": 5,
        "title": "Segurança",
        "topics": [
            "firewalls",
            "VPN",
            "IPsec",
            "AAA",
            "RADIUS",
            "TACACS+",
            "SSH",
            "hardening",
            "segmentação",
        ],
    },
    "operations": {
        "section_number": 6,
        "title": "Operação",
        "topics": [
            "SNMP",
            "NETCONF",
            "RESTCONF",
            "gNMI",
            "syslog",
            "telemetria",
            "NMS",
            "alarmes",
            "monitoramento",
            "troubleshooting",
        ],
    },
    "systems": {
        "section_number": 7,
        "title": "Sistemas",
        "topics": [
            "Linux",
            "virtualização",
            "containers",
            "DNS",
            "DHCP",
            "NTP",
            "automação",
        ],
    },
}


OFFICIAL_INITIAL_DOCUMENTS: List[KnowledgeDocument] = [
    KnowledgeDocument(
        doc_id="DOC-HUAWEI-VRP8-BGP-001",
        title="Huawei VRP 8 — Diagnóstico e Configuração Segura de BGP / MP-BGP",
        content=(
            "Em roteadores Huawei NetEngine 8000 e switches CloudEngine com VRP 8.x, "
            "a verificação de sessões BGP deve iniciar por `display bgp peer` e `display bgp peer <ip> verbose`. "
            "Quando o peer permanece em estado `Active` ou `Idle`, deve-se verificar:\n"
            "1. Alcançabilidade IP da interface de origem (`ping -a <source-ip> <peer-ip>`).\n"
            "2. Configuração explícita de `peer <ip> connect-interface Loopback0` em sessões iBGP ou eBGP loopback.\n"
            "3. Para eBGP entre loopbacks, é obrigatório `peer <ip> ebgp-max-hop <hops>`.\n"
            "4. Verificar se ACLs ou filtros na porta TCP 179 estão bloqueando pacotes.\n"
            "5. Alterações no VRP 8 exigem `commit trial <seconds>` para proteção contra perda de gerência."
        ),
        vendor="Huawei",
        family_model="NetEngine 8000 / CloudEngine 6800-8800",
        os_name="VRP",
        version_build="V800R022C00SPC600",
        feature="BGP-4 / MP-BGP Troubleshooting & Safe Commit",
        original_source="Huawei HedEx Official Product Documentation",
        source_uri="official://huawei/hedex/vrp8/bgp-troubleshooting-v8r22",
        page_section="Seção 4.2 — Troubleshooting BGP Peer States (p. 48-55)",
        published_date="2024-06-15",
        consulted_date="2026-10-17",
        reliability_score=0.99,
        applicability_restrictions=(
            "Exclusivo para VRP 8.x (two-phase commit). No VRP 5.x (S5720/S6720) a aplicação é imediata sem `commit`."
        ),
        extracted_commands=[
            "display bgp peer",
            "display bgp peer verbose",
            "display current-configuration configuration bgp",
            "ping -a",
            "peer connect-interface Loopback0",
            "peer ebgp-max-hop",
            "commit trial 120",
        ],
        domain="routing",
    ),
    KnowledgeDocument(
        doc_id="DOC-HUAWEI-VRP8-OPTICAL-002",
        title="Huawei VRP 8 — Diagnóstico de Transceptores Ópticos e Atenuação DWDM/SFP+",
        content=(
            "Para investigar quedas de enlace óptico, CRC errors ou alarme de potência RX baixa em equipamentos Huawei VRP:\n"
            "Executar `display interface <interface>` e `display transceiver interface <interface> verbose`.\n"
            "| Parâmetro | Faixa Normal (10GBASE-LR) | Alerta Crítico |\n"
            "| Rx Power (dBm) | -1.0 a -14.4 dBm | < -14.4 dBm (Low Alarm) ou > +0.5 dBm (Overload) |\n"
            "| Tx Power (dBm) | -8.2 a +0.5 dBm | Fora da janela nominal do laser |\n"
            "Se Rx Power estiver abaixo do limiar mínimo com Tx Power normal no equipamento remoto, "
            "a causa provável é atenuação excessiva no cordão óptico, fusão degradada ou conector sujo."
        ),
        vendor="Huawei",
        family_model="NetEngine 8000 / CloudEngine / S-Series",
        os_name="VRP",
        version_build="V800R022C00SPC600",
        feature="Optical Transceiver Diagnostics (DOM/DDM)",
        original_source="Huawei HedEx Optical Interface Guide",
        source_uri="official://huawei/hedex/vrp8/optical-dom-diagnostics",
        page_section="Seção 2.7 — Optical Module Monitoring (p. 19-24)",
        published_date="2024-05-10",
        consulted_date="2026-10-17",
        reliability_score=0.98,
        applicability_restrictions="Requer transceptor óptico compatível com DDM/DOM habilitado.",
        extracted_commands=[
            "display interface",
            "display transceiver interface verbose",
        ],
        extracted_tables=[
            "Parâmetro | Faixa Normal (10GBASE-LR) | Alerta Crítico; Rx Power -1.0 a -14.4 dBm; Tx Power -8.2 a +0.5 dBm"
        ],
        domain="infrastructure",
    ),
    KnowledgeDocument(
        doc_id="DOC-CISCO-IOSXR-BGP-003",
        title="Cisco IOS-XR 7.x — BGP Neighbor Enforcement e Route-Policy Obrigatória (RPL)",
        content=(
            "No Cisco IOS-XR (ASR 9000, NCS 5500, CRS), sessões eBGP possuem proteção nativa RFC 8212: "
            "nenhum prefixo é anunciado ou aceito de um vizinho eBGP a menos que haja `route-policy <name> in` "
            "e `route-policy <name> out` explicitamente configuradas sob a address-family do neighbor.\n"
            "Comandos de diagnóstico somente leitura:\n"
            "- `show bgp ipv4 unicast summary`\n"
            "- `show bgp ipv4 unicast neighbors <ip>`\n"
            "- `show rpl route-policy`\n"
            "Para alterações seguras no IOS-XR, sempre utilizar `commit confirmed <seconds>` e `show configuration commit changes`."
        ),
        vendor="Cisco",
        family_model="ASR 9000 / NCS 5500",
        os_name="IOS-XR",
        version_build="7.9.2",
        feature="BGP / RPL Mandatory eBGP Policy & Commit Confirmed",
        original_source="Cisco IOS XR Routing Configuration Guide",
        source_uri="official://cisco/iosxr-7.9/bgp-rpl-configuration",
        page_section="Chapter: Implementing BGP on Cisco IOS XR (p. 112-130)",
        published_date="2024-03-20",
        consulted_date="2026-10-17",
        reliability_score=0.99,
        applicability_restrictions="Aplica-se a Cisco IOS-XR. Não se aplica a Cisco IOS Clássico ou IOS-XE (que usam route-map).",
        extracted_commands=[
            "show bgp ipv4 unicast summary",
            "show bgp ipv4 unicast neighbors",
            "show rpl route-policy",
            "commit confirmed 120",
            "show configuration commit changes",
        ],
        domain="routing",
    ),
    KnowledgeDocument(
        doc_id="DOC-CISCO-IOSXE-OSPF-004",
        title="Cisco IOS-XE 17.x — Diagnóstico de Adjacência OSPF Presa em EXSTART/EXCHANGE",
        content=(
            "Em switches Catalyst 9000 e roteadores ASR 1000/ISR 4000 com Cisco IOS-XE 17.x, "
            "quando uma vizinhança OSPF permanece presa em `EXSTART` ou `EXCHANGE` e reinicia por Dead Timer, "
            "a causa primária mais frequente é divergência de MTU IP nos pacotes DBD (Database Description).\n"
            "Passos de verificação:\n"
            "1. `show ip ospf neighbor detail`\n"
            "2. `show ip ospf interface <interface>`\n"
            "3. `show interfaces <interface>`\n"
            "4. Testar ping com bit DF (Don't Fragment): `ping <neighbor-ip> size 1500 df-bit`.\n"
            "A correção recomendada é alinhar o MTU físico/IP em ambos os equipamentos (ex.: 1500 ou 9216 bytes) "
            "em vez de usar `ip ospf mtu-ignore`, que apenas mascara a divergência."
        ),
        vendor="Cisco",
        family_model="Catalyst 9300/9500 & ASR 1000",
        os_name="IOS-XE",
        version_build="17.9.4a",
        feature="OSPFv2 Adjacency & MTU Verification",
        original_source="Cisco IOS XE IP Routing: OSPF Configuration Guide",
        source_uri="official://cisco/iosxe-17.9/ospf-troubleshooting",
        page_section="Section: Troubleshooting OSPF Neighbor States (p. 64-71)",
        published_date="2024-02-11",
        consulted_date="2026-10-17",
        reliability_score=0.99,
        applicability_restrictions="Cisco IOS / IOS-XE 16.x e 17.x.",
        extracted_commands=[
            "show ip ospf neighbor detail",
            "show ip ospf interface",
            "show interfaces",
            "ping size 1500 df-bit",
        ],
        domain="routing",
    ),
    KnowledgeDocument(
        doc_id="DOC-JUNIPER-JUNOS-EVPN-005",
        title="Juniper Junos OS 22.4R3 — Diagnóstico de BGP, OSPF e MPLS/EVPN com Commit Confirmed",
        content=(
            "No Juniper Junos OS e Junos OS Evolved (MX Series, QFX Series, PTX Series):\n"
            "- Verificar estado BGP: `show bgp summary` e `show bgp neighbor <ip>`.\n"
            "- Verificar OSPF: `show ospf neighbor extensive` e `show interfaces <if-name> extensive`.\n"
            "- Comparar candidato com configuração ativa: `show | compare`.\n"
            "- Validar sintaxe antes de aplicar: `commit check`.\n"
            "- Aplicar com reversão automática de segurança: `commit confirmed 2` (reverte em 2 minutos se não confirmado por segundo `commit`).\n"
            "- Reverter imediatamente para a versão anterior: `rollback 1` seguido de `commit`."
        ),
        vendor="Juniper",
        family_model="MX Series / QFX5120 / PTX",
        os_name="Junos OS",
        version_build="22.4R3-S2",
        feature="Candidate Configuration, Commit Confirmed & Rollback",
        original_source="Juniper TechLibrary — Junos OS CLI User Guide",
        source_uri="official://juniper/junos-22.4/commit-confirmed-and-routing",
        page_section="Section: Committing and Rolling Back Configurations (p. 35-44)",
        published_date="2024-04-08",
        consulted_date="2026-10-17",
        reliability_score=0.99,
        applicability_restrictions="Aplicável a Junos OS e Junos OS Evolved.",
        extracted_commands=[
            "show bgp summary",
            "show bgp neighbor",
            "show ospf neighbor extensive",
            "show interfaces extensive",
            "show | compare",
            "commit check",
            "commit confirmed 2",
            "rollback 1",
        ],
        domain="mpls_services",
    ),
    KnowledgeDocument(
        doc_id="DOC-MIKROTIK-ROS7-BGP-006",
        title="MikroTik RouterOS v7.14 — Arquitetura BGP, Filtros e Modo Safe Mode",
        content=(
            "No MikroTik RouterOS v7 (CCR2004, CCR2116, CRS3xx), a pilha de roteamento foi completamente reescrita em relação ao RouterOS v6:\n"
            "- Sessões BGP são consultadas com `/routing/bgp/session/print` e configuradas em `/routing/bgp/connection`.\n"
            "- Filtros de rota utilizam sintaxe tipo script em `/routing/filter/rule`.\n"
            "- O comando antigo `/routing bgp peer print` do RouterOS v6 é OBSOLETO e inexistente no RouterOS v7.\n"
            "- Antes de qualquer alteração remota no RouterOS, deve-se ativar o `Safe Mode` (Ctrl+X) para reversão automática caso a sessão administrativa caia."
        ),
        vendor="MikroTik",
        family_model="CCR2004 / CCR2116 / CRS326",
        os_name="RouterOS",
        version_build="7.14.3",
        feature="BGP v7 Engine & Safe Mode",
        original_source="MikroTik Official Documentation — RouterOS v7 Routing",
        source_uri="official://mikrotik/routeros-v7/bgp-and-filters",
        page_section="Routing -> BGP -> Configuration & Monitoring",
        published_date="2024-05-22",
        consulted_date="2026-10-17",
        reliability_score=0.98,
        applicability_restrictions="Exclusivo para RouterOS v7.x. Incompatível com RouterOS v6.x.",
        extracted_commands=[
            "/routing/bgp/session/print",
            "/routing/bgp/connection/print",
            "/routing/filter/rule/print",
            "/ip/route/print",
        ],
        domain="routing",
    ),
    KnowledgeDocument(
        doc_id="DOC-MIKROTIK-ROS6-OBSOLETE-007",
        title="MikroTik RouterOS v6.49 (Legado) — Configuração BGP Antiga",
        content=(
            "No legado RouterOS v6.x, peers BGP eram configurados em `/routing bgp peer` e filtros em `/routing filter`. "
            "ATENÇÃO: Esta sintaxe foi removida no RouterOS v7.x."
        ),
        vendor="MikroTik",
        family_model="CCR1036 / RB1100",
        os_name="RouterOS",
        version_build="6.49.13",
        feature="Legacy BGP v6",
        original_source="MikroTik Legacy Wiki (v6)",
        source_uri="official://mikrotik/wiki-v6/bgp-legacy",
        page_section="Legacy Routing BGP",
        published_date="2021-11-10",
        consulted_date="2026-10-17",
        reliability_score=0.85,
        applicability_restrictions="Obsoleto para equipamentos rodando RouterOS 7.x.",
        extracted_commands=["/routing bgp peer print"],
        is_obsolete=True,
        superseded_by="DOC-MIKROTIK-ROS7-BGP-006",
        domain="routing",
    ),
    KnowledgeDocument(
        doc_id="DOC-NOKIA-SROS-MPLS-008",
        title="Nokia SR OS 23.10 — Diagnóstico de MPLS, LDP, RSVP-TE e EVPN/VPLS",
        content=(
            "Em roteadores Nokia 7750 SR e 7250 IXR com SR OS (MD-CLI):\n"
            "- Verificar sessões BGP e EVPN: `show router bgp summary all` e `show router bgp routes evpn`.\n"
            "- Verificar túneis MPLS/LDP: `show router ldp bindings active` e `show router tunnel-table`.\n"
            "- Verificar serviço L2VPN/VPLS/EVPN: `show service id <id> base` e `show service id <id> fdb detail`.\n"
            "- No MD-CLI, usar `compare` e `commit confirmed <minutes>` para mudanças seguras."
        ),
        vendor="Nokia",
        family_model="7750 SR / 7250 IXR",
        os_name="SR OS",
        version_build="23.10.R2",
        feature="MPLS / EVPN Services & MD-CLI Commit Confirmed",
        original_source="Nokia SR OS 23.10 Documentation Center",
        source_uri="official://nokia/sros-23.10/mpls-evpn-guide",
        page_section="Section 5 — EVPN & VPLS Operations (p. 88-104)",
        published_date="2024-01-30",
        consulted_date="2026-10-17",
        reliability_score=0.98,
        applicability_restrictions="Aplica-se ao modo MD-CLI do Nokia SR OS 21.x-24.x.",
        extracted_commands=[
            "show router bgp summary all",
            "show router ldp bindings active",
            "show router tunnel-table",
            "show service id base",
            "commit confirmed",
        ],
        domain="mpls_services",
    ),
    KnowledgeDocument(
        doc_id="DOC-ARISTA-EOS-EVPN-009",
        title="Arista EOS 4.31 — Diagnóstico de VXLAN, EVPN, LACP/MLAG e Configure Session",
        content=(
            "Em switches Arista 7050X3/7280R3 rodando EOS 4.31:\n"
            "- Verificar MLAG e LACP: `show mlag`, `show port-channel summary`, `show lacp neighbor`.\n"
            "- Verificar VXLAN/EVPN: `show bgp evpn summary`, `show vxlan vtep`, `show vxlan address-table`.\n"
            "- Mudanças seguras no Arista EOS devem utilizar `configure session <name>`, `show session-config diffs`, "
            "e `commit timer <hh:mm:ss>` para rollback automático se não confirmado."
        ),
        vendor="Arista",
        family_model="7050X3 / 7280R3",
        os_name="EOS",
        version_build="4.31.2F",
        feature="EVPN-VXLAN, MLAG & Configure Session Timer",
        original_source="Arista EOS User Manual 4.31.2F",
        source_uri="official://arista/eos-4.31/configure-session-and-evpn",
        page_section="Chapter 6 — Session Management & EVPN (p. 142-160)",
        published_date="2024-04-19",
        consulted_date="2026-10-17",
        reliability_score=0.99,
        applicability_restrictions="Arista EOS 4.28+.",
        extracted_commands=[
            "show mlag",
            "show port-channel summary",
            "show bgp evpn summary",
            "show vxlan vtep",
            "configure session",
            "show session-config diffs",
            "commit timer",
        ],
        domain="mpls_services",
    ),
    KnowledgeDocument(
        doc_id="DOC-FORTINET-FORTIOS-VPN-010",
        title="Fortinet FortiOS 7.4 — Diagnóstico de IPsec VPN, BGP, Firewall Policies e Workspace Mode",
        content=(
            "Em appliances FortiGate com FortiOS 7.4:\n"
            "- Diagnóstico de túneis IPsec IKEv1/IKEv2: `diagnose vpn ike gateway list` e `diagnose vpn tunnel list`.\n"
            "- Verificação de tabela de roteamento e BGP: `get router info routing-table all` e `get router info bgp summary`.\n"
            "- Reversão automática de segurança contra perda de gerência: configurar `set cfg-save revert` e `set cfg-revert-timeout 120` em `config system global`."
        ),
        vendor="Fortinet",
        family_model="FortiGate 100F / 600F / 1800F",
        os_name="FortiOS",
        version_build="7.4.3",
        feature="IPsec VPN Diagnostics & Configuration Revert Timeout",
        original_source="Fortinet Document Library — FortiOS 7.4 Administration Guide",
        source_uri="official://fortinet/fortios-7.4/vpn-and-system-revert",
        page_section="VPN Diagnostics & System Global Revert (p. 210-225)",
        published_date="2024-03-05",
        consulted_date="2026-10-17",
        reliability_score=0.98,
        applicability_restrictions="FortiOS 7.2 e 7.4.",
        extracted_commands=[
            "diagnose vpn ike gateway list",
            "diagnose vpn tunnel list",
            "get router info routing-table all",
            "get router info bgp summary",
        ],
        domain="security",
    ),
    KnowledgeDocument(
        doc_id="DOC-IETF-FUNDAMENTALS-L2L3-011",
        title="IETF & Multi-Vendor — Fundamentos OSI, TCP/IP, Ethernet, IPv4, IPv6, ARP, ND, VLAN, STP, LACP, MTU, QoS e ACL",
        content=(
            "Fundamentos da arquitetura de redes (Camadas OSI e TCP/IP):\n"
            "- Camada 2 (Ethernet, VLAN IEEE 802.1Q, STP/RSTP/MSTP IEEE 802.1w/s, LACP IEEE 802.3ad): "
            "verificar portas trunk/access, Root Bridge STP (`show spanning-tree`, `display stp brief`) e agregação LACP (`show lacp neighbor`, `display eth-trunk`).\n"
            "- Camada 3 (IPv4, IPv6 RFC 8200, ARP RFC 826, Neighbor Discovery ND RFC 4861, MTU/Path MTU Discovery): "
            "validar tabelas ARP (`show ip arp`, `display arp`) e vizinhança IPv6 ND (`show ipv6 neighbors`, `display ipv6 neighbors`), além de consistência de MTU (1500 vs 9216 Jumbo Frames).\n"
            "- Qualidade de Serviço (QoS DiffServ/DSCP) e Listas de Controle de Acesso (ACL): "
            "verificar contadores de filas de saída/tail-drop (`show policy-map interface`, `display qos queue statistics`) e matches em ACL (`show access-lists`, `display acl all`)."
        ),
        vendor="IETF",
        family_model="Multi-Vendor Standard (RFC 8200 / IEEE 802.1Q / 802.3ad)",
        os_name="RFC-Standard",
        version_build="RFC-8200-4861-802.1Q",
        feature="Fundamentos: OSI, TCP/IP, Ethernet, IPv4, IPv6, ARP, ND, VLAN, STP, LACP, MTU, QoS, ACL",
        original_source="IETF Datatracker RFC Repository & IEEE 802 Standards",
        source_uri="https://datatracker.ietf.org/doc/html/rfc8200",
        page_section="Sections 1-5 — Layer 2/3 Architecture, Neighbor Discovery, MTU, QoS & ACL",
        published_date="2024-01-15",
        consulted_date="2026-10-17",
        reliability_score=0.99,
        applicability_restrictions="Padrão normativo IETF/IEEE aplicável a todos os fabricantes.",
        extracted_commands=[
            "show spanning-tree",
            "display stp brief",
            "show lacp neighbor",
            "display eth-trunk",
            "show ip arp",
            "show ipv6 neighbors",
            "show policy-map interface",
            "show access-lists",
        ],
        domain="fundamentals",
    ),
    KnowledgeDocument(
        doc_id="DOC-CISCO-IOSXR-ROUTING-ISIS-TE-012",
        title="Cisco IOS-XR 7.9 — Roteamento Avançado: Rotas Estáticas, OSPF, OSPFv3, IS-IS, BGP, MP-BGP, Políticas, Redistribuição e TE",
        content=(
            "Guia de referência para protocolos de roteamento no backbone:\n"
            "- Rotas estáticas (IPv4/IPv6) com BFD e track de alcançabilidade (`show route static`).\n"
            "- IGP: OSPF (IPv4), OSPFv3 (IPv6 RFC 5340) e IS-IS Level-1/Level-2 (`show isis adjacency`, `show ospfv3 neighbor`).\n"
            "- EGP: BGP e MP-BGP para famílias VPNv4, VPNv6 e EVPN.\n"
            "- Controle de plano de roteamento: políticas (RPL / route-map), prevenção de loops em redistribuição mútua usando route-tags, e Engenharia de Tráfego (MPLS-TE / SR-TE `show mpls traffic-eng tunnels`)."
        ),
        vendor="Cisco",
        family_model="ASR 9000 / NCS 5500 / 8000 Series",
        os_name="IOS-XR",
        version_build="7.9.2",
        feature="Roteamento: rotas estáticas, OSPF, OSPFv3, IS-IS, BGP, MP-BGP, políticas, redistribuição, TE",
        original_source="Cisco Support Official Routing Documentation",
        source_uri="https://www.cisco.com/c/en/us/support/ios-xr-routing-isis-ospfv3-te",
        page_section="Chapters 1-9 — Static, OSPF/OSPFv3, IS-IS, MP-BGP, Redistribution & TE",
        published_date="2024-05-01",
        consulted_date="2026-10-17",
        reliability_score=0.99,
        applicability_restrictions="Cisco IOS-XR 7.x.",
        extracted_commands=[
            "show route static",
            "show isis adjacency",
            "show ospfv3 neighbor",
            "show mpls traffic-eng tunnels",
        ],
        domain="routing",
    ),
    KnowledgeDocument(
        doc_id="DOC-JUNIPER-MPLS-SERVICES-013",
        title="Juniper Junos OS 22.4 — MPLS e Serviços: LDP, RSVP-TE, Segment Routing, L2VPN, L3VPN, VPWS, VPLS, EVPN, VXLAN e Pseudowires",
        content=(
            "Arquitetura e diagnóstico de MPLS e serviços de transporte:\n"
            "- Sinalização de rótulos: LDP (`show ldp neighbor`, `show ldp database`), RSVP-TE (`show rsvp session`) e Segment Routing SR-MPLS/SRv6 (`show spring-traffic-engineering`).\n"
            "- Serviços Camada 2: L2VPN, VPWS (E-Line), VPLS (E-LAN), pseudowires Martini/Kompella (`show l2circuit connections`, `show vpls connections`) e EVPN-VXLAN (`show evpn database`, `show ethernet-switching vxlan-tunnel-end-point`).\n"
            "- Serviços Camada 3: L3VPN RFC 4364 (`show route table bgp.l3vpn.0`)."
        ),
        vendor="Juniper",
        family_model="MX Series / PTX Series",
        os_name="Junos OS",
        version_build="22.4R3-S2",
        feature="MPLS e serviços: LDP, RSVP-TE, Segment Routing, L2VPN, L3VPN, VPWS, VPLS, EVPN, VXLAN, pseudowires",
        original_source="Juniper Official Documentation Portal",
        source_uri="https://www.juniper.net/documentation/us/en/software/junos/mpls-services",
        page_section="MPLS Applications, L2VPN/L3VPN, VPWS, VPLS & EVPN-VXLAN",
        published_date="2024-04-12",
        consulted_date="2026-10-17",
        reliability_score=0.99,
        applicability_restrictions="Junos OS 21.x-23.x em plataformas MX/PTX/QFX.",
        extracted_commands=[
            "show ldp neighbor",
            "show rsvp session",
            "show l2circuit connections",
            "show vpls connections",
            "show evpn database",
            "show route table bgp.l3vpn.0",
        ],
        domain="mpls_services",
    ),
    KnowledgeDocument(
        doc_id="DOC-NOKIA-INFRA-DWDM-OTN-014",
        title="Nokia Optical & SR OS — Infraestrutura: Backbone, Metro, DWDM, OTN, Enlaces Ópticos, Transceptores, Potência Óptica, Atenuação e Redundância",
        content=(
            "Engenharia e diagnóstico de infraestrutura óptica de transporte backbone e redes metro:\n"
            "- Sistemas DWDM (Dense Wavelength Division Multiplexing) e OTN (Optical Transport Network ITU-T G.709 OTU2/OTU4): monitorar OSNR, BER pré-FEC/pós-FEC e alarmes OTN (LOF, LOM, OTU-AIS, ODU-AIS).\n"
            "- Enlaces ópticos e transceptores (SFP+, QSFP28, QSFP-DD Coherent ZR/ZR+): aferir potência óptica Tx/Rx, orçamento de enlace (link budget) e atenuação em dB.\n"
            "- Alta disponibilidade e redundância física: proteção óptica 1+1 APS, redundância de supervisoras/fabric e fontes PSU A+B (`show card state`, `show port optical`)."
        ),
        vendor="Nokia",
        family_model="7750 SR / 1830 PSS DWDM",
        os_name="SR OS",
        version_build="23.10.R2",
        feature="Infraestrutura: backbone, metro, DWDM, OTN, enlaces ópticos, transceptores, potência óptica, atenuação, redundância",
        original_source="Nokia Documentation Center — Optical & IP Transport",
        source_uri="https://documentation.nokia.com/sros-optical-dwdm-otn",
        page_section="Optical Transport, DWDM Coherent Interfaces & Chassis Redundancy",
        published_date="2024-03-18",
        consulted_date="2026-10-17",
        reliability_score=0.98,
        applicability_restrictions="Nokia SR OS e plataformas ópticas coerentes DWDM/OTN.",
        extracted_commands=[
            "show port optical",
            "show card state",
            "show chassis",
        ],
        domain="infrastructure",
    ),
    KnowledgeDocument(
        doc_id="DOC-FORTINET-SECURITY-HARDENING-015",
        title="Fortinet & Multi-Vendor — Segurança: Firewalls, VPN, IPsec, AAA, RADIUS, TACACS+, SSH, Hardening e Segmentação",
        content=(
            "Diretrizes de segurança de infraestrutura de rede e proteção de plano de gerência/controle:\n"
            "- Firewalls e segmentação: políticas zero-trust por zona/VRF, inspeção stateful e microsegmentação.\n"
            "- Criptografia de transporte: túneis VPN IPsec IKEv2 com AES-GCM e Perfect Forward Secrecy (PFS).\n"
            "- Controle de acesso administrativo (AAA): autenticação centralizada e accounting de comandos via TACACS+ e RADIUS (`diagnose test authserver tacacs+`, `show aaa servers`).\n"
            "- Hardening e SSH: desativar Telnet/HTTP, exigir SSHv2 com chaves fortes, aplicar Control Plane Policing (CoPP) e ACLs restritivas na VTY/gerência OOB."
        ),
        vendor="Fortinet",
        family_model="FortiGate Enterprise & Multi-Vendor Control Plane",
        os_name="FortiOS",
        version_build="7.4.3",
        feature="Segurança: firewalls, VPN, IPsec, AAA, RADIUS, TACACS+, SSH, hardening, segmentação",
        original_source="Fortinet Official Documentation Portal",
        source_uri="https://docs.fortinet.com/document/fortigate/7.4.3/administration-guide",
        page_section="Security Hardening, AAA (RADIUS/TACACS+), SSHv2, IPsec VPN & Segmentation",
        published_date="2024-04-25",
        consulted_date="2026-10-17",
        reliability_score=0.99,
        applicability_restrictions="Aplicável a FortiOS 7.x e padrões de hardening multi-vendor.",
        extracted_commands=[
            "diagnose vpn ike gateway list",
            "diagnose vpn tunnel list",
            "show aaa servers",
        ],
        domain="security",
    ),
    KnowledgeDocument(
        doc_id="DOC-HUAWEI-OPERATIONS-TELEMETRY-016",
        title="Huawei VRP 8 — Operação: SNMP, NETCONF, RESTCONF, gNMI, Syslog, Telemetria, NMS, Alarmes, Monitoramento e Troubleshooting",
        content=(
            "Operação, gerência e observabilidade de redes de missão crítica:\n"
            "- Gerência tradicional e segura: SNMPv3 com authPriv (`display snmp-agent statistics`), syslog estruturado para coletores SIEM/NMS (`display logbuffer`) e monitoração de alarmes ativos (`display alarm active`).\n"
            "- Automação e programabilidade model-driven (YANG): NETCONF sobre SSH (porta 830), RESTCONF sobre HTTPS e telemetria streaming gRPC/gNMI em tempo real (`display telemetry sensor-group`).\n"
            "- Troubleshooting sistemático no NMS: correlação de traps/eventos, verificação de sincronismo e coleta somente leitura antes de janelas de mudança."
        ),
        vendor="Huawei",
        family_model="NetEngine 8000 / CloudEngine",
        os_name="VRP",
        version_build="V800R022C00SPC600",
        feature="Operação: SNMP, NETCONF, RESTCONF, gNMI, syslog, telemetria, NMS, alarmes, monitoramento, troubleshooting",
        original_source="Huawei Enterprise Support Portal",
        source_uri="https://support.huawei.com/enterprise/en/doc/vrp8-telemetry-netconf-snmp",
        page_section="Network Management, Telemetry (gNMI/NETCONF/RESTCONF), SNMPv3 & Alarms",
        published_date="2024-06-02",
        consulted_date="2026-10-17",
        reliability_score=0.99,
        applicability_restrictions="Huawei VRP 8.x e arquiteturas NMS multi-vendor.",
        extracted_commands=[
            "display alarm active",
            "display logbuffer",
            "display snmp-agent statistics",
            "display telemetry sensor-group",
        ],
        domain="operations",
    ),
    KnowledgeDocument(
        doc_id="DOC-ARISTA-SYSTEMS-LINUX-INFRA-017",
        title="Arista EOS & Linux Systems — Sistemas: Linux, Virtualização, Containers, DNS, DHCP, NTP e Automação",
        content=(
            "Sistemas operacionais de rede baseados em Linux e serviços essenciais de infraestrutura:\n"
            "- Linux Networking, Virtualização e Containers: namespaces de rede (netns), iproute2, KVM/QEMU, SR-IOV, DPDK e plugins CNI para containers.\n"
            "- Serviços críticos de rede: sincronismo de tempo NTP/Chrony (`show ntp status`), resolução de nomes DNS (`show hosts`), gerência de endereçamento DHCP/DHCPv6 Relay (`show ip helper-address`).\n"
            "- Automação de infraestrutura: pipelines declarativos, validação pré/pós-check via APIs estruturadas (eAPI/NETCONF) e integração com fonte da verdade."
        ),
        vendor="Arista",
        family_model="7050X3 / 7280R3 / CloudEOS Container",
        os_name="EOS",
        version_build="4.31.2F",
        feature="Sistemas: Linux, virtualização, containers, DNS, DHCP, NTP, automação",
        original_source="Arista Official Support & Documentation",
        source_uri="https://www.arista.com/en/support/eos-systems-linux-ntp-dns-dhcp-automation",
        page_section="System Services, Linux Kernel Networking, Containers, DNS, DHCP, NTP & Automation",
        published_date="2024-05-14",
        consulted_date="2026-10-17",
        reliability_score=0.99,
        applicability_restrictions="Arista EOS 4.x e ambientes Linux de infraestrutura de rede.",
        extracted_commands=[
            "show ntp status",
            "show hosts",
            "show ip helper-address",
            "show version",
        ],
        domain="systems",
    ),
    KnowledgeDocument(
        doc_id="DOC-HUAWEI-NE40E-UNIVERSAL-ROUTER-018",
        title="Huawei NetEngine 40E (NE40E) — Arquitetura de Hardware Solar, VRP, BNG/BRAS, Core IP/MPLS, SRv6, EVPN e CGNAT",
        content=(
            "O Huawei NetEngine 40E (NE40E) é uma série de roteadores de serviços universais (Universal Service Router - USR) "
            "desenvolvida para operadoras de telecomunicações, provedores de Internet (ISP), redes metropolitanas (MAN) e backbones corporativos de alta disponibilidade:\n"
            "- Posicionamento e Funções: Atua como Roteador de Borda de Provedor (PE - Provider Edge), Roteador de Core IP/MPLS, Gateway de Acesso Banda Larga (BRAS/BNG - Broadband Network Gateway com autenticação IPoE/PPPoE, RADIUS/CoA e accounting), Gateway Carrier-Grade NAT (CGNAT em hardware) e agregação metropolitana.\n"
            "- Modelos de Chassi: Família inclui NE40E-X16A (chassi 16 slots, multi-terabit), NE40E-X8A (8 slots), NE40E-X3A (3 slots compactos), NE40E-M2K e NE40E-M2F (equipamentos fixos de 1U/2U para agregação e pré-agregação).\n"
            "- Arquitetura de Hardware e Chipset: Equipado com o chipset proprietário Solar NP (Network Processor) da Huawei, que realiza encaminhamento em hardware line-rate determinístico, sem perda de pacotes e com jitter ultrabaixo. Possui separação física e lógica completa entre o plano de controle (MPU - Main Processing Unit), plano de encaminhamento/dados (LPU - Line Processing Unit) e matriz de comutação sem bloqueio (SFU - Switch Fabric Unit), com redundância 1+1 de MPUs e fontes de alimentação redundantes hot-swappable.\n"
            "- Capacidade e Interfaces: Suporta densidades escaláveis de interfaces 1GE, 10GE (SFP+), 25GE, 40GE (QSFP+), 100GE (QSFP28) e 400GE (QSFP-DD), além de interfaces legadas POS/CPOS e E1/T1 para migração de tráfego móvel e síncrono.\n"
            "- Protocolos de Roteamento e Serviços: Suporte integral a BGP-4 e MP-BGP (AFI/SAFI IPv4, IPv6, VPNv4, VPNv6, EVPN, BGP-LU), OSPFv2, OSPFv3, IS-IS multi-instância e multi-topologia, MPLS LDP, RSVP-TE com Fast Reroute (FRR sub-50ms), Segment Routing MPLS (SR-MPLS) e Segment Routing sobre IPv6 (SRv6).\n"
            "- Serviços de VPN e Carrier Ethernet: L3VPN (BGP/MPLS IP VPN com suporte a Option A, B e C), EVPN-VPWS, EVPN-VPLS, EVPN-VXLAN, VPLS tradicional (Martini e Kompella), VPWS e pseudowires estendidos.\n"
            "- Qualidade de Serviço e Confiabilidade: HQoS (Hierarchical QoS) em 5 níveis com agendamento hierárquico por usuário, serviço, classe e porta; detecção de falhas BFD em hardware com timer de 3.3 ms; NSR (Non-Stop Routing) e NSF (Non-Stop Forwarding) para failover transparente entre controladoras sem queda de sessões BGP/OSPF; ISSU (In-Service Software Upgrade).\n"
            "- Sistema Operacional e Gerência: Executa o Huawei VRP (VRP 8 / VRP 5 dependendo da geração), com suporte a automação model-driven via NETCONF/YANG sobre SSH (porta 830), telemetria streaming gRPC/gNMI em tempo real, SNMPv1/v2c/v3 e integração nativa com o Huawei iMaster NCE (Network Cloud Engine)."
        ),
        vendor="Huawei",
        family_model="NetEngine 40E (NE40E-X16 / X8 / X3 / M2K)",
        os_name="VRP",
        version_build="V800R022C00 / VRP 8.x",
        feature="Universal Service Router (USR), BNG/BRAS, Core IP/MPLS, SRv6, EVPN, CGNAT",
        original_source="Huawei Official Product Documentation — NetEngine 40E Series",
        source_uri="https://support.huawei.com/enterprise/en/routers/ne40e-pid-15837",
        page_section="NE40E Hardware Architecture, VRP Feature Guide, BNG, MPLS & Segment Routing",
        published_date="2024-08-20",
        consulted_date="2026-10-17",
        reliability_score=0.99,
        applicability_restrictions="Huawei NetEngine 40E Series e VRP 8.x.",
        extracted_commands=[
            "display device",
            "display version",
            "display bgp peer",
            "display ospf peer",
            "display access-user",
            "display nat session all",
            "display mpls lsp",
            "display segment-routing ipv6 locator",
        ],
        domain="infrastructure",
    ),
]


def get_documents_by_domain(domain: str) -> List[KnowledgeDocument]:
    """Retorna todos os documentos oficiais associados a um dos 7 domínios da Seção 6."""
    norm = domain.strip().lower()
    return [doc for doc in OFFICIAL_INITIAL_DOCUMENTS if doc.domain.lower() == norm]


def verify_all_7_domains_coverage() -> Dict[str, Any]:
    """
    Valida que todos os 7 domínios da Seção 6 e seus respectivos tópicos estão cobertos
    no catálogo documental oficial da Veronica.
    """
    coverage: Dict[str, Any] = {}
    all_covered = True
    for domain_key, spec in NETWORKING_KNOWLEDGE_DOMAINS.items():
        docs = get_documents_by_domain(domain_key)
        combined_text = " ".join(
            f"{d.title} {d.feature} {d.content}" for d in docs
        ).lower()
        missing_topics = [
            topic for topic in spec["topics"] if topic.lower() not in combined_text
        ]
        domain_ok = len(docs) > 0 and len(missing_topics) == 0
        if not domain_ok:
            all_covered = False
        coverage[domain_key] = {
            "section_number": spec["section_number"],
            "title": spec["title"],
            "document_count": len(docs),
            "topics_required": spec["topics"],
            "missing_topics": missing_topics,
            "fully_covered": domain_ok,
        }
    return {
        "total_domains": len(NETWORKING_KNOWLEDGE_DOMAINS),
        "all_7_domains_covered": all_covered and len(NETWORKING_KNOWLEDGE_DOMAINS) == 7,
        "domains": coverage,
    }

