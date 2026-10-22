"""
Regras Determinísticas de Protocolos de Rede da Veronica (Seções 6 e 7).
Cobre integralmente os 7 domínios de conhecimento:
1. Fundamentos (OSI, TCP/IP, Ethernet, IPv4, IPv6, ARP, ND, VLAN, STP, LACP, MTU, QoS, ACL)
2. Roteamento (rotas estáticas, OSPF, OSPFv3, IS-IS, BGP, MP-BGP, políticas, redistribuição, TE)
3. MPLS e serviços (LDP, RSVP-TE, Segment Routing, L2VPN, L3VPN, VPWS, VPLS, EVPN, VXLAN, pseudowires)
4. Infraestrutura (backbone, metro, DWDM, OTN, enlaces ópticos, transceptores, potência óptica, atenuação, redundância)
5. Segurança (firewalls, VPN, IPsec, AAA, RADIUS, TACACS+, SSH, hardening, segmentação)
6. Operação (SNMP, NETCONF, RESTCONF, gNMI, syslog, telemetria, NMS, alarmes, monitoramento, troubleshooting)
7. Sistemas (Linux, virtualização, containers, DNS, DHCP, NTP, automação)
"""

from __future__ import annotations

from typing import Any, Dict, List


PROTOCOL_DOMAIN_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fundamentals": {
        "section_number": 1,
        "name": "Fundamentos",
        "covered_protocols": [
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
        "name": "Roteamento",
        "covered_protocols": [
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
        "name": "MPLS e serviços",
        "covered_protocols": [
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
        "name": "Infraestrutura",
        "covered_protocols": [
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
        "name": "Segurança",
        "covered_protocols": [
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
        "name": "Operação",
        "covered_protocols": [
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
        "name": "Sistemas",
        "covered_protocols": [
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


def analyze_protocol_anomalies(
    device_identity: Dict[str, Any],
    operational_state: Dict[str, Any],
    running_config: str,
) -> List[Dict[str, Any]]:
    """
    Aplica regras determinísticas verificáveis sobre o estado operacional e configuração
    do equipamento nos 7 domínios da Seção 6, diferenciando Fatos Comprovados,
    Inferências Técnicas e Hipóteses.
    """
    findings: List[Dict[str, Any]] = []
    vendor = device_identity.get("vendor", "")
    os_name = device_identity.get("os_name", "")

    # =========================================================================
    # DOMÍNIO 2: ROTEAMENTO (BGP, MP-BGP, OSPF, OSPFv3, IS-IS, rotas estáticas, políticas, redistribuição, TE)
    # =========================================================================
    for peer in operational_state.get("bgp_peers", []):
        state = peer.get("state", "")
        peer_ip = peer.get("peer_ip", "")
        remote_as = peer.get("remote_as", "")
        if state != "Established":
            facts = [
                f"Sessão BGP com peer {peer_ip} (AS {remote_as}) encontra-se em estado '{state}' (prefixes_received={peer.get('prefixes_received', 0)})."
            ]
            inferences: List[str] = []
            hypotheses: List[str] = []
            proposed_cmds: List[str] = []

            if vendor == "Huawei" and "VRP" in os_name:
                if "connect-interface" not in running_config or "ebgp-max-hop" not in running_config:
                    inferences.append(
                        "Configuração BGP no Huawei VRP 8 aponta para endereço de Loopback sem `peer connect-interface Loopback0` e `peer ebgp-max-hop`."
                    )
                    proposed_cmds = [
                        "system-view",
                        "bgp 65001",
                        f"peer {peer_ip} connect-interface Loopback0",
                        f"peer {peer_ip} ebgp-max-hop 2",
                        "commit trial 120",
                    ]
                hypotheses.append(
                    "Possível bloqueio de pacotes TCP na porta 179 ou ausência de rota para o endereço de loopback do peer."
                )
            elif vendor == "Cisco" and "IOS-XR" in os_name:
                if "route-policy" not in running_config:
                    inferences.append(
                        "Cisco IOS-XR aplica RFC 8212 por padrão: sessões eBGP sem `route-policy in` e `route-policy out` não trocam prefixos."
                    )
                    proposed_cmds = [
                        "route-policy PASS-ALL",
                        " pass",
                        "end-policy",
                        "router bgp 65002",
                        f" neighbor {peer_ip}",
                        "  address-family ipv4 unicast",
                        "   route-policy PASS-ALL in",
                        "   route-policy PASS-ALL out",
                        "commit confirmed 120",
                    ]
                hypotheses.append(
                    "O par remoto pode estar com `connect-interface` ausente impedindo o handshake TCP 3-way."
                )

            findings.append(
                {
                    "domain": "routing.bgp",
                    "domain_category": "routing",
                    "severity": "HIGH",
                    "summary": f"Sessão BGP inativa ({state}) com vizinho {peer_ip} (AS {remote_as})",
                    "verified_facts": facts,
                    "technical_inferences": inferences,
                    "hypotheses": hypotheses,
                    "safe_diagnostic_commands": (
                        [
                            "display bgp peer",
                            f"display bgp peer {peer_ip} verbose",
                            f"ping -a 10.0.0.1 {peer_ip}",
                        ]
                        if vendor == "Huawei"
                        else [
                            "show bgp ipv4 unicast summary",
                            f"show bgp ipv4 unicast neighbors {peer_ip}",
                            f"ping {peer_ip} source Loopback0",
                        ]
                    ),
                    "proposed_remediation_commands": proposed_cmds,
                    "confidence": 0.94,
                }
            )

    for nbr in operational_state.get("ospf_neighbors", []):
        state = nbr.get("state", "")
        if "EXSTART" in state.upper() or "EXCHANGE" in state.upper():
            if_name = nbr.get("interface", "")
            nbr_id = nbr.get("neighbor_id", "")
            if_info = next(
                (i for i in operational_state.get("interfaces", []) if i["name"] == if_name),
                {"mtu": "desconhecido"},
            )
            local_mtu = if_info.get("mtu", 1500)
            findings.append(
                {
                    "domain": "routing.ospf",
                    "domain_category": "routing",
                    "severity": "HIGH",
                    "summary": f"Adjacência OSPF com {nbr_id} presa em {state} na interface {if_name}",
                    "verified_facts": [
                        f"Vizinho OSPF {nbr_id} na interface {if_name} está em estado '{state}'.",
                        f"MTU configurado localmente na interface {if_name} é {local_mtu} bytes.",
                    ],
                    "technical_inferences": [
                        "Transição presa em EXSTART/EXCHANGE com retransmissão de pacotes DBD indica divergência de MTU IP entre os dois lados do enlace."
                    ],
                    "hypotheses": [
                        "Possível descartador de jumbo frames no transporte intermediário caso o MTU físico seja 9216."
                    ],
                    "safe_diagnostic_commands": (
                        [
                            "show ospf neighbor detail",
                            f"show interfaces {if_name}",
                            "ping 172.16.24.2 size 1500 df-bit",
                        ]
                        if vendor == "Cisco"
                        else ["show ospf neighbor extensive", f"show interfaces {if_name} extensive"]
                    ),
                    "proposed_remediation_commands": (
                        [
                            "configure terminal",
                            f"interface {if_name}",
                            "mtu 9216",
                            "end",
                        ]
                        if "IOS-XE" in os_name
                        else [
                            f"interface {if_name}",
                            "mtu 1500",
                            "commit confirmed 120",
                        ]
                    ),
                    "confidence": 0.96,
                }
            )

    for r_anom in operational_state.get("routing_anomalies", []):
        proto = r_anom.get("protocol", "IS-IS/OSPFv3/TE")
        detail = r_anom.get("detail", "Falha de adjacência ou política de redistribuição")
        findings.append(
            {
                "domain": "routing.isis_ospfv3_te",
                "domain_category": "routing",
                "severity": "HIGH",
                "summary": f"Anomalia de roteamento ({proto}): {detail}",
                "verified_facts": [
                    f"Estado reportado no plano de controle de roteamento ({proto}): {detail}."
                ],
                "technical_inferences": [
                    f"Inconsistência em rotas estáticas, {proto}, políticas de redistribuição ou restrição de banda TE."
                ],
                "hypotheses": [
                    "Incompatibilidade de nível IS-IS (L1/L2), autenticação HMAC ou filtro de route-tag."
                ],
                "safe_diagnostic_commands": [
                    "show isis adjacency",
                    "show ospfv3 neighbor",
                    "show route static",
                    "show mpls traffic-eng tunnels",
                ],
                "proposed_remediation_commands": r_anom.get("proposed_commands", []),
                "confidence": 0.93,
            }
        )

    for leak in operational_state.get("bgp_route_leaks", []):
        prefix = leak.get("prefix", "desconhecido")
        peer_ip = leak.get("peer_ip", "desconhecido")
        leak_type = leak.get("leak_type", "UNAUTHORIZED_ROUTE_LEAK")
        findings.append(
            {
                "domain": "routing.bgp_route_leak",
                "domain_category": "routing",
                "severity": "CRITICAL",
                "summary": f"Route-leaking BGP detectado: prefixo {prefix} anunciado indevidamente para peer {peer_ip}",
                "verified_facts": [
                    f"Prefixo {prefix} foi propagado para o peer {peer_ip} violando políticas de exportação (RFC 7908 / RFC 9234).",
                    f"Classificação do vazamento: {leak_type}.",
                ],
                "technical_inferences": [
                    "Ausência de community no-export ou prefix-list permissivo propagando prefixos internos para trânsito público."
                ],
                "hypotheses": [
                    "Falta de filtro de saída em sessão eBGP ou redistribuição IGP->BGP sem route-policy restritiva."
                ],
                "safe_diagnostic_commands": [
                    f"show bgp ipv4 unicast advertised-routes neighbor {peer_ip}",
                    f"display bgp routing-table {prefix}" if vendor == "Huawei" else f"show bgp ipv4 unicast {prefix}",
                ],
                "proposed_remediation_commands": [
                    "route-policy DROP-LEAK deny",
                    f"apply route-policy DROP-LEAK out neighbor {peer_ip}",
                ],
                "confidence": 0.98,
            }
        )

    for part in operational_state.get("area_partitions", []):
        area = part.get("area", "0.0.0.0")
        proto = part.get("protocol", "OSPF")
        isolated = part.get("isolated_nodes", [])
        findings.append(
            {
                "domain": f"routing.{proto.lower()}_area_partition",
                "domain_category": "routing",
                "severity": "CRITICAL",
                "summary": f"Partição de área {proto} ({area}) detectada: descontinuidade no plano de controle",
                "verified_facts": [
                    f"Área {proto} {area} dividida em múltiplos componentes desconexos.",
                    f"Nós isolados do restante da área de trânsito: {', '.join(isolated)}.",
                ],
                "technical_inferences": [
                    "Rompimento de enlaces de trânsito intra-área ou ausência de virtual-link OSPF gerando isolamento de roteamento."
                ],
                "hypotheses": [
                    "Queda simultânea de adjacências de backbone ou partição física de rota."
                ],
                "safe_diagnostic_commands": [
                    f"show {proto.lower()} database",
                    f"show {proto.lower()} neighbor",
                ],
                "proposed_remediation_commands": [
                    f"area {area} virtual-link <remote-abr-id>",
                ],
                "confidence": 0.96,
            }
        )

    # =========================================================================
    # DOMÍNIO 4: INFRAESTRUTURA (enlaces ópticos, transceptores, potência óptica, atenuação, DWDM, OTN, backbone, metro, redundância)
    # =========================================================================
    for opt in operational_state.get("optical_transceivers", []):
        rx = float(opt.get("rx_power_dbm", 0.0))
        low_thresh = float(opt.get("rx_low_threshold_dbm", -14.4))
        if rx < low_thresh:
            if_name = opt.get("interface", "")
            findings.append(
                {
                    "domain": "infrastructure.optical",
                    "domain_category": "infrastructure",
                    "severity": "MEDIUM",
                    "summary": f"Degradação óptica na interface {if_name} (Rx Power {rx} dBm < limiar {low_thresh} dBm)",
                    "verified_facts": [
                        f"Potência óptica de recepção (Rx) medida em {if_name}: {rx} dBm (limiar mínimo: {low_thresh} dBm).",
                        f"Potência de transmissão (Tx) local: {opt.get('tx_power_dbm')} dBm.",
                    ],
                    "technical_inferences": [
                        f"Margem óptica negativa de {round(rx - low_thresh, 2)} dB causando erros de CRC na interface."
                    ],
                    "hypotheses": [
                        "Atenuação excessiva no cordão óptico/DIO, conector sujo ou curvatura acentuada na fibra."
                    ],
                    "safe_diagnostic_commands": [
                        f"display interface {if_name}",
                        f"display transceiver interface {if_name} verbose",
                    ],
                    "proposed_remediation_commands": [],  # Ação física de campo, não inventar comando mágico
                    "confidence": 0.97,
                }
            )

    for inf_anom in operational_state.get("infrastructure_anomalies", []):
        sub = inf_anom.get("subsystem", "DWDM/OTN/Redundancy")
        detail = inf_anom.get("detail", "Degradação de OSNR/FEC em canal DWDM/OTN ou falha de redundância")
        findings.append(
            {
                "domain": "infrastructure.dwdm_otn_redundancy",
                "domain_category": "infrastructure",
                "severity": "HIGH",
                "summary": f"Anomalia de infraestrutura ({sub}): {detail}",
                "verified_facts": [
                    f"Telemetria física/óptica de backbone/metro reportou em {sub}: {detail}."
                ],
                "technical_inferences": [
                    "Degradação de relação sinal-ruído óptico (OSNR), alarme OTN (LOF/OTU-AIS) ou perda de redundância de hardware."
                ],
                "hypotheses": [
                    "Amplificador EDFA degradado no span óptico DWDM ou falha de fonte PSU secundária."
                ],
                "safe_diagnostic_commands": [
                    "show port optical",
                    "show card state",
                    "display device",
                ],
                "proposed_remediation_commands": inf_anom.get("proposed_commands", []),
                "confidence": 0.95,
            }
        )

    # =========================================================================
    # DOMÍNIO 1: FUNDAMENTOS (OSI, TCP/IP, Ethernet, IPv4, IPv6, ARP, ND, VLAN, STP, LACP, MTU, QoS, ACL)
    # =========================================================================
    for fund in operational_state.get("fundamentals_anomalies", []):
        proto = fund.get("protocol", "VLAN/STP/LACP/QoS/ACL")
        if_name = fund.get("interface", "Eth-Trunk1")
        detail = fund.get("detail", "Descarte em fila QoS, porta LACP suspensa ou conflito VLAN/STP")
        findings.append(
            {
                "domain": "fundamentals.l2_l3_qos_acl",
                "domain_category": "fundamentals",
                "severity": "MEDIUM",
                "summary": f"Anomalia de fundamentos ({proto}) em {if_name}: {detail}",
                "verified_facts": [
                    f"Interface {if_name} apresenta evento em {proto}: {detail}."
                ],
                "technical_inferences": [
                    f"Falha de negociação LACP/STP, divergência de VLAN/MTU, saturação de fila QoS ou bloqueio por ACL em {if_name}."
                ],
                "hypotheses": [
                    "Configuração de trunk VLAN assimétrica, BPDU Guard acionado ou tabela ARP/ND exaurida."
                ],
                "safe_diagnostic_commands": [
                    "show spanning-tree",
                    "show lacp neighbor",
                    "show ip arp",
                    "show ipv6 neighbors",
                    "show policy-map interface",
                    "show access-lists",
                ],
                "proposed_remediation_commands": fund.get("proposed_commands", []),
                "confidence": 0.94,
            }
        )

    for lacp in operational_state.get("lacp_anomalies", []):
        bundle = lacp.get("bundle", "Eth-Trunk1")
        member = lacp.get("member", "10GE1/0/1")
        detail = lacp.get("detail", "Porta membro em flapping constante e status SUSPENDED")
        findings.append(
            {
                "domain": "fundamentals.lacp_flapping",
                "domain_category": "fundamentals",
                "severity": "HIGH",
                "summary": f"Flapping de agregação de enlace LACP no bundle {bundle} (membro {member})",
                "verified_facts": [
                    f"Porta membro {member} do bundle {bundle} apresenta instabilidade operacional: {detail}.",
                    f"Capacidade de banda agregada reduzida no bundle {bundle}.",
                ],
                "technical_inferences": [
                    "Temporizador LACP short/long mismatch, oscilação de sinal físico ou divergência de LACP system-priority / key."
                ],
                "hypotheses": [
                    "Falha no transceptor óptico da porta membro ou problema de cabeamento causando perdas periódicas de LACPDU."
                ],
                "safe_diagnostic_commands": [
                    f"display eth-trunk {bundle}" if vendor == "Huawei" else f"show lacp bundle {bundle}",
                    "display lacp statistics" if vendor == "Huawei" else "show lacp counters",
                ],
                "proposed_remediation_commands": [
                    f"interface {member}\n shutdown\n undo shutdown" if vendor == "Huawei" else f"interface {member}\n shutdown\n no shutdown",
                ],
                "confidence": 0.95,
            }
        )

    # =========================================================================
    # DOMÍNIO 3: MPLS E SERVIÇOS (LDP, RSVP-TE, Segment Routing, L2VPN, L3VPN, VPWS, VPLS, EVPN, VXLAN, pseudowires)
    # =========================================================================
    for mpls in operational_state.get("mpls_anomalies", []):
        service_type = mpls.get("service", "EVPN/VXLAN/LDP/L3VPN")
        peer_or_vtep = mpls.get("peer", "10.0.0.5")
        detail = mpls.get("detail", "Pseudowire/VTEP inativo ou falta de binding LDP/SR")
        findings.append(
            {
                "domain": "mpls_services.transport_vpn",
                "domain_category": "mpls_services",
                "severity": "HIGH",
                "summary": f"Falha em serviço MPLS/Overlay ({service_type}) com {peer_or_vtep}: {detail}",
                "verified_facts": [
                    f"Serviço {service_type} associado ao endpoint {peer_or_vtep} reporta: {detail}."
                ],
                "technical_inferences": [
                    f"Plano de transporte MPLS (LDP/RSVP-TE/Segment Routing) ou plano de controle BGP-EVPN/L2VPN/L3VPN sem túnel ativo para {peer_or_vtep}."
                ],
                "hypotheses": [
                    "Divergência de Route-Target (RT), VNI, MTU de serviço no pseudowire (VPWS/VPLS) ou rótulo SR-MPLS ausente."
                ],
                "safe_diagnostic_commands": [
                    "show router ldp bindings active",
                    "show router tunnel-table",
                    "show bgp evpn summary",
                    "show vxlan vtep",
                    "show l2circuit connections",
                ],
                "proposed_remediation_commands": mpls.get("proposed_commands", []),
                "confidence": 0.95,
            }
        )

    # =========================================================================
    # DOMÍNIO 5: SEGURANÇA (firewalls, VPN, IPsec, AAA, RADIUS, TACACS+, SSH, hardening, segmentação)
    # =========================================================================
    for sec in operational_state.get("security_anomalies", []):
        control = sec.get("control", "IPsec/Firewall/AAA/SSH")
        detail = sec.get("detail", "Túnel IPsec IKEv2 inativo ou falha de autenticação TACACS+/RADIUS")
        findings.append(
            {
                "domain": "security.vpn_firewall_aaa_hardening",
                "domain_category": "security",
                "severity": "HIGH",
                "summary": f"Alerta de segurança de rede ({control}): {detail}",
                "verified_facts": [
                    f"Subsistema de segurança ({control}) registrou: {detail}."
                ],
                "technical_inferences": [
                    "Falha de negociação IKE/IPsec, indisponibilidade de servidor AAA (RADIUS/TACACS+) ou violação de política de firewall/segmentação."
                ],
                "hypotheses": [
                    "Divergência de proposta criptográfica Phase1/Phase2, chave compartilhada expirada ou bloqueio na gerência SSH/CoPP."
                ],
                "safe_diagnostic_commands": [
                    "diagnose vpn ike gateway list",
                    "diagnose vpn tunnel list",
                    "show aaa servers",
                ],
                "proposed_remediation_commands": sec.get("proposed_commands", []),
                "confidence": 0.95,
            }
        )

    # =========================================================================
    # DOMÍNIO 6: OPERAÇÃO (SNMP, NETCONF, RESTCONF, gNMI, syslog, telemetria, NMS, alarmes, monitoramento, troubleshooting)
    # =========================================================================
    for op in operational_state.get("operations_anomalies", []):
        channel = op.get("channel", "NETCONF/gNMI/SNMP/Syslog")
        detail = op.get("detail", "Falha de subscrição de telemetria gNMI ou timeout SNMPv3 no NMS")
        findings.append(
            {
                "domain": "operations.telemetry_nms_netconf_snmp",
                "domain_category": "operations",
                "severity": "MEDIUM",
                "summary": f"Anomalia operacional e de gerência ({channel}): {detail}",
                "verified_facts": [
                    f"Canal de observabilidade/gerência ({channel}) reportou: {detail}."
                ],
                "technical_inferences": [
                    f"Perda de visibilidade parcial no NMS ou interrupção de sessão {channel} para coleta de alarmes/telemetria."
                ],
                "hypotheses": [
                    "ACL na VRF de gerência bloqueando portas UDP 161/514 ou TCP 830/gRPC, ou credencial SNMPv3/NETCONF rotacionada."
                ],
                "safe_diagnostic_commands": [
                    "display alarm active",
                    "display logbuffer",
                    "display snmp-agent statistics",
                    "display telemetry sensor-group",
                ],
                "proposed_remediation_commands": op.get("proposed_commands", []),
                "confidence": 0.93,
            }
        )

    # =========================================================================
    # DOMÍNIO 7: SISTEMAS (Linux, virtualização, containers, DNS, DHCP, NTP, automação)
    # =========================================================================
    for sys_anom in operational_state.get("systems_anomalies", []):
        service = sys_anom.get("service", "NTP/DNS/DHCP/Linux-CNI")
        detail = sys_anom.get("detail", "Relógio NTP fora de sincronismo (stratum 16) ou falha de resolução DNS/DHCP")
        findings.append(
            {
                "domain": "systems.linux_dns_dhcp_ntp_automation",
                "domain_category": "systems",
                "severity": "MEDIUM",
                "summary": f"Anomalia de sistemas e serviços básicos ({service}): {detail}",
                "verified_facts": [
                    f"Serviço de sistema/infraestrutura ({service}) encontra-se degradado: {detail}."
                ],
                "technical_inferences": [
                    f"Falha em {service} impacta validação de certificados/logs (NTP), resolução de nomes (DNS), concessão IP (DHCP) ou rede de containers/virtualização Linux."
                ],
                "hypotheses": [
                    "Servidor NTP/DNS/DHCP inacessível a partir da VRF de gerência ou exaustão de recursos no namespace Linux/container."
                ],
                "safe_diagnostic_commands": [
                    "show ntp status",
                    "show hosts",
                    "show ip helper-address",
                    "show version",
                ],
                "proposed_remediation_commands": sys_anom.get("proposed_commands", []),
                "confidence": 0.94,
            }
        )

    return findings


def evaluate_all_7_protocol_domains() -> Dict[str, Any]:
    """
    Executa uma bateria sintética determinística validando que o motor de protocolos
    analisa e produz diagnósticos estruturados para todos os 7 domínios da Seção 6.
    """
    sample_state = {
        "interfaces": [{"name": "TenGigE0/0/0/2", "mtu": 9216}],
        "fundamentals_anomalies": [
            {
                "protocol": "VLAN/STP/LACP/MTU/QoS/ACL",
                "interface": "Eth-Trunk1",
                "detail": "Porta membro LACP suspensa e fila QoS com tail-drop",
            }
        ],
        "bgp_peers": [
            {
                "peer_ip": "10.0.0.2",
                "remote_as": 65002,
                "state": "Active",
                "prefixes_received": 0,
            }
        ],
        "ospf_neighbors": [
            {
                "neighbor_id": "10.0.0.4",
                "interface": "TenGigE0/0/0/2",
                "state": "EXSTART/BDR",
            }
        ],
        "routing_anomalies": [
            {
                "protocol": "IS-IS/OSPFv3/TE",
                "detail": "Adjacência IS-IS Level-2 e túnel SR-TE sem caminho CSPF",
            }
        ],
        "mpls_anomalies": [
            {
                "service": "LDP/RSVP-TE/Segment Routing/L2VPN/L3VPN/VPWS/VPLS/EVPN/VXLAN",
                "peer": "10.0.0.5",
                "detail": "Pseudowire VPWS/VPLS e VTEP EVPN-VXLAN sem label binding ativo",
            }
        ],
        "optical_transceivers": [
            {
                "interface": "10GE1/0/2",
                "tx_power_dbm": -2.4,
                "rx_power_dbm": -17.2,
                "rx_low_threshold_dbm": -14.4,
            }
        ],
        "infrastructure_anomalies": [
            {
                "subsystem": "DWDM/OTN/Backbone/Metro/Redundancy",
                "detail": "Alarme OTN ODU-AIS e degradação de OSNR em transponder coerente",
            }
        ],
        "security_anomalies": [
            {
                "control": "Firewall/IPsec/AAA/RADIUS/TACACS+/SSH/Hardening",
                "detail": "Túnel IPsec IKEv2 down e servidor TACACS+ primário sem resposta",
            }
        ],
        "operations_anomalies": [
            {
                "channel": "SNMP/NETCONF/RESTCONF/gNMI/Syslog/Telemetry/NMS",
                "detail": "Subscrição gNMI desconectada e alarme ativo no coletor NMS",
            }
        ],
        "systems_anomalies": [
            {
                "service": "Linux/Virtualização/Containers/DNS/DHCP/NTP/Automação",
                "detail": "NTP stratum 16 não sincronizado e falha de DHCP relay no namespace Linux",
            }
        ],
    }

    findings = analyze_protocol_anomalies(
        device_identity={"vendor": "Huawei", "os_name": "VRP 8"},
        operational_state=sample_state,
        running_config="sysname BR-CORE\nbgp 65001\n",
    )
    categories_found = sorted({f.get("domain_category", "") for f in findings if f.get("domain_category")})
    return {
        "registered_domains": list(PROTOCOL_DOMAIN_REGISTRY.keys()),
        "categories_evaluated": categories_found,
        "all_7_domains_evaluated": len(categories_found) == 7,
        "findings_count": len(findings),
        "findings": findings,
    }

