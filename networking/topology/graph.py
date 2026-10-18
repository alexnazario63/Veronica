"""
Inventário e Grafo de Topologia de Redes da Veronica (Seções 5, 7 e 8).
Mantém os nós, enlaces, dependências de serviços e calcula o Raio de Impacto (Blast Radius).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List


def create_default_network_inventory() -> Dict[str, Dict[str, Any]]:
    """
    Cria o inventário realista do Digital Twin multi-vendor para diagnóstico e simulação segura.
    """
    return {
        "NE8000-CORE-01": {
            "device_id": "NE8000-CORE-01",
            "hostname": "BR-SP-NE8000-CORE-01",
            "vendor": "Huawei",
            "family_model": "NetEngine 8000 M8",
            "os_name": "VRP 8",
            "version": "V800R022C00SPC600",
            "management_ip": "10.255.0.11",
            "management_interface": "MEth0/0/0",
            "oob_available": True,
            "role": "BACKBONE_PE",
            "validated_for_write": True,
            "safe_commit_command": "commit trial 120",
            "cpu_percent": 21.4,
            "memory_percent": 44.8,
            "services_healthy": True,
            "running_config": (
                "sysname BR-SP-NE8000-CORE-01\n"
                "interface Loopback0\n"
                " ip address 10.0.0.1 255.255.255.255\n"
                "interface 10GE1/0/1\n"
                " description LINK-TO-ASR9K-CORE-02\n"
                " ip address 172.16.12.1 255.255.255.252\n"
                " mtu 9216\n"
                "bgp 65001\n"
                " router-id 10.0.0.1\n"
                " peer 10.0.0.2 as-number 65002\n"
                " # MISSING: peer 10.0.0.2 connect-interface Loopback0 & ebgp-max-hop\n"
            ),
            "interfaces": [
                {"name": "10GE1/0/1", "status": "UP", "protocol": "UP", "ip": "172.16.12.1/30", "mtu": 9216, "crc_errors": 0},
                {"name": "10GE1/0/2", "status": "UP", "protocol": "UP", "ip": "172.16.13.1/30", "mtu": 1500, "crc_errors": 48},
                {"name": "Loopback0", "status": "UP", "protocol": "UP", "ip": "10.0.0.1/32", "mtu": 1500, "crc_errors": 0},
            ],
            "bgp_peers": [
                {
                    "peer_ip": "10.0.0.2",
                    "remote_as": 65002,
                    "state": "Active",
                    "uptime": "00:00:00",
                    "prefixes_received": 0,
                    "last_error": "TCP Connection Timeout / Missing ebgp-max-hop & connect-interface",
                }
            ],
            "ospf_neighbors": [
                {
                    "neighbor_id": "10.0.0.3",
                    "interface": "10GE1/0/2",
                    "state": "FULL/DR",
                    "area": "0.0.0.0",
                }
            ],
            "optical_transceivers": [
                {
                    "interface": "10GE1/0/2",
                    "wavelength_nm": 1310,
                    "tx_power_dbm": -2.4,
                    "rx_power_dbm": -16.8,
                    "rx_low_threshold_dbm": -14.4,
                    "status": "LOW_RX_POWER_ALARM",
                }
            ],
            "alarms": [
                "BGP_PEER_NOT_ESTABLISHED: Peer 10.0.0.2 (AS 65002) stuck in Active state",
                "OPTICAL_RX_LOW: Interface 10GE1/0/2 Rx Power -16.8 dBm below threshold (-14.4 dBm)",
            ],
            "syslog": [
                "Oct 17 13:20:11 BR-SP-NE8000-CORE-01 %%01BGP/3/STATE_CHG: BGP peer 10.0.0.2 state changed from Connect to Active.",
                "Oct 17 13:22:04 BR-SP-NE8000-CORE-01 %%01OPTICAL/4/RX_POWER_LOW: 10GE1/0/2 Rx power (-16.8 dBm) below threshold (-14.4 dBm).",
            ],
            "diagnostic_outputs": {
                "display bgp peer": "Peer 10.0.0.2 V 4 AS 65002 MsgRcvd 0 MsgSent 0 OutQ 0 Up/Down 00:00:00 State Active PrefRcv 0",
                "display transceiver interface 10GE1/0/2 verbose": "Rx Power: -16.80 dBm (Low Alarm Threshold: -14.40 dBm), Tx Power: -2.40 dBm",
            },
            "neighbors": [
                {"neighbor_device": "ASR9K-CORE-02", "local_if": "10GE1/0/1", "remote_if": "TenGigE0/0/0/1", "protocol": "LLDP/BGP"},
                {"neighbor_device": "MX204-EDGE-03", "local_if": "10GE1/0/2", "remote_if": "xe-0/1/0", "protocol": "LLDP/OSPF"},
            ],
            "dependent_services": ["L3VPN-CORPORATE-BANK", "BACKBONE-TRANSIT-AS65002", "VOIP-CORE"],
            "affected_customers_estimate": 1450,
        },
        "ASR9K-CORE-02": {
            "device_id": "ASR9K-CORE-02",
            "hostname": "BR-SP-ASR9K-CORE-02",
            "vendor": "Cisco",
            "family_model": "ASR 9904",
            "os_name": "IOS-XR",
            "version": "7.9.2",
            "management_ip": "10.255.0.12",
            "management_interface": "MgmtEth0/RP0/CPU0/0",
            "oob_available": True,
            "role": "BACKBONE_PE",
            "validated_for_write": True,
            "safe_commit_command": "commit confirmed 120",
            "cpu_percent": 16.0,
            "memory_percent": 39.2,
            "services_healthy": True,
            "running_config": (
                "hostname BR-SP-ASR9K-CORE-02\n"
                "interface Loopback0\n"
                " ipv4 address 10.0.0.2 255.255.255.255\n"
                "interface TenGigE0/0/0/1\n"
                " description LINK-TO-NE8000-CORE-01\n"
                " ipv4 address 172.16.12.2 255.255.255.252\n"
                " mtu 9216\n"
                "interface TenGigE0/0/0/2\n"
                " description LINK-TO-C9500-DIST-04\n"
                " ipv4 address 172.16.24.1 255.255.255.252\n"
                " mtu 9216\n"
                "router bgp 65002\n"
                " bgp router-id 10.0.0.2\n"
                " neighbor 10.0.0.1\n"
                "  remote-as 65001\n"
                "  update-source Loopback0\n"
                "  ebgp-multihop 2\n"
                "  address-family ipv4 unicast\n"
                "  ! MISSING route-policy in/out (RFC 8212 enforcement)\n"
            ),
            "interfaces": [
                {"name": "TenGigE0/0/0/1", "status": "UP", "protocol": "UP", "ip": "172.16.12.2/30", "mtu": 9216, "crc_errors": 0},
                {"name": "TenGigE0/0/0/2", "status": "UP", "protocol": "UP", "ip": "172.16.24.1/30", "mtu": 9216, "crc_errors": 0},
            ],
            "bgp_peers": [
                {
                    "peer_ip": "10.0.0.1",
                    "remote_as": 65001,
                    "state": "Active",
                    "uptime": "00:00:00",
                    "prefixes_received": 0,
                    "last_error": "eBGP neighbor has no inbound/outbound policy configured (RFC 8212)",
                }
            ],
            "ospf_neighbors": [
                {
                    "neighbor_id": "10.0.0.4",
                    "interface": "TenGigE0/0/0/2",
                    "state": "EXSTART/BDR",
                    "area": "0.0.0.0",
                }
            ],
            "optical_transceivers": [],
            "alarms": [
                "OSPF_ADJ_STUCK_EXSTART: Neighbor 10.0.0.4 on TenGigE0/0/0/2 stuck in EXSTART (MTU 9216 vs remote 1500)",
                "BGP_MISSING_RPL: Neighbor 10.0.0.1 missing mandatory eBGP route-policy on IOS-XR",
            ],
            "syslog": [
                "RP/0/RP0/CPU0:Oct 17 13:21:10 : ospf[1018]: %ROUTING-OSPF-5-ADJCHG : Process 1, Nbr 10.0.0.4 on TenGigE0/0/0/2 from EXSTART to DOWN, Too many retransmissions (MTU mismatch)",
            ],
            "diagnostic_outputs": {
                "show bgp ipv4 unicast summary": "Neighbor 10.0.0.1 Spk 0 AS 65001 MsgRcvd 0 MsgSent 0 State Active (! No policy)",
                "show ospf neighbor detail": "Neighbor 10.0.0.4, interface TenGigE0/0/0/2, state EXSTART, DBD retransmissions: 14",
            },
            "neighbors": [
                {"neighbor_device": "NE8000-CORE-01", "local_if": "TenGigE0/0/0/1", "remote_if": "10GE1/0/1", "protocol": "LLDP/BGP"},
                {"neighbor_device": "C9500-DIST-04", "local_if": "TenGigE0/0/0/2", "remote_if": "TenGigabitEthernet1/0/1", "protocol": "CDP/OSPF"},
            ],
            "dependent_services": ["BACKBONE-TRANSIT-AS65002", "METRO-AGGREGATION-SP"],
            "affected_customers_estimate": 1200,
        },
        "MX204-EDGE-03": {
            "device_id": "MX204-EDGE-03",
            "hostname": "BR-SP-MX204-EDGE-03",
            "vendor": "Juniper",
            "family_model": "MX204",
            "os_name": "Junos OS",
            "version": "22.4R3-S2",
            "management_ip": "10.255.0.13",
            "management_interface": "fxp0.0",
            "oob_available": True,
            "role": "MPLS_EDGE",
            "validated_for_write": True,
            "safe_commit_command": "commit confirmed 2",
            "cpu_percent": 14.5,
            "memory_percent": 36.0,
            "services_healthy": True,
            "running_config": (
                "system {\n    host-name BR-SP-MX204-EDGE-03;\n}\n"
                "interfaces {\n    xe-0/1/0 {\n        unit 0 {\n            family inet {\n                address 172.16.13.2/30;\n            }\n        }\n    }\n}\n"
            ),
            "interfaces": [
                {"name": "xe-0/1/0", "status": "UP", "protocol": "UP", "ip": "172.16.13.2/30", "mtu": 1500, "crc_errors": 0}
            ],
            "bgp_peers": [
                {"peer_ip": "10.0.0.5", "remote_as": 65001, "state": "Established", "uptime": "42d 08:11:00", "prefixes_received": 640}
            ],
            "ospf_neighbors": [
                {"neighbor_id": "10.0.0.1", "interface": "xe-0/1/0.0", "state": "Full", "area": "0.0.0.0"}
            ],
            "optical_transceivers": [],
            "alarms": [],
            "syslog": [],
            "diagnostic_outputs": {"show bgp summary": "Peer 10.0.0.5 AS 65001 Establ 640/640/0"},
            "neighbors": [
                {"neighbor_device": "NE8000-CORE-01", "local_if": "xe-0/1/0", "remote_if": "10GE1/0/2", "protocol": "LLDP/OSPF"},
                {"neighbor_device": "SR7750-PE-05", "local_if": "et-0/0/0", "remote_if": "1/1/c1/1", "protocol": "LLDP/MPLS"},
            ],
            "dependent_services": ["EVPN-DATACENTER-INTERCONNECT"],
            "affected_customers_estimate": 620,
        },
        "C9500-DIST-04": {
            "device_id": "C9500-DIST-04",
            "hostname": "BR-SP-C9500-DIST-04",
            "vendor": "Cisco",
            "family_model": "Catalyst 9500-48Y4C",
            "os_name": "IOS-XE",
            "version": "17.9.4a",
            "management_ip": "10.255.0.14",
            "management_interface": "GigabitEthernet0/0",
            "oob_available": True,
            "role": "DISTRIBUTION",
            "validated_for_write": True,
            "safe_commit_command": "configure terminal revert timer 2",
            "cpu_percent": 19.0,
            "memory_percent": 41.0,
            "services_healthy": True,
            "running_config": (
                "hostname BR-SP-C9500-DIST-04\n"
                "interface TenGigabitEthernet1/0/1\n"
                " description UPLINK-TO-ASR9K-CORE-02\n"
                " ip address 172.16.24.2 255.255.255.252\n"
                " mtu 1500\n"
                " ip ospf 1 area 0\n"
            ),
            "interfaces": [
                {"name": "TenGigabitEthernet1/0/1", "status": "UP", "protocol": "UP", "ip": "172.16.24.2/30", "mtu": 1500, "crc_errors": 0}
            ],
            "bgp_peers": [],
            "ospf_neighbors": [
                {"neighbor_id": "10.0.0.2", "interface": "TenGigabitEthernet1/0/1", "state": "EXSTART/DR", "area": "0.0.0.0"}
            ],
            "optical_transceivers": [],
            "alarms": ["OSPF_MTU_MISMATCH: Neighbor 10.0.0.2 stuck in EXSTART (local MTU 1500 vs remote 9216)"],
            "syslog": [
                "%OSPF-5-ADJCHG: Process 1, Nbr 10.0.0.2 on TenGigabitEthernet1/0/1 from EXSTART to DOWN, Neighbor Down: Too many retransmissions"
            ],
            "diagnostic_outputs": {
                "show ip ospf neighbor detail": "Neighbor 10.0.0.2, interface address 172.16.24.1, state is EXSTART, DBD retransmissions"
            },
            "neighbors": [
                {"neighbor_device": "ASR9K-CORE-02", "local_if": "TenGigabitEthernet1/0/1", "remote_if": "TenGigE0/0/0/2", "protocol": "CDP/OSPF"},
                {"neighbor_device": "CCR2004-CPE-06", "local_if": "TenGigabitEthernet1/0/2", "remote_if": "sfp-sfpplus1", "protocol": "LLDP/BGP"},
            ],
            "dependent_services": ["METRO-AGGREGATION-SP", "CAMPUS-ENTERPRISE-ACCESS"],
            "affected_customers_estimate": 480,
        },
        "SR7750-PE-05": {
            "device_id": "SR7750-PE-05",
            "hostname": "BR-RJ-SR7750-PE-05",
            "vendor": "Nokia",
            "family_model": "7750 SR-7s",
            "os_name": "SR OS",
            "version": "23.10.R2",
            "management_ip": "10.255.0.15",
            "management_interface": "A/1",
            "oob_available": True,
            "role": "MPLS_PE",
            "validated_for_write": True,
            "safe_commit_command": "commit confirmed 2",
            "cpu_percent": 15.2,
            "memory_percent": 38.4,
            "services_healthy": True,
            "running_config": "configure system name BR-RJ-SR7750-PE-05\nconfigure service vpls 100 customer 1 create\n",
            "interfaces": [
                {"name": "1/1/c1/1", "status": "UP", "protocol": "UP", "ip": "172.16.35.1/30", "mtu": 9216, "crc_errors": 0}
            ],
            "bgp_peers": [
                {"peer_ip": "10.0.0.3", "remote_as": 65001, "state": "Established", "uptime": "42d 08:11:00", "prefixes_received": 640}
            ],
            "ospf_neighbors": [],
            "optical_transceivers": [],
            "alarms": [],
            "syslog": [],
            "diagnostic_outputs": {"show router bgp summary all": "Neighbor 10.0.0.3 AS 65001 State Established"},
            "neighbors": [
                {"neighbor_device": "MX204-EDGE-03", "local_if": "1/1/c1/1", "remote_if": "et-0/0/0", "protocol": "LLDP/MPLS"}
            ],
            "dependent_services": ["EVPN-DATACENTER-INTERCONNECT"],
            "affected_customers_estimate": 510,
        },
        "CCR2004-CPE-06": {
            "device_id": "CCR2004-CPE-06",
            "hostname": "BR-SP-CCR2004-CPE-06",
            "vendor": "MikroTik",
            "family_model": "CCR2004-1G-12S+2XS",
            "os_name": "RouterOS 7",
            "version": "7.14.3",
            "management_ip": "10.255.0.16",
            "management_interface": "ether1",
            "oob_available": True,
            "role": "ENTERPRISE_CPE",
            "validated_for_write": True,
            "safe_commit_command": "Safe Mode + /system/backup/save",
            "cpu_percent": 11.0,
            "memory_percent": 27.0,
            "services_healthy": True,
            "running_config": (
                "/system identity set name=BR-SP-CCR2004-CPE-06\n"
                "/routing/bgp/connection add name=UPLINK-C9500 remote.address=172.16.46.1 as=65060\n"
            ),
            "interfaces": [
                {"name": "sfp-sfpplus1", "status": "UP", "protocol": "UP", "ip": "172.16.46.2/30", "mtu": 1500, "crc_errors": 0}
            ],
            "bgp_peers": [
                {"peer_ip": "172.16.46.1", "remote_as": 65002, "state": "Established", "uptime": "19d 04:12:00", "prefixes_received": 4}
            ],
            "ospf_neighbors": [],
            "optical_transceivers": [],
            "alarms": [],
            "syslog": [],
            "diagnostic_outputs": {"/routing/bgp/session/print": "0 E name=UPLINK-C9500 remote.address=172.16.46.1 remote.as=65002"},
            "neighbors": [
                {"neighbor_device": "C9500-DIST-04", "local_if": "sfp-sfpplus1", "remote_if": "TenGigabitEthernet1/0/2", "protocol": "LLDP/BGP"}
            ],
            "dependent_services": ["BRANCH-WAN-CONNECTIVITY"],
            "affected_customers_estimate": 95,
        },
    }


class NetworkTopologyGraph:
    """
    Gerencia o grafo de topologia da rede, calcula vizinhanças e avalia o impacto
    operacional (Blast Radius) antes de qualquer mudança.
    """

    def __init__(self, inventory: Dict[str, Dict[str, Any]]) -> None:
        self.inventory = inventory

    def export_topology(self) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        links: List[Dict[str, Any]] = []
        seen_links = set()

        for dev_id, dev in self.inventory.items():
            alarms = dev.get("alarms", [])
            status = "DEGRADED" if alarms else "HEALTHY"
            nodes.append(
                {
                    "id": dev_id,
                    "hostname": dev["hostname"],
                    "vendor": dev["vendor"],
                    "family_model": dev["family_model"],
                    "os_name": dev["os_name"],
                    "version": dev["version"],
                    "role": dev.get("role", "CORE"),
                    "management_ip": dev["management_ip"],
                    "status": status,
                    "alarm_count": len(alarms),
                    "alarms": alarms,
                }
            )
            for nbr in dev.get("neighbors", []):
                target = nbr["neighbor_device"]
                pair = tuple(sorted([dev_id, target]))
                link_key = (pair[0], pair[1], nbr["protocol"])
                if link_key not in seen_links:
                    seen_links.add(link_key)
                    links.append(
                        {
                            "source": dev_id,
                            "target": target,
                            "source_if": nbr["local_if"],
                            "target_if": nbr["remote_if"],
                            "protocol": nbr["protocol"],
                        }
                    )

        return {"nodes": nodes, "links": links}

    def analyze_impact(self, device_id: str, commands: List[str]) -> Dict[str, Any]:
        if device_id not in self.inventory:
            return {
                "can_evaluate": False,
                "risk_level": "BLOCKED_UNKNOWN_DEVICE",
                "reason": f"Equipamento '{device_id}' ausente da topologia verificada.",
            }

        dev = self.inventory[device_id]
        neighbors = [n["neighbor_device"] for n in dev.get("neighbors", [])]
        services = list(dev.get("dependent_services", []))
        customers = int(dev.get("affected_customers_estimate", 0))

        joined = " ".join(commands).lower()
        is_destructive = any(k in joined for k in ("shutdown", "reset", "erase", "delete /", "zeroize"))
        touches_core_routing = any(k in joined for k in ("bgp", "ospf", "route-policy", "mtu"))

        if is_destructive:
            risk = "CRITICAL"
        elif touches_core_routing and dev.get("role") in ("BACKBONE_PE", "MPLS_PE"):
            risk = "HIGH"
        elif touches_core_routing:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "can_evaluate": True,
            "device_id": device_id,
            "hostname": dev["hostname"],
            "risk_level": risk,
            "affected_neighbors": neighbors,
            "affected_services": services,
            "estimated_customers_in_blast_radius": customers,
            "redundancy_available": len(neighbors) >= 2,
            "oob_recovery_available": bool(dev.get("oob_available", True)),
        }
