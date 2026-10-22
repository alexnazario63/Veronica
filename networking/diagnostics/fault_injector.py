"""
Injetor de Falhas Sintéticas em Topologias Virtuais da Veronica (Fase 12).
Permite induzir de forma determinística e controlada anomalias de rede multi-camadas
(óptica, agregação de enlace, roteamento e protocolos de controle) sobre o Digital Twin,
recuperar a telemetria alterada, calcular o Blast Radius exato e validar o motor de
diagnóstico em 11 etapas.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Union

from networking.topology.graph import NetworkTopologyGraph, create_default_network_inventory
from networking.vendors.adapters import BaseVendorAdapter, DigitalTwinNetworkAdapter
from networking.diagnostics.engine import NetworkDiagnosticEngine
from knowledge.retrieval.rag_engine import HybridRAGEngine


class SyntheticFaultScenario(str, Enum):
    """Cenários padronizados de falha sintética da Fase 12."""
    OPTICAL_DEGRADATION = "optical_degradation"
    LACP_FLAPPING = "lacp_flapping"
    BGP_ROUTE_LEAK = "bgp_route_leak"
    MTU_MISMATCH = "mtu_mismatch"
    AREA_PARTITION = "area_partition"


@dataclass
class InjectedFaultRecord:
    """Registro de auditoria e controle de ciclo de vida de uma falha sintética injetada."""
    fault_id: str
    scenario: SyntheticFaultScenario
    target_device: str
    affected_devices: List[str]
    parameters: Dict[str, Any]
    injected_at: str
    baseline_snapshot: Dict[str, Dict[str, Any]]
    active: bool = True
    metadata: Optional[Dict[str, Any]] = None


class FaultInjector:
    """
    Injetor de Falhas Sintéticas integrado ao grafo topológico multi-camadas e ao
    motor NetworkDiagnosticEngine em 11 etapas.
    """

    def __init__(
        self,
        topology: Optional[NetworkTopologyGraph] = None,
        diagnostic_engine: Optional[NetworkDiagnosticEngine] = None,
        adapter: Optional[BaseVendorAdapter] = None,
    ) -> None:
        if topology is not None:
            self.topology = topology
        else:
            self.topology = NetworkTopologyGraph(create_default_network_inventory())

        self.inventory: Dict[str, Dict[str, Any]] = self.topology.inventory

        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = DigitalTwinNetworkAdapter(self.inventory)

        if diagnostic_engine is not None:
            self.diagnostic_engine = diagnostic_engine
        else:
            self.diagnostic_engine = NetworkDiagnosticEngine(
                adapter=self.adapter,
                topology=self.topology,
                rag=HybridRAGEngine(),
            )

        self.injected_faults: Dict[str, InjectedFaultRecord] = {}
        self._initial_inventory_snapshot = copy.deepcopy(self.inventory)

    @staticmethod
    def _generate_fault_id(scenario: str, target: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        token = hashlib.sha256(f"{scenario}:{target}:{timestamp}".encode()).hexdigest()[:8]
        return f"FLT-{scenario.upper()[:4]}-{target[:6]}-{token}"

    def _snapshot_devices(self, device_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        snapshot: Dict[str, Dict[str, Any]] = {}
        for dev_id in device_ids:
            if dev_id in self.inventory:
                snapshot[dev_id] = copy.deepcopy(self.inventory[dev_id])
        return snapshot

    # =========================================================================
    # CENÁRIO 1: DEGRADAÇÃO ÓPTICA (DOM - RX POWER ABAIXO DO LIMIAR CRÍTICO)
    # =========================================================================
    def inject_optical_degradation(
        self,
        device_id: str = "NE8000-CORE-01",
        interface: str = "10GE1/0/2",
        rx_power_dbm: float = -18.5,
        rx_low_threshold_dbm: float = -14.4,
        tx_power_dbm: float = -2.4,
        induced_crc_errors: int = 150,
    ) -> Dict[str, Any]:
        """
        Induz atenuação óptica excessiva (curvatura de fibra, cordão sujo ou defeito de laser),
        derrubando a potência de recepção abaixo do limiar de alarme e gerando erros de CRC.
        """
        if device_id not in self.inventory:
            raise KeyError(f"Equipamento '{device_id}' não encontrado no inventário.")

        fault_id = self._generate_fault_id("optical", device_id)
        snapshot = self._snapshot_devices([device_id])

        dev = self.inventory[device_id]
        optics = dev.setdefault("optical_transceivers", [])
        matched_opt = next((o for o in optics if o.get("interface") == interface), None)
        if matched_opt:
            matched_opt["rx_power_dbm"] = rx_power_dbm
            matched_opt["rx_low_threshold_dbm"] = rx_low_threshold_dbm
            matched_opt["tx_power_dbm"] = tx_power_dbm
            matched_opt["status"] = "LOW_RX_POWER_ALARM"
        else:
            optics.append(
                {
                    "interface": interface,
                    "wavelength_nm": 1310,
                    "tx_power_dbm": tx_power_dbm,
                    "rx_power_dbm": rx_power_dbm,
                    "rx_low_threshold_dbm": rx_low_threshold_dbm,
                    "status": "LOW_RX_POWER_ALARM",
                }
            )

        # Incrementa CRC e erros de entrada na interface
        for iface in dev.get("interfaces", []):
            if iface.get("name") == interface:
                iface["crc_errors"] = int(iface.get("crc_errors", 0)) + induced_crc_errors
                iface["in_errors"] = int(iface.get("in_errors", 0)) + induced_crc_errors

        alarm_msg = (
            f"OPTICAL_RX_LOW: Interface {interface} Rx Power {rx_power_dbm} dBm "
            f"below threshold ({rx_low_threshold_dbm} dBm)"
        )
        if alarm_msg not in dev.setdefault("alarms", []):
            dev["alarms"].append(alarm_msg)

        dev.setdefault("syslog", []).append(
            f"%%01OPTICAL/4/RX_POWER_LOW: {interface} Rx power ({rx_power_dbm} dBm) below threshold ({rx_low_threshold_dbm} dBm)."
        )

        dev.setdefault("diagnostic_outputs", {})[f"display transceiver interface {interface} verbose"] = (
            f"Rx Power: {rx_power_dbm:.2f} dBm (Low Alarm Threshold: {rx_low_threshold_dbm:.2f} dBm), "
            f"Tx Power: {tx_power_dbm:.2f} dBm"
        )

        record = InjectedFaultRecord(
            fault_id=fault_id,
            scenario=SyntheticFaultScenario.OPTICAL_DEGRADATION,
            target_device=device_id,
            affected_devices=[device_id],
            parameters={
                "interface": interface,
                "rx_power_dbm": rx_power_dbm,
                "rx_low_threshold_dbm": rx_low_threshold_dbm,
                "induced_crc_errors": induced_crc_errors,
            },
            injected_at=datetime.now(timezone.utc).isoformat(),
            baseline_snapshot=snapshot,
        )
        self.injected_faults[fault_id] = record

        return {
            "fault_id": fault_id,
            "scenario": record.scenario.value,
            "target_device": device_id,
            "interface": interface,
            "rx_power_dbm": rx_power_dbm,
            "threshold_dbm": rx_low_threshold_dbm,
            "active_alarm": alarm_msg,
            "injected_at": record.injected_at,
        }

    # =========================================================================
    # CENÁRIO 2: FLAPPING DE LACP / PERDA DE AGREGAÇÃO DE ENLACE
    # =========================================================================
    def inject_lacp_flapping(
        self,
        device_id: str = "NE8000-CORE-01",
        bundle_interface: str = "Eth-Trunk1",
        member_interfaces: Optional[List[str]] = None,
        flap_interval_seconds: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Induz instabilidade contínua de agregação LACP (802.3ad/802.1AX),
        provocando suspensão intermitente de portas membros, oscilação de largura de banda
        e perda de pacotes por tail-drop.
        """
        if device_id not in self.inventory:
            raise KeyError(f"Equipamento '{device_id}' não encontrado no inventário.")

        fault_id = self._generate_fault_id("lacp", device_id)
        snapshot = self._snapshot_devices([device_id])
        dev = self.inventory[device_id]

        members = member_interfaces or ["10GE1/0/1", "10GE1/0/2"]

        # Registra anomalia de fundamentos (LACP)
        dev.setdefault("fundamentals_anomalies", []).append(
            {
                "protocol": "LACP",
                "interface": bundle_interface,
                "detail": f"Portas membros {members} em flapping contínuo (intervalo={flap_interval_seconds}s) e status SUSPENDED",
            }
        )
        dev.setdefault("lacp_anomalies", []).append(
            {
                "bundle": bundle_interface,
                "member": members[0],
                "detail": f"Porta membro {members[0]} suspensa devido a timeout de LACPDU",
            }
        )

        alarm_msg = f"LACP_MEMBER_FLAPPING: Bundle {bundle_interface} member port {members[0]} flapping and suspended"
        if alarm_msg not in dev.setdefault("alarms", []):
            dev["alarms"].append(alarm_msg)

        dev.setdefault("syslog", []).append(
            f"%LACP-4-BUNDLE_DISALLOW: Port {members[0]} suspended from bundle {bundle_interface} due to LACPDU timeout."
        )

        record = InjectedFaultRecord(
            fault_id=fault_id,
            scenario=SyntheticFaultScenario.LACP_FLAPPING,
            target_device=device_id,
            affected_devices=[device_id],
            parameters={
                "bundle_interface": bundle_interface,
                "member_interfaces": members,
                "flap_interval_seconds": flap_interval_seconds,
            },
            injected_at=datetime.now(timezone.utc).isoformat(),
            baseline_snapshot=snapshot,
        )
        self.injected_faults[fault_id] = record

        return {
            "fault_id": fault_id,
            "scenario": record.scenario.value,
            "target_device": device_id,
            "bundle_interface": bundle_interface,
            "member_interfaces": members,
            "active_alarm": alarm_msg,
            "injected_at": record.injected_at,
        }

    # =========================================================================
    # CENÁRIO 3: ROUTE-LEAKING BGP / ANÚNCIO INDEVIDO DE PREFIXOS
    # =========================================================================
    def inject_bgp_route_leak(
        self,
        device_id: str = "NE8000-CORE-01",
        peer_ip: str = "10.0.0.2",
        leaked_prefix: str = "10.100.0.0/16",
        leak_type: str = "UNAUTHORIZED_TRANSIT_LEAK",
        remote_as: Optional[int] = 65002,
    ) -> Dict[str, Any]:
        """
        Induz um vazamento de rotas BGP (RFC 7908 / RFC 9234), onde prefixos internos
        ou de clientes são anunciados indevidamente para provedores de trânsito upstream,
        gerando sequestro de tráfego e risco de saturação de enlaces.
        """
        if device_id not in self.inventory:
            raise KeyError(f"Equipamento '{device_id}' não encontrado no inventário.")

        fault_id = self._generate_fault_id("bgpleak", device_id)
        snapshot = self._snapshot_devices([device_id])
        dev = self.inventory[device_id]

        dev.setdefault("routing_anomalies", []).append(
            {
                "protocol": "BGP Route Leak",
                "detail": f"Vazamento indevido de prefixo {leaked_prefix} para peer de trânsito {peer_ip} (AS {remote_as})",
            }
        )
        dev.setdefault("bgp_route_leaks", []).append(
            {
                "prefix": leaked_prefix,
                "peer_ip": peer_ip,
                "remote_as": remote_as,
                "leak_type": leak_type,
            }
        )

        alarm_msg = f"BGP_ROUTE_LEAK: Prefix {leaked_prefix} leaked to transit peer {peer_ip} without export filter"
        if alarm_msg not in dev.setdefault("alarms", []):
            dev["alarms"].append(alarm_msg)

        dev.setdefault("syslog", []).append(
            f"%%01BGP/4/ROUTE_LEAK_WARNING: Prefix {leaked_prefix} advertised to AS {remote_as} violating routing policy."
        )

        record = InjectedFaultRecord(
            fault_id=fault_id,
            scenario=SyntheticFaultScenario.BGP_ROUTE_LEAK,
            target_device=device_id,
            affected_devices=[device_id],
            parameters={
                "peer_ip": peer_ip,
                "leaked_prefix": leaked_prefix,
                "leak_type": leak_type,
                "remote_as": remote_as,
            },
            injected_at=datetime.now(timezone.utc).isoformat(),
            baseline_snapshot=snapshot,
        )
        self.injected_faults[fault_id] = record

        return {
            "fault_id": fault_id,
            "scenario": record.scenario.value,
            "target_device": device_id,
            "peer_ip": peer_ip,
            "leaked_prefix": leaked_prefix,
            "active_alarm": alarm_msg,
            "injected_at": record.injected_at,
        }

    # =========================================================================
    # CENÁRIO 4: MTU MISMATCH ENTRE ADJACÊNCIAS DE ROTEAMENTO
    # =========================================================================
    def inject_mtu_mismatch(
        self,
        device_id: str = "ASR9K-CORE-02",
        interface: str = "TenGigE0/0/0/2",
        neighbor_device: str = "C9500-DIST-04",
        local_mtu: int = 9216,
        remote_mtu: int = 1500,
        protocol: str = "OSPF",
    ) -> Dict[str, Any]:
        """
        Induz assimetria de MTU entre dois roteadores adjacentes, fazendo com que
        pacotes de controle grandes (ex.: DBD do OSPF ou LSP do IS-IS) sejam descartados,
        travando a sessão de roteamento em EXSTART/EXCHANGE.
        """
        if device_id not in self.inventory:
            raise KeyError(f"Equipamento '{device_id}' não encontrado no inventário.")
        if neighbor_device not in self.inventory:
            raise KeyError(f"Equipamento vizinho '{neighbor_device}' não encontrado no inventário.")

        fault_id = self._generate_fault_id("mtu", device_id)
        snapshot = self._snapshot_devices([device_id, neighbor_device])

        dev_local = self.inventory[device_id]
        dev_remote = self.inventory[neighbor_device]

        # Ajusta MTU local
        for iface in dev_local.get("interfaces", []):
            if iface.get("name") == interface:
                iface["mtu"] = local_mtu

        # Atualiza estado do vizinho OSPF para EXSTART
        for nbr in dev_local.get("ospf_neighbors", []):
            if nbr.get("interface") == interface:
                nbr["state"] = "EXSTART/BDR"

        for nbr in dev_remote.get("ospf_neighbors", []):
            nbr["state"] = "EXSTART/DR"

        alarm_msg = (
            f"OSPF_MTU_MISMATCH: Neighbor on {interface} stuck in EXSTART "
            f"(local MTU {local_mtu} vs remote {remote_mtu})"
        )
        if alarm_msg not in dev_local.setdefault("alarms", []):
            dev_local["alarms"].append(alarm_msg)

        dev_local.setdefault("syslog", []).append(
            f"%ROUTING-OSPF-5-ADJCHG: Neighbor on {interface} stuck in EXSTART: DBD retransmissions (MTU mismatch)."
        )

        record = InjectedFaultRecord(
            fault_id=fault_id,
            scenario=SyntheticFaultScenario.MTU_MISMATCH,
            target_device=device_id,
            affected_devices=[device_id, neighbor_device],
            parameters={
                "interface": interface,
                "neighbor_device": neighbor_device,
                "local_mtu": local_mtu,
                "remote_mtu": remote_mtu,
                "protocol": protocol,
            },
            injected_at=datetime.now(timezone.utc).isoformat(),
            baseline_snapshot=snapshot,
        )
        self.injected_faults[fault_id] = record

        return {
            "fault_id": fault_id,
            "scenario": record.scenario.value,
            "target_device": device_id,
            "neighbor_device": neighbor_device,
            "local_mtu": local_mtu,
            "remote_mtu": remote_mtu,
            "active_alarm": alarm_msg,
            "injected_at": record.injected_at,
        }

    # =========================================================================
    # CENÁRIO 5: PARTIÇÃO DE ÁREA OSPF / IS-IS
    # =========================================================================
    def inject_area_partition(
        self,
        partitioned_area: str = "0.0.0.0",
        isolated_nodes: Optional[List[str]] = None,
        protocol: str = "OSPF",
    ) -> Dict[str, Any]:
        """
        Induz a fragmentação de uma área de trânsito (ex.: Backbone Área 0), rompendo os
        enlaces de fronteira entre dois subconjuntos de roteadores e criando componentes
        desconexos sem virtual-link.
        """
        target_nodes = isolated_nodes or ["NE8000-CORE-01", "ASR9K-CORE-02"]
        snapshot = self._snapshot_devices(list(self.inventory.keys()))

        primary_device = target_nodes[0]
        fault_id = self._generate_fault_id("partition", primary_device)

        # Desativa adjacências entre os nós isolados e os nós externos
        for dev_id in target_nodes:
            if dev_id not in self.inventory:
                continue
            dev = self.inventory[dev_id]
            dev.setdefault("routing_anomalies", []).append(
                {
                    "protocol": f"{protocol} Area Partition",
                    "detail": f"Partição na Área {partitioned_area}: nós {target_nodes} isolados do restante do backbone",
                }
            )
            dev.setdefault("area_partitions", []).append(
                {
                    "area": partitioned_area,
                    "isolated_nodes": target_nodes,
                    "protocol": protocol,
                }
            )

            alarm_msg = f"AREA_PARTITION: Backbone Area {partitioned_area} partitioned, breaking intra-area transit"
            if alarm_msg not in dev.setdefault("alarms", []):
                dev["alarms"].append(alarm_msg)

            dev.setdefault("syslog", []).append(
                f"%{protocol}-4-AREA_PARTITION: Area {partitioned_area} partitioned. Inter-area transit disrupted."
            )

        record = InjectedFaultRecord(
            fault_id=fault_id,
            scenario=SyntheticFaultScenario.AREA_PARTITION,
            target_device=primary_device,
            affected_devices=list(self.inventory.keys()),
            parameters={
                "partitioned_area": partitioned_area,
                "isolated_nodes": target_nodes,
                "protocol": protocol,
            },
            injected_at=datetime.now(timezone.utc).isoformat(),
            baseline_snapshot=snapshot,
        )
        self.injected_faults[fault_id] = record

        return {
            "fault_id": fault_id,
            "scenario": record.scenario.value,
            "target_device": primary_device,
            "partitioned_area": partitioned_area,
            "isolated_nodes": target_nodes,
            "active_alarm": f"AREA_PARTITION: Area {partitioned_area} partitioned",
            "injected_at": record.injected_at,
        }

    # =========================================================================
    # DESPACHANTE GENÉRICO DE INJEÇÃO
    # =========================================================================
    def inject_fault(self, scenario: Union[str, SyntheticFaultScenario], **kwargs: Any) -> Dict[str, Any]:
        """Despacha a injeção para o método correspondente ao cenário informado."""
        scen_str = scenario.value if isinstance(scenario, SyntheticFaultScenario) else str(scenario).lower()

        if scen_str in ("optical_degradation", "optical", "dom"):
            return self.inject_optical_degradation(**kwargs)
        elif scen_str in ("lacp_flapping", "lacp"):
            return self.inject_lacp_flapping(**kwargs)
        elif scen_str in ("bgp_route_leak", "route_leak", "bgp_leak"):
            return self.inject_bgp_route_leak(**kwargs)
        elif scen_str in ("mtu_mismatch", "mtu"):
            return self.inject_mtu_mismatch(**kwargs)
        elif scen_str in ("area_partition", "partition", "ospf_partition"):
            return self.inject_area_partition(**kwargs)
        else:
            raise ValueError(f"Cenário de falha sintética desconhecido: '{scenario}'.")

    # =========================================================================
    # RECUPERAÇÃO DE TELEMETRIA ALTERADA
    # =========================================================================
    def get_altered_telemetry(self, device_id: str, fault_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Recupera a telemetria operacional atual do equipamento após a injeção de falha,
        comparando com o baseline para destacar os deltas alterados.
        """
        if device_id not in self.inventory:
            raise KeyError(f"Equipamento '{device_id}' não encontrado no inventário.")

        current = self.adapter.get_operational_state(device_id)
        current_cfg = self.adapter.get_running_config(device_id)
        current_diags = self.adapter.collect_diagnostics(device_id)

        baseline_dev = self._initial_inventory_snapshot.get(device_id, {})
        base_alarms = set(baseline_dev.get("alarms", []))
        curr_alarms = set(self.inventory[device_id].get("alarms", []))
        new_alarms = list(curr_alarms - base_alarms)

        active_flts = [
            f.fault_id for f in self.injected_faults.values()
            if f.active and device_id in f.affected_devices
        ]

        return {
            "device_id": device_id,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "active_fault_ids": active_flts,
            "operational_state": current,
            "running_config_hash": current_cfg.get("config_hash"),
            "alarms": self.inventory[device_id].get("alarms", []),
            "new_alarms_detected": new_alarms,
            "syslog_tail": self.inventory[device_id].get("syslog", [])[-3:],
            "cli_outputs": current_diags.get("cli_outputs", {}),
        }

    # =========================================================================
    # CÁLCULO EXATO DE BLAST RADIUS E GRAFO DE IMPACTO DO INCIDENTE
    # =========================================================================
    def calculate_blast_radius(self, fault_id_or_device: str) -> Dict[str, Any]:
        """
        Calcula o Blast Radius exato da falha injetada sobre a topologia multi-camadas,
        determinando nós com impacto direto e transitivo, serviços afetados, clientes
        impactados e pontos de articulação.
        """
        target_device = fault_id_or_device
        scenario_name = "manual_evaluation"
        fault_record: Optional[InjectedFaultRecord] = None

        if fault_id_or_device in self.injected_faults:
            fault_record = self.injected_faults[fault_id_or_device]
            target_device = fault_record.target_device
            scenario_name = fault_record.scenario.value

        base_blast = self.topology.calculate_blast_radius(target_device)
        dev = self.inventory.get(target_device, {})

        # Análise multi-camadas (L1, L2, L3, Serviços)
        multi_layer_impact: Dict[str, Any] = {
            "layer_1_physical_optical": {
                "impacted": scenario_name == "optical_degradation",
                "severity": "CRITICAL" if scenario_name == "optical_degradation" else "NONE",
            },
            "layer_2_datalink_lacp": {
                "impacted": scenario_name == "lacp_flapping",
                "severity": "HIGH" if scenario_name == "lacp_flapping" else "NONE",
            },
            "layer_3_network_routing": {
                "impacted": scenario_name in ("mtu_mismatch", "bgp_route_leak", "area_partition"),
                "severity": "CRITICAL" if scenario_name in ("bgp_route_leak", "area_partition") else "HIGH",
            },
            "layer_overlay_services": {
                "impacted": len(base_blast.get("affected_services", [])) > 0,
                "services": base_blast.get("affected_services", []),
            },
        }

        # Nível de risco consolidado
        if scenario_name in ("bgp_route_leak", "area_partition"):
            risk_level = "CRITICAL"
        elif scenario_name in ("mtu_mismatch", "lacp_flapping"):
            risk_level = "HIGH"
        else:
            risk_level = "MEDIUM"

        return {
            "root_device": target_device,
            "scenario": scenario_name,
            "risk_level": risk_level,
            "affected_devices": base_blast.get("affected_devices", []),
            "affected_services": base_blast.get("affected_services", []),
            "estimated_customers_impacted": base_blast.get("estimated_customers", 0),
            "articulation_devices": base_blast.get("articulation_devices", []),
            "multi_layer_impact": multi_layer_impact,
            "blast_radius_graph": {
                "nodes": base_blast.get("nodes", []),
                "edges": base_blast.get("edges", []),
            },
            "graph_backend": base_blast.get("graph_backend", "python"),
        }

    def generate_impact_graph(self, fault_id: str) -> Dict[str, Any]:
        """Gera o grafo de impacto formatado para renderização no NetCanvas."""
        blast = self.calculate_blast_radius(fault_id)
        fault = self.injected_faults.get(fault_id)

        nodes_canvas = []
        for n in blast["blast_radius_graph"]["nodes"]:
            dev_id = n["id"]
            dev_info = self.inventory.get(dev_id, {})
            alarms = dev_info.get("alarms", [])
            status = "FAILED" if dev_id == blast["root_device"] else ("DEGRADED" if alarms else "HEALTHY")
            nodes_canvas.append(
                {
                    "id": dev_id,
                    "label": dev_info.get("hostname", dev_id),
                    "vendor": dev_info.get("vendor", "Unknown"),
                    "os": dev_info.get("os_name", "Unknown"),
                    "status": status,
                    "impact": n["impact"],
                    "alarm_count": len(alarms),
                }
            )

        return {
            "fault_id": fault_id,
            "scenario": blast["scenario"],
            "risk_level": blast["risk_level"],
            "root_device": blast["root_device"],
            "nodes": nodes_canvas,
            "edges": blast["blast_radius_graph"]["edges"],
            "affected_services": blast["affected_services"],
            "estimated_customers": blast["estimated_customers_impacted"],
        }

    # =========================================================================
    # REVERSÃO E RESTAURAÇÃO DE FALHAS
    # =========================================================================
    def revert_fault(self, fault_id: str) -> Dict[str, Any]:
        """Reverte uma falha sintética específica, restaurando o snapshot dos dispositivos afetados."""
        if fault_id not in self.injected_faults:
            raise KeyError(f"Registro de falha '{fault_id}' não encontrado.")

        record = self.injected_faults[fault_id]
        if not record.active:
            return {"fault_id": fault_id, "already_reverted": True}

        # Restaura dispositivos do snapshot
        for dev_id, saved_data in record.baseline_snapshot.items():
            if dev_id in self.inventory:
                self.inventory[dev_id] = copy.deepcopy(saved_data)

        record.active = False
        return {
            "fault_id": fault_id,
            "scenario": record.scenario.value,
            "target_device": record.target_device,
            "restored_at": datetime.now(timezone.utc).isoformat(),
            "status": "RESTORED",
        }

    def restore_all(self) -> Dict[str, Any]:
        """Restaura completamente o inventário ao estado limpo inicial."""
        reverted_count = 0
        for f_id in list(self.injected_faults.keys()):
            if self.injected_faults[f_id].active:
                self.revert_fault(f_id)
                reverted_count += 1

        self.inventory.clear()
        self.inventory.update(copy.deepcopy(self._initial_inventory_snapshot))

        return {
            "restored_all": True,
            "reverted_faults_count": reverted_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # EXECUÇÃO INTEGRADA: INJEÇÃO -> TELEMETRIA -> BLAST RADIUS -> DIAGNÓSTICO
    # =========================================================================
    def inject_and_evaluate(
        self,
        scenario: Union[str, SyntheticFaultScenario],
        diagnostic_engine: Optional[NetworkDiagnosticEngine] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Executa o ciclo completo de teste e diagnóstico sintético:
        1. Injeta a falha sintética
        2. Coleta a telemetria alterada
        3. Calcula o Blast Radius exato
        4. Executa o diagnóstico investigativo em 11 etapas do NetworkDiagnosticEngine
        5. Retorna o relatório consolidado com evidências e causas-raiz
        """
        engine = diagnostic_engine or self.diagnostic_engine

        # 1. Injeta a falha
        injection_res = self.inject_fault(scenario, **kwargs)
        fault_id = injection_res["fault_id"]
        target = injection_res["target_device"]

        # 2. Recupera telemetria alterada
        telemetry = self.get_altered_telemetry(target, fault_id=fault_id)

        # 3. Calcula Blast Radius exato
        blast = self.calculate_blast_radius(fault_id)
        impact_graph = self.generate_impact_graph(fault_id)

        # 4. Formula descrição do problema para a investigação em 11 etapas
        scen_str = scenario.value if isinstance(scenario, SyntheticFaultScenario) else str(scenario).lower()
        if scen_str in ("optical_degradation", "optical"):
            prob_desc = f"Alarme de potência óptica baixa na interface de {target} com degradação de sinal e erros de CRC"
        elif scen_str in ("lacp_flapping", "lacp"):
            prob_desc = f"Flapping de agregação de enlace LACP e porta suspensa em {target}"
        elif scen_str in ("bgp_route_leak", "route_leak"):
            prob_desc = f"Vazamento indevido de prefixo BGP detectado em {target} propagando rotas de trânsito"
        elif scen_str in ("mtu_mismatch", "mtu"):
            prob_desc = f"Adjacência OSPF presa em EXSTART devido a MTU mismatch em {target}"
        elif scen_str in ("area_partition", "partition"):
            prob_desc = f"Partição de área OSPF detectada fragmentando o trânsito do backbone em {target}"
        else:
            prob_desc = f"Incidente sintético {scenario} em {target}"

        # 5. Executa os 11 passos do motor de diagnóstico
        diag_report = engine.investigate(
            problem_description=prob_desc,
            target_devices=[target] + blast.get("affected_devices", [])[:2],
        )

        return {
            "fault_id": fault_id,
            "scenario": scen_str,
            "target_device": target,
            "injection_details": injection_res,
            "altered_telemetry": telemetry,
            "blast_radius": blast,
            "impact_graph": impact_graph,
            "diagnostic_investigation_11_steps": diag_report,
            "incident_verified": len(diag_report.get("root_causes", [])) > 0,
        }


# Alias semântico conforme especificação
SyntheticFaultInjector = FaultInjector
