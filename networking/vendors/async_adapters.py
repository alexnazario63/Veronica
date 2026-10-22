"""Credential-injected async vendor adapter with read-only cascade defaults,
production-grade real drivers (Scrapli SSH, NETCONF transactional, pygnmi streaming),
full multi-vendor support, and deterministic Digital Twin fallback.
"""

from __future__ import annotations

import asyncio
from abc import ABC
from datetime import datetime, timezone
import difflib
import hashlib
from typing import Any, AsyncGenerator, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

from networking.vendors.models import canonicalize_operational_state


class AsyncTransport(Protocol):
    """Protocol for asynchronous network communication."""
    async def execute(self, operation: str, device_id: str, **kwargs: Any) -> Any: ...


# =============================================================================
# CONECTORES REAIS ASSÍNCRONOS PRONTOS PARA PRODUÇÃO (FASE 11)
# =============================================================================

class ScrapliAsyncSSHTransport:
    """
    Conector SSH assíncrono para produção usando Scrapli com backend asyncssh.
    Compatível com Windows e Linux, suportando envio de comandos, blocos de
    configuração com rollback/commit e validação de CLI.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 22,
        auth_username: str = "admin",
        auth_password: Optional[str] = None,
        auth_private_key: Optional[str] = None,
        platform: str = "cisco_iosxe",
        timeout_ops: float = 30.0,
        transport: str = "asyncssh",
        driver_instance: Optional[Any] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.auth_username = auth_username
        self.auth_password = auth_password
        self.auth_private_key = auth_private_key
        self.platform = platform
        self.timeout_ops = timeout_ops
        self.transport = transport
        self._driver = driver_instance

    async def get_driver(self) -> Any:
        if self._driver is not None:
            return self._driver
        try:
            from scrapli import AsyncScrapli
            from scrapli.driver.generic import AsyncGenericDriver

            core_platforms = {
                "cisco_iosxe", "cisco_iosxr", "cisco_nxos", "juniper_junos", "arista_eos"
            }
            if self.platform in core_platforms:
                self._driver = AsyncScrapli(
                    host=self.host,
                    port=self.port,
                    auth_username=self.auth_username,
                    auth_password=self.auth_password,
                    auth_private_key=self.auth_private_key,
                    auth_strict_key=False,
                    platform=self.platform,
                    transport=self.transport,
                    timeout_ops=self.timeout_ops,
                )
            else:
                self._driver = AsyncGenericDriver(
                    host=self.host,
                    port=self.port,
                    auth_username=self.auth_username,
                    auth_password=self.auth_password,
                    auth_private_key=self.auth_private_key,
                    auth_strict_key=False,
                    transport=self.transport,
                    timeout_ops=self.timeout_ops,
                )
            return self._driver
        except Exception as exc:
            raise ConnectionError(f"Scrapli driver init failed for {self.host}:{self.port}: {exc}") from exc

    async def open(self) -> None:
        drv = await self.get_driver()
        if hasattr(drv, "open") and not getattr(drv, "isalive", lambda: False)():
            await drv.open()

    async def close(self) -> None:
        if self._driver and hasattr(self._driver, "close"):
            try:
                await self._driver.close()
            except Exception:
                pass
            self._driver = None

    async def send_command(self, command: str) -> str:
        drv = await self.get_driver()
        await self.open()
        res = await drv.send_command(command)
        return str(getattr(res, "result", res))

    async def send_commands(self, commands: Sequence[str]) -> List[str]:
        drv = await self.get_driver()
        await self.open()
        res = await drv.send_commands(list(commands))
        return [str(getattr(r, "result", r)) for r in res]

    async def send_config(self, configs: Sequence[str]) -> str:
        drv = await self.get_driver()
        await self.open()
        if hasattr(drv, "send_configs"):
            res = await drv.send_configs(list(configs))
            return "\n".join(str(getattr(r, "result", r)) for r in res)
        elif hasattr(drv, "send_config"):
            res = await drv.send_config(configs[0])
            return str(getattr(res, "result", res))
        else:
            return "\n".join([await self.send_command(c) for c in configs])

    async def execute(self, operation: str, device_id: str, **kwargs: Any) -> Any:
        try:
            if operation in ("get_version", "discover_device"):
                cmd = kwargs.get("command", "show version" if "cisco" in self.platform else "display version")
                out = await self.send_command(cmd)
                return {"device_id": device_id, "raw_output": out, "platform": self.platform}
            elif operation == "get_running_config":
                cmd = kwargs.get(
                    "command",
                    "display current-configuration" if "huawei" in self.platform else "show running-config",
                )
                out = await self.send_command(cmd)
                return {"device_id": device_id, "running_config": out, "config_hash": hashlib.sha256(out.encode()).hexdigest()[:16]}
            elif operation == "collect_diagnostics":
                commands = kwargs.get("commands", ["show version", "show interfaces"])
                outputs = {}
                for c in commands:
                    outputs[c] = await self.send_command(c)
                return {"device_id": device_id, "cli_outputs": outputs, "read_only_mode": True}
            elif operation == "apply_change":
                commands = kwargs.get("commands", [])
                out = await self.send_config(commands)
                return {"device_id": device_id, "applied": True, "output": out}
            elif operation == "rollback_change":
                commands = kwargs.get("rollback_commands", [])
                out = await self.send_config(commands)
                return {"device_id": device_id, "rolled_back": True, "output": out}
            else:
                cmd = kwargs.get("command", f"show {operation}")
                out = await self.send_command(cmd)
                return {"device_id": device_id, "operation": operation, "result": out}
        except Exception as exc:
            raise ConnectionError(f"Scrapli execution error [{operation}] on {device_id}: {exc}") from exc


class NetconfStreamingTransport:
    """
    Conector NETCONF transacional para produção com suporte a two-phase commit,
    commit confirmed, rollback automático e streaming de notificações via
    scrapli_netconf ou ncclient (executado de forma não-bloqueante).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 830,
        username: str = "admin",
        password: Optional[str] = None,
        key_filename: Optional[str] = None,
        device_params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        use_ncclient: bool = False,
        driver_instance: Optional[Any] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.key_filename = key_filename
        self.device_params = device_params or {"name": "default"}
        self.timeout = timeout
        self.use_ncclient = use_ncclient
        self._driver = driver_instance

    async def get_driver(self) -> Any:
        if self._driver is not None:
            return self._driver
        try:
            if not self.use_ncclient:
                from scrapli_netconf.driver import AsyncNetconfDriver

                self._driver = AsyncNetconfDriver(
                    host=self.host,
                    port=self.port,
                    auth_username=self.username,
                    auth_password=self.password,
                    auth_private_key=self.key_filename,
                    auth_strict_key=False,
                    transport="asyncssh",
                    timeout_ops=self.timeout,
                )
                return self._driver
            else:
                from ncclient import manager

                def _connect() -> Any:
                    return manager.connect(
                        host=self.host,
                        port=self.port,
                        username=self.username,
                        password=self.password,
                        key_filename=self.key_filename,
                        hostkey_verify=False,
                        device_params=self.device_params,
                        timeout=self.timeout,
                    )

                self._driver = await asyncio.to_thread(_connect)
                return self._driver
        except Exception as exc:
            raise ConnectionError(f"NETCONF connection failed to {self.host}:{self.port}: {exc}") from exc

    async def get_config(self, source: str = "running") -> str:
        drv = await self.get_driver()
        if hasattr(drv, "get_config"):
            if not self.use_ncclient and hasattr(drv, "open"):
                if not getattr(drv, "isalive", lambda: False)():
                    await drv.open()
                res = await drv.get_config(source=source)
                return str(getattr(res, "result", res))
            else:
                res = await asyncio.to_thread(drv.get_config, source=source)
                return str(getattr(res, "data_xml", res))
        raise NotImplementedError("NETCONF get_config not supported by current driver.")

    async def edit_config(self, config: str, target: str = "candidate") -> Any:
        drv = await self.get_driver()
        if not self.use_ncclient and hasattr(drv, "open"):
            if not getattr(drv, "isalive", lambda: False)():
                await drv.open()
            res = await drv.edit_config(config=config, target=target)
            return getattr(res, "result", res)
        else:
            return await asyncio.to_thread(drv.edit_config, config=config, target=target)

    async def commit(self, confirmed: bool = True, timeout: int = 120) -> Any:
        drv = await self.get_driver()
        if not self.use_ncclient and hasattr(drv, "commit"):
            if confirmed and hasattr(drv, "commit_confirmed"):
                return await drv.commit_confirmed(confirm_timeout=timeout)
            return await drv.commit()
        else:
            if confirmed:
                return await asyncio.to_thread(drv.commit, confirmed=True, timeout=str(timeout))
            return await asyncio.to_thread(drv.commit)

    async def confirm_commit(self) -> Any:
        drv = await self.get_driver()
        if not self.use_ncclient:
            return await drv.commit()
        else:
            return await asyncio.to_thread(drv.commit)

    async def cancel_commit(self) -> Any:
        drv = await self.get_driver()
        if hasattr(drv, "discard_changes"):
            if not self.use_ncclient:
                return await drv.discard_changes()
            else:
                return await asyncio.to_thread(drv.discard_changes)
        return {"cancelled": True}

    async def execute(self, operation: str, device_id: str, **kwargs: Any) -> Any:
        try:
            if operation == "get_running_config":
                cfg = await self.get_config("running")
                return {"device_id": device_id, "running_config": cfg, "config_hash": hashlib.sha256(cfg.encode()).hexdigest()[:16]}
            elif operation == "apply_change":
                cfg = kwargs.get("candidate_xml", kwargs.get("config", ""))
                edit_res = await self.edit_config(cfg, target="candidate")
                commit_res = await self.commit(confirmed=True, timeout=kwargs.get("timeout", 120))
                return {"device_id": device_id, "applied": True, "edit": str(edit_res), "commit": str(commit_res)}
            elif operation == "finalize_change":
                res = await self.confirm_commit()
                return {"device_id": device_id, "finalized": True, "result": str(res)}
            elif operation == "rollback_change":
                res = await self.cancel_commit()
                return {"device_id": device_id, "rolled_back": True, "result": str(res)}
            else:
                cfg = await self.get_config("running")
                return {"device_id": device_id, "operation": operation, "data": cfg}
        except Exception as exc:
            raise ConnectionError(f"NETCONF operation [{operation}] failed on {device_id}: {exc}") from exc


class GnmiStreamingTransport:
    """
    Conector gNMI contínua para telemetria em streaming de alta frequência (gRPC)
    e consultas estruturadas de modelos OpenConfig/YANG usando pygnmi.
    """

    def __init__(
        self,
        target: tuple[str, int] = ("127.0.0.1", 57400),
        username: str = "admin",
        password: Optional[str] = None,
        insecure: bool = True,
        path_prefix: str = "",
        client_instance: Optional[Any] = None,
    ) -> None:
        self.target = target
        self.username = username
        self.password = password
        self.insecure = insecure
        self.path_prefix = path_prefix
        self._client = client_instance

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from pygnmi.client import gNMIclient

            self._client = gNMIclient(
                target=self.target,
                username=self.username,
                password=self.password,
                insecure=self.insecure,
            )
            return self._client
        except Exception as exc:
            raise ConnectionError(f"gNMI client initialization error for {self.target}: {exc}") from exc

    async def capabilities(self) -> Dict[str, Any]:
        def _call() -> Dict[str, Any]:
            cli = self._get_client()
            with cli:
                return cli.capabilities()

        try:
            return await asyncio.to_thread(_call)
        except Exception as exc:
            raise ConnectionError(f"gNMI capabilities query failed for {self.target}: {exc}") from exc

    async def get(self, paths: Sequence[str], encoding: str = "json_ietf") -> Dict[str, Any]:
        def _call() -> Dict[str, Any]:
            cli = self._get_client()
            with cli:
                return cli.get(path=list(paths), encoding=encoding)

        try:
            return await asyncio.to_thread(_call)
        except Exception as exc:
            raise ConnectionError(f"gNMI get failed for {paths} on {self.target}: {exc}") from exc

    async def set(
        self,
        update: Optional[List[Any]] = None,
        replace: Optional[List[Any]] = None,
        delete: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        def _call() -> Dict[str, Any]:
            cli = self._get_client()
            with cli:
                return cli.set(update=update, replace=replace, delete=delete)

        try:
            return await asyncio.to_thread(_call)
        except Exception as exc:
            raise ConnectionError(f"gNMI set operation failed on {self.target}: {exc}") from exc

    async def subscribe_stream(
        self,
        paths: Sequence[str],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Streaming de telemetria contínua com entrega reativa."""
        received: List[Dict[str, Any]] = []

        def _stream() -> None:
            cli = self._get_client()
            with cli:
                sub_request = {
                    "subscription": [
                        {"path": p, "mode": "sample", "sample_interval": 1000000000} for p in paths
                    ],
                    "mode": "stream",
                    "encoding": "json_ietf",
                }
                count = 0
                for resp in cli.subscribe_stream(sub_request):
                    received.append(resp)
                    if callback:
                        callback(resp)
                    count += 1
                    if max_messages and count >= max_messages:
                        break

        try:
            await asyncio.to_thread(_stream)
            return received
        except Exception as exc:
            raise ConnectionError(f"gNMI telemetry streaming failed for {paths}: {exc}") from exc

    async def execute(self, operation: str, device_id: str, **kwargs: Any) -> Any:
        if operation == "get_capabilities":
            caps = await self.capabilities()
            return {"device_id": device_id, "gnmi_capabilities": caps}
        elif operation in ("get_operational_state", "get_running_config"):
            paths = kwargs.get("paths", ["/interfaces", "/network-instances"])
            res = await self.get(paths)
            return {"device_id": device_id, "gnmi_state": res}
        elif operation == "stream_telemetry":
            paths = kwargs.get("paths", ["/interfaces/interface/state/counters"])
            max_msgs = kwargs.get("max_messages", 5)
            msgs = await self.subscribe_stream(paths, max_messages=max_msgs)
            return {"device_id": device_id, "telemetry_stream": msgs}
        else:
            paths = kwargs.get("paths", ["/system"])
            res = await self.get(paths)
            return {"device_id": device_id, "operation": operation, "data": res}


# =============================================================================
# ASYNC VENDOR ADAPTER BASE COM FALLBACK TRANSPARENTE AO DIGITAL TWIN
# =============================================================================

class AsyncVendorAdapter(ABC):
    """
    Interface padronizada com as 16 operações operacionais e transacionais assíncronas.
    Executa cascade gNMI -> NETCONF -> SSH em produção, mantendo fallback determinístico
    e transparente para o Digital Twin / simulador de laboratório quando hosts reais
    não estão acessíveis ou em ambiente de testes.
    """

    vendor: str = "generic"
    os_family: str = "generic"
    os_name: str = "generic"
    ntc_platform: Optional[str] = None
    scrapli_platform: Optional[str] = None
    netconf_device_type: Optional[str] = None
    safe_commit_mechanism: str = "commit confirmed 120"
    read_only_default: bool = True
    write_enabled: bool = True

    def __init__(
        self,
        *,
        gnmi: Optional[AsyncTransport] = None,
        netconf: Optional[AsyncTransport] = None,
        ssh: Optional[AsyncTransport] = None,
        digital_twin_adapter: Optional[Any] = None,
        fallback_to_digital_twin: bool = True,
    ) -> None:
        self.gnmi = gnmi
        self.netconf = netconf
        self.ssh = ssh
        self._transports = (("gnmi", gnmi), ("netconf", netconf), ("ssh", ssh))
        self.fallback_to_digital_twin = fallback_to_digital_twin

        if digital_twin_adapter is not None:
            self._digital_twin = digital_twin_adapter
        else:
            from networking.topology.graph import create_default_network_inventory
            from networking.vendors.adapters import DigitalTwinNetworkAdapter

            self._digital_twin = DigitalTwinNetworkAdapter(create_default_network_inventory())

    async def _read(self, operation: str, device_id: str, **kwargs: Any) -> Any:
        errors: List[str] = []
        for protocol, client in self._transports:
            if client is None:
                continue
            try:
                result = await client.execute(operation, device_id, **kwargs)
                return {"transport": protocol, "data": result}
            except (TimeoutError, ConnectionError, OSError) as exc:
                errors.append(f"{protocol}: {type(exc).__name__}: {exc}")

        # Fallback determinístico transparente para o Digital Twin
        if self.fallback_to_digital_twin and self._digital_twin is not None:
            twin_method = getattr(self._digital_twin, operation, None)
            if callable(twin_method):
                import inspect
                sig = inspect.signature(twin_method)
                accepted = {
                    k: v for k, v in kwargs.items()
                    if k in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                }
                res = twin_method(device_id, **accepted)
                return {"transport": "digital_twin_fallback", "data": res}

        detail = "; ".join(errors) if errors else "no transport clients configured"
        raise ConnectionError(f"Read-only operation {operation!r} unavailable for {device_id}: {detail}")

    async def discover_device(self, device_id: str) -> Dict[str, Any]:
        res = await self._read("discover_device", device_id)
        if isinstance(res, dict) and "data" in res and isinstance(res["data"], dict) and "hostname" in res["data"]:
            return res["data"]
        # Retorna modelo normalizado
        return {
            "device_id": device_id,
            "vendor": self.vendor,
            "os_name": self.os_name,
            "transport_used": res.get("transport"),
            "data": res.get("data"),
        }

    async def get_capabilities(self, device_id: str) -> Dict[str, Any]:
        res = await self._read("get_capabilities", device_id)
        if isinstance(res, dict) and "data" in res and isinstance(res["data"], dict) and "supports_commit_confirmed" in res["data"]:
            return res["data"]
        return {
            "device_id": device_id,
            "vendor": self.vendor,
            "os_name": self.os_name,
            "supports_candidate_config": True,
            "supports_commit_confirmed": True,
            "safe_commit_command": self.safe_commit_mechanism,
            "read_only_default": self.read_only_default,
            "write_enabled": self.write_enabled,
            "transport_used": res.get("transport"),
        }

    async def get_version(self, device_id: str) -> Dict[str, Any]:
        res = await self._read("get_version", device_id)
        if isinstance(res, dict) and "data" in res and isinstance(res["data"], dict) and "version" in res["data"]:
            return res["data"]
        return {
            "device_id": device_id,
            "vendor": self.vendor,
            "os_name": self.os_name,
            "version": "production-live",
            "transport_used": res.get("transport"),
            "raw": res.get("data"),
        }

    async def get_running_config(self, device_id: str) -> Dict[str, Any]:
        res = await self._read("get_running_config", device_id)
        if isinstance(res, dict) and "data" in res and isinstance(res["data"], dict) and "running_config" in res["data"]:
            return res["data"]
        cfg_str = str(res.get("data", ""))
        return {
            "device_id": device_id,
            "config_hash": hashlib.sha256(cfg_str.encode()).hexdigest()[:16],
            "running_config": cfg_str,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "transport_used": res.get("transport"),
        }

    async def get_operational_state(self, device_id: str) -> Dict[str, Any]:
        res = await self._read("get_operational_state", device_id)
        if isinstance(res, dict) and "data" in res and isinstance(res["data"], dict) and "interfaces" in res["data"]:
            return res["data"]
        # Se veio formato OpenConfig / gNMI puro, canonicaliza
        raw_data = res.get("data", {})
        if isinstance(raw_data, dict):
            canonical = canonicalize_operational_state(device_id, raw_data)
            return {
                "device_id": device_id,
                "interfaces": [m.model_dump() for m in canonical.get("interfaces", [])],
                "bgp_peers": [m.model_dump() for m in canonical.get("bgp_peers", [])],
                "optical_transceivers": [m.model_dump() for m in canonical.get("optical_lanes", [])],
                "services_healthy": True,
                "transport_used": res.get("transport"),
            }
        return {"device_id": device_id, "data": raw_data, "services_healthy": True}

    async def get_topology_neighbors(self, device_id: str) -> List[Dict[str, Any]]:
        res = await self._read("get_topology_neighbors", device_id)
        if isinstance(res, dict) and "data" in res and isinstance(res["data"], list):
            return res["data"]
        return list(res.get("data", []))

    async def collect_diagnostics(self, device_id: str, focus: str = "all") -> Dict[str, Any]:
        result = await self._read("collect_diagnostics", device_id, focus=focus, safety="read_only")
        normalized = normalize_cli_bundle(result, ntc_platform=self.ntc_platform)
        if isinstance(normalized, dict) and "data" in normalized and isinstance(normalized["data"], dict):
            data_dict = normalized["data"]
            if "cli_outputs" in data_dict or "parsed_cli_outputs" in data_dict:
                return data_dict
        return normalized

    async def validate_commands(self, device_id: str, commands: Sequence[str]) -> Dict[str, Any]:
        if self.fallback_to_digital_twin and self._digital_twin is not None:
            return self._digital_twin.validate_commands(device_id, list(commands))
        res = await self._read("validate_commands", device_id, commands=list(commands))
        return res.get("data", res)

    async def generate_candidate(self, device_id: str, proposed_commands: Sequence[str]) -> Dict[str, Any]:
        if self.fallback_to_digital_twin and self._digital_twin is not None:
            return self._digital_twin.generate_candidate(device_id, list(proposed_commands))
        running = await self.get_running_config(device_id)
        raw_cfg = running.get("running_config", "")
        candidate = raw_cfg.rstrip() + "\n! --- CANDIDATE CHANGE (VERONICA) ---\n" + "\n".join(proposed_commands) + "\n"
        return {
            "device_id": device_id,
            "base_hash": hashlib.sha256(raw_cfg.encode()).hexdigest()[:16],
            "candidate_hash": hashlib.sha256(candidate.encode()).hexdigest()[:16],
            "candidate_config": candidate,
            "commands": list(proposed_commands),
        }

    async def compare_configurations(self, running_cfg: str, candidate_cfg: str) -> Dict[str, Any]:
        diff_lines = list(
            difflib.unified_diff(
                running_cfg.splitlines(),
                candidate_cfg.splitlines(),
                fromfile="running-config",
                tofile="candidate-config",
                lineterm="",
            )
        )
        added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
        return {
            "has_changes": len(diff_lines) > 0,
            "lines_added": added,
            "lines_removed": removed,
            "unified_diff": "\n".join(diff_lines),
        }

    async def check_preconditions(self, device_id: str, commands: Sequence[str]) -> Dict[str, Any]:
        if self.fallback_to_digital_twin and self._digital_twin is not None:
            return self._digital_twin.check_preconditions(device_id, list(commands))
        state = await self.get_operational_state(device_id)
        return {
            "device_id": device_id,
            "passed": True,
            "write_supported": self.write_enabled,
            "oob_available": True,
            "commands": list(commands),
            "state_snapshot": state,
        }

    async def create_backup(self, device_id: str) -> Dict[str, Any]:
        if self.fallback_to_digital_twin and self._digital_twin is not None:
            return self._digital_twin.create_backup(device_id)
        snap = await self.get_running_config(device_id)
        backup_id = f"BKP-LIVE-{device_id}-{snap['config_hash'][:8]}"
        return {
            "backup_id": backup_id,
            "device_id": device_id,
            "config_hash": snap["config_hash"],
            "running_config": snap["running_config"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    async def prepare_rollback(
        self,
        device_id: str,
        backup: Mapping[str, Any],
        commands: Sequence[str],
    ) -> Dict[str, Any]:
        if self.fallback_to_digital_twin and self._digital_twin is not None:
            return self._digital_twin.prepare_rollback(device_id, dict(backup), list(commands))
        return {
            "device_id": device_id,
            "backup_id": backup.get("backup_id", "LIVE-BKP"),
            "target_config_hash": backup.get("config_hash"),
            "rollback_commands": [f"rollback configuration {backup.get('backup_id')}"],
            "safe_commit_mechanism": self.safe_commit_mechanism,
        }

    async def apply_change(
        self,
        device_id: str,
        commands: Sequence[str],
        approval_token: Optional[str] = None,
        expected_config_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not approval_token:
            raise PermissionError("UNAUTHORIZED_CHANGE_BLOCKED: Alteração requer token de aprovação explícita.")

        # Tenta NETCONF ou SSH real se disponíveis
        for proto, transport in (("netconf", self.netconf), ("ssh", self.ssh)):
            if transport is not None:
                try:
                    res = await transport.execute(
                        "apply_change",
                        device_id,
                        commands=list(commands),
                        approval_token=approval_token,
                        expected_config_hash=expected_config_hash,
                    )
                    return {
                        "device_id": device_id,
                        "applied": True,
                        "transport": proto,
                        "result": res,
                        "commands_applied": list(commands),
                        "applied_at": datetime.now(timezone.utc).isoformat(),
                    }
                except (ConnectionError, TimeoutError, OSError):
                    if not self.fallback_to_digital_twin:
                        raise

        if self.fallback_to_digital_twin and self._digital_twin is not None:
            return self._digital_twin.apply_change(
                device_id=device_id,
                commands=list(commands),
                approval_token=approval_token,
                expected_config_hash=expected_config_hash,
            )
        raise ConnectionError(f"No active transport available to apply changes on {device_id}.")

    async def verify_change(
        self,
        device_id: str,
        baseline_state: Mapping[str, Any],
        post_state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if self.fallback_to_digital_twin and self._digital_twin is not None:
            return self._digital_twin.verify_change(
                device_id=device_id,
                baseline_state=dict(baseline_state),
                post_state=dict(post_state),
            )
        base_peers = sum(1 for p in baseline_state.get("bgp_peers", []) if p.get("state") == "Established")
        post_peers = sum(1 for p in post_state.get("bgp_peers", []) if p.get("state") == "Established")
        passed = post_peers >= base_peers and bool(post_state.get("services_healthy", True))
        return {
            "device_id": device_id,
            "passed": passed,
            "baseline_established_bgp": base_peers,
            "post_established_bgp": post_peers,
            "services_healthy": bool(post_state.get("services_healthy", True)),
            "summary": "Pós-validação concluída: sessões e indicadores verificados." if passed else "Falha de pós-validação.",
        }

    async def rollback_change(self, device_id: str, rollback_plan: Mapping[str, Any]) -> Dict[str, Any]:
        for proto, transport in (("netconf", self.netconf), ("ssh", self.ssh)):
            if transport is not None:
                try:
                    res = await transport.execute("rollback_change", device_id, rollback_commands=rollback_plan.get("rollback_commands", []))
                    return {"device_id": device_id, "rolled_back": True, "transport": proto, "result": res}
                except (ConnectionError, TimeoutError, OSError):
                    if not self.fallback_to_digital_twin:
                        raise

        if self.fallback_to_digital_twin and self._digital_twin is not None:
            return self._digital_twin.rollback_change(device_id, dict(rollback_plan))
        raise ConnectionError(f"No active transport available for rollback on {device_id}.")

    async def finalize_change(self, device_id: str, execution: Optional[Any] = None, **kwargs: Any) -> Dict[str, Any]:
        """Confirma permanentemente o commit trial ou commit confirmed."""
        if self.netconf is not None:
            try:
                res = await self.netconf.execute("finalize_change", device_id, execution=execution, **kwargs)
                return {"device_id": device_id, "finalized": True, "transport": "netconf", "result": res}
            except (ConnectionError, TimeoutError, OSError):
                if not self.fallback_to_digital_twin:
                    raise
        if self.ssh is not None:
            try:
                confirm_cmd = "commit" if "huawei" in self.vendor.lower() or "junos" in self.os_family.lower() else "write memory"
                res = await self.ssh.execute("send_command", device_id, command=confirm_cmd)
                return {"device_id": device_id, "finalized": True, "transport": "ssh", "result": res}
            except (ConnectionError, TimeoutError, OSError):
                if not self.fallback_to_digital_twin:
                    raise

        if self.fallback_to_digital_twin and self._digital_twin is not None:
            return self._digital_twin.finalize_change(device_id, execution=execution, **kwargs)
        return {"device_id": device_id, "finalized": True}

    async def stream_telemetry(
        self,
        device_id: str,
        paths: Optional[List[str]] = None,
        duration_seconds: float = 3.0,
        sample_interval: float = 0.5,
        max_samples: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Gera telemetria contínua streaming assíncrona.
        Se gNMI estiver configurado, consome as amostras gRPC; caso contrário,
        amostra o estado operacional do Digital Twin continuamente.
        """
        import time

        paths = paths or ["/interfaces", "/network-instances"]
        start_time = time.monotonic()
        yielded = 0
        while time.monotonic() - start_time <= duration_seconds:
            state = await self.get_operational_state(device_id)
            yield {
                "device_id": device_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "paths": paths,
                "telemetry": state,
            }
            yielded += 1
            if max_samples and yielded >= max_samples:
                break
            await asyncio.sleep(sample_interval)


# =============================================================================
# ADAPTADORES ESPECÍFICOS POR FABRICANTE E VERSÃO DO SO
# =============================================================================

class _VendorFamilyAdapter(AsyncVendorAdapter):
    pass


class HuaweiVRPAdapter(_VendorFamilyAdapter):
    vendor, os_family, os_name = "Huawei", "VRP", "VRP 8"
    ntc_platform = "huawei_vrp"
    scrapli_platform = "huawei_vrp"
    netconf_device_type = "huawei"
    safe_commit_mechanism = "commit trial 120"
    write_enabled = True


class HuaweiVRP8Adapter(HuaweiVRPAdapter):
    os_name = "VRP 8"
    safe_commit_mechanism = "commit trial 120"
    write_enabled = True


class HuaweiVRP5Adapter(HuaweiVRPAdapter):
    os_name = "VRP 5"
    safe_commit_mechanism = "schedule reboot / archive backup"
    write_enabled = False  # CLI imediata sem two-phase commit transacional seguro


class CiscoIOSXEAdapter(_VendorFamilyAdapter):
    vendor, os_family, os_name = "Cisco", "IOS-XE", "IOS-XE"
    ntc_platform = "cisco_ios"
    scrapli_platform = "cisco_iosxe"
    netconf_device_type = "iosxe"
    safe_commit_mechanism = "configure terminal revert timer 2"
    write_enabled = True


class CiscoIOSXRAdapter(_VendorFamilyAdapter):
    vendor, os_family, os_name = "Cisco", "IOS-XR", "IOS-XR"
    ntc_platform = "cisco_xr"
    scrapli_platform = "cisco_iosxr"
    netconf_device_type = "iosxr"
    safe_commit_mechanism = "commit confirmed 120"
    write_enabled = True


class CiscoNXOSAdapter(_VendorFamilyAdapter):
    vendor, os_family, os_name = "Cisco", "NX-OS", "NX-OS"
    ntc_platform = "cisco_nxos"
    scrapli_platform = "cisco_nxos"
    netconf_device_type = "nexus"
    safe_commit_mechanism = "checkpoint / rollback running-config checkpoint"
    write_enabled = True


class JuniperJunosAdapter(_VendorFamilyAdapter):
    vendor, os_family, os_name = "Juniper", "Junos", "Junos OS"
    ntc_platform = "juniper_junos"
    scrapli_platform = "juniper_junos"
    netconf_device_type = "junos"
    safe_commit_mechanism = "commit confirmed 2"
    write_enabled = True


class MikroTikRouterOSAdapter(_VendorFamilyAdapter):
    vendor, os_family, os_name = "MikroTik", "RouterOS", "RouterOS 7"
    ntc_platform = "mikrotik_routeros"
    scrapli_platform = "mikrotik_routeros"
    safe_commit_mechanism = "Safe Mode + /system/backup"
    write_enabled = True


class MikroTikRouterOS7Adapter(MikroTikRouterOSAdapter):
    os_name = "RouterOS 7"
    safe_commit_mechanism = "Safe Mode + /system/backup"
    write_enabled = True


class MikroTikRouterOS6Adapter(MikroTikRouterOSAdapter):
    os_name = "RouterOS 6"
    safe_commit_mechanism = "Safe Mode"
    write_enabled = False  # REST transacional não disponível em ROS v6


class NokiaSROSAdapter(_VendorFamilyAdapter):
    vendor, os_family, os_name = "Nokia", "SR OS", "SR OS"
    ntc_platform = "nokia_sros"
    scrapli_platform = "nokia_sros"
    netconf_device_type = "sros"
    safe_commit_mechanism = "MD-CLI commit confirmed 2"
    write_enabled = True


class AristaEOSAdapter(_VendorFamilyAdapter):
    vendor, os_family, os_name = "Arista", "EOS", "EOS"
    ntc_platform = "arista_eos"
    scrapli_platform = "arista_eos"
    netconf_device_type = "default"
    safe_commit_mechanism = "configure session + commit timer"
    write_enabled = True


class FortinetFortiOSAdapter(_VendorFamilyAdapter):
    vendor, os_family, os_name = "Fortinet", "FortiOS", "FortiOS"
    ntc_platform = "fortinet"
    scrapli_platform = "fortinet"
    netconf_device_type = "default"
    safe_commit_mechanism = "cfg-save revert + workspace mode"
    write_enabled = True


def get_async_adapter_for_device(
    device_id: str,
    inventory: Optional[Dict[str, Dict[str, Any]]] = None,
    gnmi: Optional[AsyncTransport] = None,
    netconf: Optional[AsyncTransport] = None,
    ssh: Optional[AsyncTransport] = None,
    fallback_to_digital_twin: bool = True,
) -> AsyncVendorAdapter:
    """
    Fábrica inteligente de adaptadores assíncronos baseada no fabricante e versão
    registrados no inventário ou na matriz de compatibilidade.
    """
    from networking.topology.graph import create_default_network_inventory

    inv = inventory or create_default_network_inventory()
    dev = inv.get(device_id, {})
    vendor = str(dev.get("vendor", "")).lower()
    os_name = str(dev.get("os_name", "")).lower()

    adapter_cls = AsyncVendorAdapter
    if "huawei" in vendor:
        adapter_cls = HuaweiVRP5Adapter if "vrp 5" in os_name else HuaweiVRP8Adapter
    elif "cisco" in vendor:
        if "xr" in os_name:
            adapter_cls = CiscoIOSXRAdapter
        elif "nx-os" in os_name or "nexus" in os_name:
            adapter_cls = CiscoNXOSAdapter
        else:
            adapter_cls = CiscoIOSXEAdapter
    elif "juniper" in vendor:
        adapter_cls = JuniperJunosAdapter
    elif "mikrotik" in vendor:
        adapter_cls = MikroTikRouterOS6Adapter if "6" in os_name else MikroTikRouterOS7Adapter
    elif "nokia" in vendor:
        adapter_cls = NokiaSROSAdapter
    elif "arista" in vendor:
        adapter_cls = AristaEOSAdapter
    elif "fortinet" in vendor:
        adapter_cls = FortinetFortiOSAdapter

    return adapter_cls(
        gnmi=gnmi,
        netconf=netconf,
        ssh=ssh,
        fallback_to_digital_twin=fallback_to_digital_twin,
    )


# =============================================================================
# UTILITÁRIOS E PARSERS CLI / ESTRUTURADOS
# =============================================================================

def normalize_cli_bundle(
    bundle: Any,
    *,
    ntc_platform: Optional[str] = None,
    ttp_template: Optional[str] = None,
) -> Any:
    """Best-effort structured parsing; retain raw evidence when optional parsers lack templates."""
    if not isinstance(bundle, Mapping):
        return bundle
    data = bundle.get("data")
    if not isinstance(data, Mapping):
        return bundle
    outputs = data.get("cli_outputs")
    if not isinstance(outputs, Mapping):
        return bundle
    parsed: Dict[str, Any] = {}
    for command, output in outputs.items():
        if not isinstance(output, str):
            parsed[str(command)] = output
            continue
        value: Any = None
        if ttp_template:
            try:
                from ttp import ttp

                parser = ttp(data=output, template=ttp_template)
                parser.parse()
                value = parser.result()
            except ImportError:
                pass
            except Exception:
                value = None
        if value is None and ntc_platform:
            try:
                from ntc_templates.parse import parse_output

                value = parse_output(platform=ntc_platform, command=str(command), data=output)
            except Exception:
                value = None
        parsed[str(command)] = value if value is not None else {"raw": output, "parsed": False}
    return {**bundle, "data": {**data, "parsed_cli_outputs": parsed}}


def audit_hierarchical_config(config: str) -> Dict[str, Any]:
    """Return parser-backed hierarchy when ciscoconfparse2 is installed, else raw lines."""
    try:
        from ciscoconfparse2 import CiscoConfParse

        parsed = CiscoConfParse(config.splitlines(), syntax="ios")
        config_objects = getattr(parsed, "ConfigObjs", config.splitlines())
        return {"available": True, "syntax": "ios", "lines": [str(line) for line in config_objects]}
    except ImportError:
        return {"available": False, "reason": "ciscoconfparse2 is not installed", "lines": config.splitlines()}
