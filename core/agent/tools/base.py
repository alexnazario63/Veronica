"""Base contracts for agent tools."""
from __future__ import annotations

from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from core.contracts import ToolSafetyTier

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
