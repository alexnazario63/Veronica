"""Async ReAct loop with deterministic tool-tier enforcement.

The model is only asked to propose a tool call. The registry and tier gates
decide what may run; the model never applies a network change itself.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import uuid
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence

from core.contracts import (
    ArtifactDelta, ArtifactOpen, MessageDelta, NetCanvasArtifact,
    SafetyGateInterrupt, ThoughtDelta, ToolEnd, ToolSafetyTier, ToolStart,
)
from knowledge.ingestion.sanitizer import sanitize_device_output, sanitize_untrusted_text

ToolHandler = Callable[[Mapping[str, Any]], Any | Awaitable[Any]]
ModelStep = Callable[[str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class AgentTool:
    name: str
    tier: ToolSafetyTier
    handler: ToolHandler
    description: str = ""


class ReActOrchestrator:
    def __init__(self, tools: Iterable[AgentTool] = (), *, model_step: ModelStep | None = None,
                 change_control: Any = None, max_steps: int = 8) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.model_step = model_step
        self.change_control = change_control
        self.max_steps = max(1, max_steps)
        self.sessions: dict[str, list[dict[str, Any]]] = {}

    async def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        value = fn(*args)
        return await value if hasattr(value, "__await__") else value

    async def _execute(self, tool: AgentTool, args: Mapping[str, Any]) -> Any:
        result = await self._call(tool.handler, args)
        if isinstance(result, str):
            result = sanitize_device_output(result, source=tool.name).text
        elif isinstance(result, Mapping):
            # Treat tool output as untrusted recursively for common text fields.
            result = dict(result)
            for key, value in tuple(result.items()):
                if isinstance(value, str) and key.lower() in {
                    "output", "content", "banner", "description", "neighbor_name", "raw", "text"
                }:
                    result[key] = sanitize_device_output(value, source=f"{tool.name}.{key}").text
        return result

    async def run(self, user_message: str, *, session_id: str | None = None,
                  context: Mapping[str, Any] | None = None):
        """Yield immutable AgentStreamEvent values; tool proposals use JSON objects."""
        sid = session_id or uuid.uuid4().hex
        history = self.sessions.setdefault(sid, [])
        safe_user = sanitize_untrusted_text(user_message, source="user_message").text
        history.append({"role": "user", "content": safe_user})
        evidence: list[dict[str, Any]] = []
        if context:
            model_profile = context.get("model_profile")
            response_style = context.get("response_style")
            if model_profile in {"vrn-fast", "vrn-pro", "vrn-ultra"} or response_style in {"technical", "natural", "mixed", "engineer"}:
                evidence.append({"response_preferences": {
                    "model_profile": model_profile if model_profile in {"vrn-fast", "vrn-pro", "vrn-ultra"} else "vrn-pro",
                    "response_style": response_style if response_style in {"technical", "natural", "mixed", "engineer"} else "technical",
                }})
        if self.model_step is None:
            yield MessageDelta(delta="No model step is configured; tools remain available through the orchestrator API.")
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
                    data.setdefault("sha256_hash", hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest())
                    artifact = NetCanvasArtifact(**data)
                    yield ArtifactOpen(artifact=artifact)
                return
            tier0 = [c for c in calls if self._tool(c).tier == ToolSafetyTier.TIER_0_READ_ONLY]
            for call in calls:
                tool = self._tool(call)
                if tool.tier == ToolSafetyTier.TIER_2_WRITE_LOCKED:
                    artifact_id = str(call.get("artifact_id", "pending"))
                    yield SafetyGateInterrupt(artifact_id=artifact_id, change_stage="6_HUMAN_APPROVAL",
                                              reason="Write operation requires deterministic change control and signed human approval.")
                    return
                if tool.tier == ToolSafetyTier.TIER_1_SIMULATE_PLAN:
                    if self.change_control is None:
                        yield ToolEnd(tool_call_id=str(call.get("id", "")), status="blocked_no_change_control")
                        continue
                    args = call.get("arguments", {})
                    try:
                        plan = await asyncio.to_thread(self.change_control.create_change_plan_stages_1_to_5, **dict(args))
                    except Exception as exc:
                        yield ToolEnd(tool_call_id=str(call.get("id", "")), status=f"blocked:{type(exc).__name__}")
                        continue
                    stage = plan.get("current_stage", 5)
                    payload = json.dumps(plan, sort_keys=True, default=str)
                    artifact = NetCanvasArtifact(id=str(plan["change_id"]), version=1,
                        type="change_plan_10_steps", baseline_content=json.dumps(plan["stages"].get("stage_2_baseline"), default=str),
                        proposed_content=json.dumps(plan["stages"].get("stage_5_simulation"), default=str),
                        blast_radius_graph=plan["stages"].get("stage_3_impact_analysis"), change_stage=str(stage),
                        sha256_hash=hashlib.sha256(payload.encode()).hexdigest())
                    yield ArtifactOpen(artifact=artifact)
                    yield ArtifactDelta(artifact_id=artifact.id, version=1, delta=payload)
            if tier0:
                async def run_one(call: Mapping[str, Any]):
                    tool = self._tool(call)
                    return call, tool, await self._execute(tool, call.get("arguments", {}))
                tasks = [asyncio.create_task(run_one(c)) for c in tier0]
                for call in tier0:
                    tool = self._tool(call)
                    yield ToolStart(tool_call_id=str(call.get("id", "")), tool_name=tool.name, tier=tool.tier)
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for item in results:
                    if isinstance(item, BaseException):
                        evidence.append({"error": f"{type(item).__name__}: tool failed"})
                        continue
                    call, tool, result = item
                    evidence.append({"tool": tool.name, "result": result})
                    yield ToolEnd(tool_call_id=str(call.get("id", "")), status="ok")
        yield MessageDelta(delta="Stopped after the configured ReAct step limit.")

    def _tool(self, call: Mapping[str, Any]) -> AgentTool:
        name = str(call.get("name", ""))
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        return self.tools[name]
