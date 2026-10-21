"""Shared immutable contracts for the Veronica chat, agent, and workspace."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class EpistemicTag(StrEnum):
    VERIFIED_FACT = "VERIFIED_FACT"
    TECHNICAL_INFERENCE = "TECHNICAL_INFERENCE"
    HYPOTHESIS = "HYPOTHESIS"


class ToolSafetyTier(StrEnum):
    TIER_0_READ_ONLY = "TIER_0_READ_ONLY"
    TIER_1_SIMULATE_PLAN = "TIER_1_SIMULATE_PLAN"
    TIER_2_WRITE_LOCKED = "TIER_2_WRITE_LOCKED"


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ThoughtDelta(ImmutableModel):
    type: Literal["thought_delta"] = "thought_delta"
    delta: str


class ToolStart(ImmutableModel):
    type: Literal["tool_start"] = "tool_start"
    tool_call_id: str
    tool_name: str
    tier: ToolSafetyTier


class ToolEnd(ImmutableModel):
    type: Literal["tool_end"] = "tool_end"
    tool_call_id: str
    status: str
    duration_ms: float | None = None


class ArtifactOpen(ImmutableModel):
    type: Literal["artifact_open"] = "artifact_open"
    artifact: "NetCanvasArtifact"


class ArtifactDelta(ImmutableModel):
    type: Literal["artifact_delta"] = "artifact_delta"
    artifact_id: str
    version: int
    delta: str


class MessageDelta(ImmutableModel):
    type: Literal["message_delta"] = "message_delta"
    delta: str


class SafetyGateInterrupt(ImmutableModel):
    type: Literal["safety_gate_interrupt"] = "safety_gate_interrupt"
    artifact_id: str
    change_stage: str
    reason: str


AgentStreamEvent: TypeAlias = Annotated[
    ThoughtDelta | ToolStart | ToolEnd | ArtifactOpen | ArtifactDelta | MessageDelta | SafetyGateInterrupt,
    Field(discriminator="type"),
]


class NetCanvasArtifact(ImmutableModel):
    id: str
    version: int = Field(ge=1)
    type: Literal[
        "config_diff", "blast_radius", "rca_report", "diagnostic_11_steps", "change_plan_10_steps"
    ]
    baseline_content: str | None = None
    proposed_content: str | None = None
    blast_radius_graph: dict[str, object] | None = None
    change_stage: str | None = None
    sha256_hash: str


ArtifactOpen.model_rebuild()
