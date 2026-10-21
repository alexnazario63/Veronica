"""Strict lifecycle state machine for a ten stage change."""

from __future__ import annotations

try:
    from transitions import Machine
except ImportError:  # Keep offline/minimal installs fail-closed and usable.
    Machine = None  # type: ignore[assignment,misc]


STAGES = (
    "1_DISCOVERY",
    "2_BASELINE",
    "3_IMPACT_ANALYSIS",
    "4_PLANNING",
    "5_SIMULATION",
    "6_HUMAN_APPROVAL",
    "7_CONTROLLED_EXECUTION",
    "8_POST_VALIDATION",
    "9_COMMIT_OR_ROLLBACK",
    "10_IMMUTABLE_AUDIT",
)


class ChangeLifecycle:
    """An explicitly triggered FSM; callers cannot skip or reverse a stage."""

    def __init__(self) -> None:
        self.state = STAGES[0]
        self._index = 0
        self.machine = None
        if Machine is not None:
            self.machine = Machine(
                model=self,
                states=list(STAGES),
                initial=STAGES[0],
                auto_transitions=False,
                ignore_invalid_triggers=False,
            )
            for index, state in enumerate(STAGES[:-1]):
                # Use a trigger name distinct from this class's ``advance`` method;
                # transitions otherwise refuses to bind the event to the model.
                self.machine.add_transition("advance_stage", state, STAGES[index + 1])

    def advance(self) -> None:
        if self.machine is not None:
            getattr(self, "advance_stage")()
            return
        if self._index >= len(STAGES) - 1:
            raise RuntimeError("Change lifecycle is already complete")
        self._index += 1
        self.state = STAGES[self._index]

    def advance_to(self, stage_number: int) -> None:
        if not 1 <= stage_number <= len(STAGES):
            raise ValueError(f"Invalid change stage: {stage_number}")
        target = STAGES[stage_number - 1]
        while self.state != target:
            if self.state == STAGES[-1]:
                raise RuntimeError("Change lifecycle is already complete")
            self.advance()

