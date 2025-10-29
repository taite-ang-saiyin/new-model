from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..models import (
    AgentActionRequest,
    AgentActionResult,
    AgentCustomization,
    AgentProfile,
    FateEventResult,
    MemoryCorrosionResult,
    Simulation,
)


class AIProvider(ABC):
    """Abstract interface encapsulating AI calls used by the backend."""

    @abstractmethod
    def generate_agents(
        self, scenario: str, customizations: List[AgentCustomization]
    ) -> List[AgentProfile]:
        """Generate five agent profiles that match the scenario."""

    @abstractmethod
    def generate_agent_action(self, request: AgentActionRequest) -> AgentActionResult:
        """Generate the next turn for a single agent."""

    @abstractmethod
    def corrode_memory(
        self, simulation: Simulation, agent: AgentProfile, action: AgentActionResult
    ) -> MemoryCorrosionResult:
        """Produce a biased/distorted memory of the recent action."""

    @abstractmethod
    def generate_fate_event(
        self, simulation: Simulation, user_prompt: str | None = None
    ) -> FateEventResult:
        """Create a dramatic twist that affects future turns."""
