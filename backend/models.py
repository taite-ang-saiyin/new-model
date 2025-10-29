from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def _uuid() -> str:
    return str(uuid.uuid4())


class SimulationStatus(str, Enum):
    """Lifecycle states for a simulation session."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class EventType(str, Enum):
    """Typed events matching the PDF terminology."""

    SYSTEM = "system"
    AGENT = "agent"
    FATE_WEAVER = "fate_weaver"
    ERROR = "error"


class Position(BaseModel):
    """Simple 2D coordinate used by the collaboration canvas."""

    x: float = Field(0.0, ge=-100.0, le=100.0)
    y: float = Field(0.0, ge=-100.0, le=100.0)


class AgentCustomization(BaseModel):
    """Optional seeds supplied by the user before simulation start."""

    slot: int = Field(..., ge=0, le=4)
    name: Optional[str] = None
    role: Optional[str] = None
    persona: Optional[str] = None
    cognitive_bias: Optional[str] = None
    emotional_state: Optional[str] = None
    mbti: Optional[str] = None
    skills: Optional[List[str]] = None
    biography: Optional[str] = None
    constraints: Optional[List[str]] = None
    quirks: Optional[List[str]] = None
    motivation: Optional[str] = None


class AgentProfile(BaseModel):
    """Full agent profile returned by the AI provider."""

    id: str = Field(default_factory=_uuid)
    name: str
    role: str
    persona: str
    cognitive_bias: str
    emotional_state: str
    mbti: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    biography: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)
    quirks: List[str] = Field(default_factory=list)
    motivation: Optional[str] = None
    secret_agenda: str
    agenda_progress: float = Field(0.0, ge=0.0, le=1.0)
    memory: str
    position: Position = Field(default_factory=Position)
    corroded_memory: Optional[str] = None
    last_action: Optional[str] = None
    traits: Dict[str, str | List[str]] = Field(default_factory=dict)


class AgentActionRequest(BaseModel):
    """Structured payload sent to the AI provider to request an action."""

    agent: AgentProfile
    recent_events: List["SimulationEvent"]
    scenario: str
    turn_index: int


class AgentActionResult(BaseModel):
    """Structured response from the AI provider for a single turn."""

    action_summary: str
    detailed_log: str
    emotional_state: str
    agenda_progress_delta: float
    new_position: Position


class MemoryCorrosionResult(BaseModel):
    """AI-generated distortion of the agent's memory."""

    distorted_memory: str
    reliability_delta: float = Field(0.0, ge=-1.0, le=1.0)


class FateEventResult(BaseModel):
    """AI-generated dramatic twist."""

    title: str
    description: str
    impact: Optional[str] = None


class SimulationEvent(BaseModel):
    """Chronological event log entry."""

    id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    type: EventType
    turn: int
    summary: str
    details: str
    actor_id: Optional[str] = None


class AgentState(AgentProfile):
    """Runtime agent state derived from the profile."""

    turn_count: int = 0


class AgentRelationshipSeed(BaseModel):
    """Relationship definition using agent slots before IDs exist."""

    from_slot: int = Field(..., ge=0, le=4)
    to_slot: int = Field(..., ge=0, le=4)
    strength: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Relationship weight (-1 hostile ... 1 allied).",
    )


class AgentRelationship(BaseModel):
    """Directional relationship strength between two agents."""

    source_agent_id: str = Field(..., description="ID of the source agent.")
    target_agent_id: str = Field(..., description="ID of the target agent.")
    strength: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Relationship weight (-1 hostile ... 1 allied).",
    )


class Simulation(BaseModel):
    """Complete simulation state stored in the repository."""

    id: str = Field(default_factory=_uuid)
    scenario: str
    status: SimulationStatus = SimulationStatus.INITIALIZING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    agents: List[AgentState] = Field(default_factory=list)
    events: List[SimulationEvent] = Field(default_factory=list)
    relationships: List[AgentRelationship] = Field(default_factory=list)
    turn_index: int = 0
    active_agent_index: int = 0
    memory_reliability: float = Field(1.0, ge=0.0, le=1.0)

    def next_agent(self) -> AgentState:
        if not self.agents:
            raise ValueError("Simulation has no agents.")
        return self.agents[self.active_agent_index]

    def advance_agent_pointer(self):
        if not self.agents:
            return
        self.active_agent_index = (self.active_agent_index + 1) % len(self.agents)
        self.turn_index += 1
        self.updated_at = datetime.utcnow()


class SimulationSummary(BaseModel):
    """Serialized response returned to API consumers."""

    simulation: Simulation


try:  # Pydantic v2
    AgentActionRequest.model_rebuild()
except AttributeError:  # Pydantic v1 fallback
    AgentActionRequest.update_forward_refs()
