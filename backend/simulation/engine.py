from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from ..ai import AIProvider

# Compatibility helper for Pydantic v1/v2
def _as_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

from ..models import (
    AgentActionRequest,
    AgentCustomization,
    AgentProfile,
    AgentRelationship,
    AgentRelationshipSeed,
    AgentState,
    EventType,
    Simulation,
    SimulationEvent,
    SimulationStatus,
)
from ..repository import SimulationRepository


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


class SimulationEngine:
    """Coordinates scenarios, agent turns, and logging."""

    def __init__(
        self, provider: AIProvider, repository: Optional[SimulationRepository] = None
    ):
        self.provider = provider
        self.repository = repository or SimulationRepository()

    def create_simulation(
        self,
        scenario: str,
        customizations: List[AgentCustomization] | None = None,
        agent_profiles: List[AgentProfile] | None = None,
        relationships: List[AgentRelationshipSeed] | None = None,
    ) -> Simulation:
        customizations = customizations or []
        agent_profiles = agent_profiles or []
        relationships = relationships or []
        simulation = Simulation(scenario=scenario)
        simulation.events.append(
            SimulationEvent(
                type=EventType.SYSTEM,
                turn=0,
                summary="Simulation created",
                details="Scenario registered. Initializing agents.",
            )
        )

        profiles: List[AgentProfile] = []
        if agent_profiles:
            profiles.extend(agent_profiles[:5])
        if len(profiles) < 5:
            generated = self.provider.generate_agents(scenario, customizations)
            needed = 5 - len(profiles)
            profiles.extend(generated[:needed])

        simulation.agents = [AgentState(**_as_dict(profile)) for profile in profiles[:5]]
        simulation.relationships = self._map_relationships(
            simulation.agents, relationships
        )
        simulation.status = SimulationStatus.RUNNING
        simulation.updated_at = datetime.utcnow()
        simulation.events.append(
            SimulationEvent(
                type=EventType.SYSTEM,
                turn=0,
                summary="Agents initialized",
                details="Five agents entered the scenario with distinct agendas.",
            )
        )
        if simulation.relationships:
            simulation.events.append(
                SimulationEvent(
                    type=EventType.SYSTEM,
                    turn=0,
                    summary="Relationships initialized",
                    details=f"{len(simulation.relationships)} relationship links configured.",
                )
            )
        self.repository.add(simulation)
        return simulation

    def get_simulation(self, simulation_id: str) -> Simulation:
        simulation = self.repository.get(simulation_id)
        if not simulation:
            raise KeyError(f"Simulation {simulation_id} not found.")
        return simulation

    def advance_turn(self, simulation_id: str) -> Simulation:
        simulation = self.get_simulation(simulation_id)
        if simulation.status != SimulationStatus.RUNNING:
            raise RuntimeError(f"Simulation {simulation_id} is not running.")

        agent = simulation.next_agent()
        request = AgentActionRequest(
            agent=agent,
            recent_events=simulation.events[-10:],
            scenario=simulation.scenario,
            turn_index=simulation.turn_index,
        )
        action_result = self.provider.generate_agent_action(request)

        agent.last_action = action_result.action_summary
        agent.emotional_state = action_result.emotional_state
        agent.agenda_progress = _clamp(
            agent.agenda_progress + action_result.agenda_progress_delta
        )
        agent.position = action_result.new_position
        agent.turn_count += 1

        simulation.events.append(
            SimulationEvent(
                type=EventType.AGENT,
                actor_id=agent.id,
                turn=simulation.turn_index,
                summary=action_result.action_summary,
                details=action_result.detailed_log,
            )
        )

        corrosion = self.provider.corrode_memory(simulation, agent, action_result)
        agent.corroded_memory = corrosion.distorted_memory
        simulation.memory_reliability = _clamp(
            simulation.memory_reliability + corrosion.reliability_delta
        )

        simulation.events.append(
            SimulationEvent(
                type=EventType.AGENT,
                actor_id=agent.id,
                turn=simulation.turn_index,
                summary="Memory corrosion applied",
                details=corrosion.distorted_memory,
            )
        )

        simulation.advance_agent_pointer()
        self.repository.update(simulation)
        return simulation

    def advance_turns(self, simulation_id: str, steps: int) -> Simulation:
        if steps < 1:
            raise ValueError("Steps must be at least 1.")
        simulation: Optional[Simulation] = None
        for _ in range(steps):
            simulation = self.advance_turn(simulation_id)
        return simulation  # type: ignore[return-value]

    def trigger_fate_event(
        self, simulation_id: str, user_prompt: str | None = None
    ) -> Simulation:
        simulation = self.get_simulation(simulation_id)
        fate_event = self.provider.generate_fate_event(
            simulation, user_prompt=user_prompt
        )
        details = fate_event.description
        if fate_event.impact:
            details = f"{details} Impact: {fate_event.impact}"
        if user_prompt:
            details = f"{details} Fate Weaver input: {user_prompt.strip()}."
        simulation.events.append(
            SimulationEvent(
                type=EventType.FATE_WEAVER,
                turn=simulation.turn_index,
                summary=fate_event.title,
                details=details,
            )
        )
        simulation.updated_at = datetime.utcnow()
        self.repository.update(simulation)
        return simulation

    @staticmethod
    def _map_relationships(
        agents: List[AgentState], seeds: List[AgentRelationshipSeed]
    ) -> List[AgentRelationship]:
        if not agents or not seeds:
            return []
        relationships: List[AgentRelationship] = []
        agent_count = len(agents)
        for seed in seeds:
            if seed.from_slot >= agent_count or seed.to_slot >= agent_count:
                raise ValueError(
                    f"Relationship references slot {seed.from_slot}->{seed.to_slot} "
                    f"but only {agent_count} agents are available."
                )
            if seed.from_slot == seed.to_slot:
                continue
            source_id = agents[seed.from_slot].id
            target_id = agents[seed.to_slot].id
            relationships.append(
                AgentRelationship(
                    source_agent_id=source_id,
                    target_agent_id=target_id,
                    strength=seed.strength,
                )
            )
        return relationships
