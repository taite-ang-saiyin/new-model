from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

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

        profiles: List[AgentProfile]
        slot_index_map: Dict[int, int]
        if agent_profiles:
            if len(agent_profiles) > 5:
                raise ValueError("A maximum of five agents is supported.")
            profiles = agent_profiles[:5]
            slot_index_map = {idx: idx for idx in range(len(profiles))}
        elif customizations:
            profiles, slot_index_map = self._build_profiles_from_customizations(
                scenario, customizations
            )
        else:
            generated = self.provider.generate_agents(scenario, customizations)
            profiles = generated[:5]
            slot_index_map = {idx: idx for idx in range(len(profiles))}

        if not profiles:
            raise ValueError("At least one agent must be supplied to start a simulation.")

        simulation.agents = [AgentState(**_as_dict(profile)) for profile in profiles]
        simulation.relationships = self._map_relationships(
            simulation.agents, relationships, slot_index_map
        )
        simulation.status = SimulationStatus.RUNNING
        simulation.updated_at = datetime.utcnow()
        agent_count = len(simulation.agents)
        simulation.events.append(
            SimulationEvent(
                type=EventType.SYSTEM,
                turn=0,
                summary="Agents initialized",
                details=(
                    f"{agent_count} agent{'s' if agent_count != 1 else ''} entered the scenario with distinct agendas."
                ),
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
        thought_steps = action_result.thought_process or self._default_thought_process(
            request, action_result
        )
        agent.thought_process = thought_steps
        simulation.events.append(
            SimulationEvent(
                type=EventType.AGENT,
                actor_id=agent.id,
                turn=simulation.turn_index,
                summary="Internal reasoning",
                details="\n".join(f"- {step}" for step in thought_steps),
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

    def _build_profiles_from_customizations(
        self, scenario: str, customizations: List[AgentCustomization]
    ) -> Tuple[List[AgentProfile], Dict[int, int]]:
        if len(customizations) > 5:
            raise ValueError("A maximum of five custom agents is supported.")
        sorted_custom = sorted(customizations, key=lambda item: item.slot)
        slot_index_map: Dict[int, int] = {}
        profiles: List[AgentProfile] = []
        scenario_context = (scenario.strip() or "the mission").rstrip(".")
        for custom in sorted_custom:
            if custom.slot in slot_index_map:
                raise ValueError(
                    f"Duplicate custom agent slot provided: {custom.slot}."
                )
            display_index = len(profiles) + 1
            name = custom.name or f"Custom Agent {display_index}"
            role = custom.role or "Specialist"
            persona = custom.persona or (
                f"Operative attuned to the demands of {scenario_context}."
            )
            cognitive_bias = custom.cognitive_bias or "balanced perspective"
            emotional_state = custom.emotional_state or "neutral"
            secret_agenda = custom.motivation or (
                f"Advance personal leverage within {scenario_context}."
            )
            memory = f"Initial briefing on {scenario_context}."
            traits = {"origin": "user_provided", "slot": custom.slot}
            profile = AgentProfile(
                name=name,
                role=role,
                persona=persona,
                cognitive_bias=cognitive_bias,
                emotional_state=emotional_state,
                mbti=custom.mbti,
                skills=custom.skills or [],
                biography=custom.biography,
                constraints=custom.constraints or [],
                quirks=custom.quirks or [],
                motivation=custom.motivation,
                secret_agenda=secret_agenda,
                memory=memory,
                traits=traits,
            )
            profiles.append(profile)
            slot_index_map[custom.slot] = len(profiles) - 1
        return profiles, slot_index_map

    @staticmethod
    def _map_relationships(
        agents: List[AgentState],
        seeds: List[AgentRelationshipSeed],
        slot_index_map: Dict[int, int] | None = None,
    ) -> List[AgentRelationship]:
        if not agents or not seeds:
            return []
        resolved_map: Dict[int, int] = slot_index_map or {
            index: index for index in range(len(agents))
        }
        relationships: List[AgentRelationship] = []
        agent_count = len(agents)
        for seed in seeds:
            if seed.from_slot not in resolved_map or seed.to_slot not in resolved_map:
                raise ValueError(
                    f"Relationship references slot {seed.from_slot}->{seed.to_slot} "
                    f"but available slots are {sorted(resolved_map.keys())}."
                )
            from_index = resolved_map[seed.from_slot]
            to_index = resolved_map[seed.to_slot]
            if from_index == to_index:
                continue
            if from_index >= agent_count or to_index >= agent_count:
                raise ValueError(
                    f"Relationship resolved to invalid agent index {from_index}->{to_index}."
                )
            source_id = agents[from_index].id
            target_id = agents[to_index].id
            relationships.append(
                AgentRelationship(
                    source_agent_id=source_id,
                    target_agent_id=target_id,
                    strength=seed.strength,
                )
            )
        return relationships

    @staticmethod
    def _default_thought_process(
        request: AgentActionRequest, action: AgentActionResult
    ) -> List[str]:
        steps = ["Thinks about the mission plan."]
        if request.recent_events:
            steps.append(
                f"Remembers the last change: {request.recent_events[-1].summary}."
            )
        action_text = action.action_summary.rstrip(".")
        if action_text:
            trimmed = action_text[:100]
            steps.append(f"Chooses to follow through: {trimmed}.")
        else:
            steps.append("Chooses the next move.")
        return steps
