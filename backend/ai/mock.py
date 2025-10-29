from __future__ import annotations

import random
from typing import Dict, List

from ..models import (
    AgentActionRequest,
    AgentActionResult,
    AgentCustomization,
    AgentProfile,
    FateEventResult,
    MemoryCorrosionResult,
    Position,
    Simulation,
    SimulationEvent,
)
from .base import AIProvider


class MockAIProvider(AIProvider):
    """
    Offline-friendly AI provider used for local development and testing.

    The behaviour is intentionally lightweight yet mirrors the structured
    responses expected from the external-provider implementation.
    """

    def __init__(self, seed: int | None = None):
        self._rand = random.Random(seed)

    def _apply_customization(
        self,
        index: int,
        base_profile: Dict[str, object],
        custom: Dict[int, AgentCustomization],
    ):
        if index not in custom:
            return base_profile
        override = custom[index]
        scalar_keys = (
            "name",
            "role",
            "persona",
            "cognitive_bias",
            "emotional_state",
            "mbti",
            "biography",
            "motivation",
        )
        list_keys = ("skills", "constraints", "quirks")
        for key in scalar_keys:
            value = getattr(override, key)
            if value:
                base_profile[key] = value
        for key in list_keys:
            value = getattr(override, key)
            if value is not None:
                base_profile[key] = value
        return base_profile

    def generate_agents(
        self, scenario: str, customizations: List[AgentCustomization]
    ) -> List[AgentProfile]:
        themes = [
            "rebellious",
            "methodical",
            "empathetic",
            "opportunistic",
            "skeptical",
        ]
        roles = ["strategist", "engineer", "diplomat", "analyst", "scout"]
        biases = [
            "confirmation bias",
            "loss aversion",
            "anchoring bias",
            "status quo bias",
            "overconfidence effect",
        ]
        emotions = ["focused", "anxious", "determined", "curious", "guarded"]
        mbti_types = ["INTJ", "ENFP", "ISTP", "ENTJ", "INFJ"]
        skill_sets = [
            ["systems thinking", "signal analysis"],
            ["improvised engineering", "resource triage"],
            ["diplomacy", "narrative framing"],
            ["pattern recognition", "threat modeling"],
            ["mobility scouting", "stealth recon"],
        ]
        biographies = [
            "Ex-intelligence operative who values precision above all.",
            "Field tinkerer that never met a broken device they could not fix.",
            "Cultural attaché who reads between the lines of every treaty.",
            "Data ethicist who questions every conclusion twice.",
            "Pathfinder driven to map the unknown before dawn.",
        ]
        constraint_sets = [
            ["Must protect civilian assets", "Limited fuel reserves"],
            ["Cannot abandon toolkit cache", "Avoids direct combat"],
            ["Bound by envoy charter", "Must keep factions calm"],
            ["Needs verifiable proof", "Suffers information overload"],
            ["Navigation beacons unstable", "Radio silence for stealth"],
        ]
        quirk_sets = [
            ["Quotes obscure manuals", "Rearranges mission pins"],
            ["Names every gadget", "Sketches blueprints mid-briefing"],
            ["Collects proverbs", "Hums when negotiating"],
            ["Taps rhythms when thinking", "Logs anomalies obsessively"],
            ["Marks paths with origami", "Keeps meteor dust samples"],
        ]
        motivations = [
            "Prove that strategy beats brute force.",
            "Redeem a failed prototype from years ago.",
            "Prevent factions from sliding into open war.",
            "Expose the truth no matter the cost.",
            "Discover what really anchors the anomaly.",
        ]

        custom_map = {item.slot: item for item in customizations}
        profiles: List[AgentProfile] = []

        for idx in range(5):
            profile_data = {
                "name": f"Agent {idx + 1}",
                "role": roles[idx % len(roles)],
                "persona": f"{themes[idx % len(themes)].title()} visionary shaped by {scenario[:40] or 'unknown origins'}.",
                "cognitive_bias": biases[idx % len(biases)],
                "emotional_state": emotions[idx % len(emotions)],
                "mbti": mbti_types[idx % len(mbti_types)],
                "skills": skill_sets[idx % len(skill_sets)],
                "biography": biographies[idx % len(biographies)],
                "constraints": constraint_sets[idx % len(constraint_sets)],
                "quirks": quirk_sets[idx % len(quirk_sets)],
                "motivation": motivations[idx % len(motivations)],
                "secret_agenda": f"Pursue advantage linked to {scenario.split()[0] if scenario else 'mystery'}.",
                "memory": "Remembers briefing with clarity.",
                "traits": {"motivation": themes[idx % len(themes)]},
            }
            profile_data = self._apply_customization(idx, profile_data, custom_map)
            base_position = Position(
                x=self._rand.uniform(-25.0, 25.0),
                y=self._rand.uniform(-25.0, 25.0),
            )
            profiles.append(AgentProfile(position=base_position, **profile_data))
        return profiles

    def generate_agent_action(self, request: AgentActionRequest) -> AgentActionResult:
        agent = request.agent
        verbs = [
            "investigates",
            "confronts",
            "resembles",
            "hacks",
            "reframes",
            "stabilizes",
        ]
        contexts = [
            "a critical subsystem",
            "a rival's intentions",
            "the mysterious artifact",
            "a hidden comms channel",
            "their own doubts",
            "the faction alliance",
        ]
        verb = self._rand.choice(verbs)
        context = self._rand.choice(contexts)
        action_summary = f"{agent.name} {verb} {context}."

        emotional_shift = self._rand.choice(
            [
                "heightened resolve",
                "cautious optimism",
                "latent suspicion",
                "renewed curiosity",
                "strategic doubt",
            ]
        )
        agenda_delta = round(self._rand.uniform(-0.08, 0.15), 3)
        new_position = Position(
            x=max(-100.0, min(100.0, agent.position.x + self._rand.uniform(-8.0, 8.0))),
            y=max(-100.0, min(100.0, agent.position.y + self._rand.uniform(-8.0, 8.0))),
        )
        detailed_log = (
            f"{agent.name} reflects on earlier events and {verb} {context}, "
            f"leading to {emotional_shift}."
        )
        return AgentActionResult(
            action_summary=action_summary,
            detailed_log=detailed_log,
            emotional_state=emotional_shift,
            agenda_progress_delta=agenda_delta,
            new_position=new_position,
        )

    def corrode_memory(
        self, simulation: Simulation, agent: AgentProfile, action: AgentActionResult
    ) -> MemoryCorrosionResult:
        distortion_templates = [
            "Insists the action succeeded flawlessly, dismissing any setbacks.",
            "Believes another agent forced the move, downplaying personal responsibility.",
            "Remembers taking a far bolder stance than reality shows.",
            "Thinks the encounter was a trap, even without evidence.",
            "Claims the event was inconsequential and barely recalls doing it.",
        ]
        distorted_memory = self._rand.choice(distortion_templates)
        reliability_delta = round(self._rand.uniform(-0.1, -0.02), 3)
        return MemoryCorrosionResult(
            distorted_memory=f"{agent.name} now recalls: {distorted_memory}",
            reliability_delta=reliability_delta,
        )

    def generate_fate_event(
        self, simulation: Simulation, user_prompt: str | None = None
    ) -> FateEventResult:
        twists = [
            (
                "Signal Cascade",
                "An ancient signal reactivates, luring agents toward conflicting coordinates.",
            ),
            (
                "Silent Saboteur",
                "Critical support systems fail simultaneously, hinting at a traitor within.",
            ),
            (
                "Echo of the Artifact",
                "A psychic projection forces each agent to relive their worst fear.",
            ),
            (
                "Alliance Ultimatum",
                "External forces demand an immediate decision that splits the team.",
            ),
            (
                "Mirage Corridor",
                "The environment warps, making navigation unreliable for the next turns.",
            ),
        ]
        title, description = self._rand.choice(twists)
        if user_prompt:
            description = f"{description} Fate Weaver adds: {user_prompt.strip()}."
        impact = "Future agent actions should reference this twist."
        return FateEventResult(title=title, description=description, impact=impact)


def summarize_recent_events(events: List[SimulationEvent], limit: int = 5) -> str:
    """Helper to build a compact summary for prompts where needed."""
    return "\n".join(event.summary for event in events[-limit:])
