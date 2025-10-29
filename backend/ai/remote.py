from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, List, Optional

import requests
from pydantic import ValidationError

from ..models import (
    AgentActionRequest,
    AgentActionResult,
    AgentCustomization,
    AgentProfile,
    FateEventResult,
    MemoryCorrosionResult,
    Position,
    Simulation,
)
from .base import AIProvider

logger = logging.getLogger(__name__)


class HuggingFaceSpaceProvider(AIProvider):
    """
    AIProvider that proxies Cogniverse prompts to a Hugging Face Space
    wrapping an instruction-tuned large language model.

    The environment must expose:
        COGNIVERSE_HF_SPACE_URL   -> base URL to the Space
        COGNIVERSE_HF_SPACE_TOKEN -> optional Bearer token (private Spaces)
    """

    def __init__(
        self,
        space_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        timeout: float = 45.0,
    ):
        base_url = space_url or os.getenv("COGNIVERSE_HF_SPACE_URL")
        if not base_url:
            raise RuntimeError(
                "COGNIVERSE_HF_SPACE_URL is not configured for HuggingFaceSpaceProvider."
            )
        self.space_url = base_url.rstrip("/")
        if not self.space_url.endswith("/run/predict"):
            self.space_url = f"{self.space_url}/run/predict"
        self.timeout = timeout
        self.session = requests.Session()
        token = auth_token or os.getenv("COGNIVERSE_HF_SPACE_TOKEN")
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    def _invoke(self, prompt: str) -> str:
        payload = {"data": [prompt]}
        response = self.session.post(
            self.space_url, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "data" in data and data["data"]:
            text = data["data"][0]
            if isinstance(text, str):
                return text.strip()
        raise ValueError(f"Unexpected response payload: {data!r}")

    @staticmethod
    def _strip_json(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            parts = cleaned.split("\n", 1)
            cleaned = parts[1] if len(parts) > 1 else parts[0]
        # Remove possible trailing ```
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        return cleaned.strip()

    @classmethod
    def _parse_json(cls, text: str) -> Dict:
        cleaned = cls._strip_json(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON: %s", cleaned)
            raise ValueError("Remote model returned invalid JSON") from exc

    @staticmethod
    def _apply_customizations(
        base_profiles: List[AgentProfile], customizations: List[AgentCustomization]
    ) -> List[AgentProfile]:
        custom_map = {item.slot: item for item in customizations}
        for idx, profile in enumerate(base_profiles):
            override = custom_map.get(idx)
            if not override:
                continue
            scalar_fields = (
                "name",
                "role",
                "persona",
                "cognitive_bias",
                "emotional_state",
                "mbti",
                "biography",
                "motivation",
            )
            list_fields = ("skills", "constraints", "quirks")
            for field in scalar_fields:
                value = getattr(override, field, None)
                if value:
                    setattr(profile, field, value)
            for field in list_fields:
                value = getattr(override, field, None)
                if value is not None:
                    setattr(profile, field, value)
        return base_profiles

    # ------------------------------------------------------------------ #
    #  AIProvider interface                                              #
    # ------------------------------------------------------------------ #
    def generate_agents(
        self, scenario: str, customizations: List[AgentCustomization]
    ) -> List[AgentProfile]:
        prompt = self._agent_prompt(scenario, customizations)
        raw = self._invoke(prompt)
        payload = self._parse_json(raw)
        agents = payload.get("agents")
        if not isinstance(agents, list):
            raise ValueError("Expected 'agents' list in response.")

        profiles: List[AgentProfile] = []
        for idx, agent_data in enumerate(agents[:5]):
            try:
                profile = self._build_agent_profile(agent_data, idx)
                profiles.append(profile)
            except ValidationError as exc:
                logger.warning("Agent profile validation failed: %s", exc)
        if len(profiles) < 5:
            logger.warning(
                "Remote provider returned %s agents; padding with defaults.",
                len(profiles),
            )
            profiles.extend(self._default_agents(5 - len(profiles), scenario))
        profiles = profiles[:5]
        return self._apply_customizations(profiles, customizations)

    def generate_agent_action(self, request: AgentActionRequest) -> AgentActionResult:
        prompt = self._action_prompt(request)
        raw = self._invoke(prompt)
        payload = self._parse_json(raw)
        result = payload.get("action")
        if not isinstance(result, dict):
            raise ValueError("Expected 'action' object in response.")

        new_position = result.get("new_position", {})
        position = Position(
            x=float(new_position.get("x", request.agent.position.x)),
            y=float(new_position.get("y", request.agent.position.y)),
        )
        agenda_delta = float(result.get("agenda_progress_delta", 0.0))

        return AgentActionResult(
            action_summary=result.get(
                "action_summary", f"{request.agent.name} acts cautiously."
            ),
            detailed_log=result.get(
                "detailed_log",
                "The agent reflects on the scenario and adjusts their approach.",
            ),
            emotional_state=result.get(
                "emotional_state", request.agent.emotional_state
            ),
            agenda_progress_delta=agenda_delta,
            new_position=position,
        )

    def corrode_memory(
        self, simulation: Simulation, agent: AgentProfile, action: AgentActionResult
    ) -> MemoryCorrosionResult:
        prompt = self._memory_prompt(simulation, agent, action)
        raw = self._invoke(prompt)
        payload = self._parse_json(raw)
        memory = payload.get("memory") or {}
        distorted = memory.get(
            "distorted_memory",
            f"{agent.name} vaguely recalls the event but the details feel unreliable.",
        )
        reliability = float(memory.get("reliability_delta", -0.05))
        return MemoryCorrosionResult(
            distorted_memory=distorted,
            reliability_delta=reliability,
        )

    def generate_fate_event(
        self, simulation: Simulation, user_prompt: str | None = None
    ) -> FateEventResult:
        prompt = self._fate_prompt(simulation, user_prompt)
        raw = self._invoke(prompt)
        payload = self._parse_json(raw)
        event = payload.get("fate") or {}
        return FateEventResult(
            title=event.get("title", "Unexpected Disturbance"),
            description=event.get(
                "description",
                "An unforeseen ripple shifts alliances and unsettles the agents.",
            ),
            impact=event.get(
                "impact", "Subsequent turns should reference the destabilising event."
            ),
        )

    # ------------------------------------------------------------------ #
    #  Prompt builders                                                   #
    # ------------------------------------------------------------------ #
    def _agent_prompt(
        self, scenario: str, customizations: List[AgentCustomization]
    ) -> str:
        custom = [
            {
                "slot": item.slot,
                "name": item.name,
                "role": item.role,
                "persona": item.persona,
                "cognitive_bias": item.cognitive_bias,
                "emotional_state": item.emotional_state,
                "mbti": item.mbti,
                "skills": item.skills,
                "biography": item.biography,
                "constraints": item.constraints,
                "quirks": item.quirks,
                "motivation": item.motivation,
            }
            for item in customizations
        ]
        return (
            "You are the agent architect for the Cogniverse simulation. "
            "Given the scenario and optional user customisations, produce exactly FIVE agents. "
            "Each agent must be a JSON object with the keys: "
            "name, role, persona, cognitive_bias, emotional_state, mbti, skills (array), "
            "biography, constraints (array), quirks (array), motivation, secret_agenda, "
            "memory (short first-person recollection), and traits (object with at least one entry). "
            "Respond with JSON only, in the format {\"agents\": [ ...five agents... ]}. "
            f"\nScenario: {scenario}\nCustomisations: {json.dumps(custom)}"
        )

    def _action_prompt(self, request: AgentActionRequest) -> str:
        events = [
            {
                "turn": event.turn,
                "type": event.type.value,
                "summary": event.summary,
            }
            for event in request.recent_events[-5:]
        ]
        return (
            "You simulate one turn for the active agent in the Cogniverse narrative engine. "
            "Return a JSON object {\"action\": {...}} with keys: "
            "action_summary (concise sentence), detailed_log (2-3 sentences), "
            "emotional_state (single descriptor), agenda_progress_delta (float between -0.2 and 0.2), "
            "new_position (object with numeric x and y between -100 and 100). "
            "Stay consistent with the agent persona and recent events.\n"
            f"Scenario: {request.scenario}\n"
            f"Turn index: {request.turn_index}\n"
            f"Agent: {request.agent.model_dump()}\n"
            f"Recent events: {json.dumps(events)}"
        )

    def _memory_prompt(
        self, simulation: Simulation, agent: AgentProfile, action: AgentActionResult
    ) -> str:
        return (
            "You model memory corrosion for the Cogniverse agent. "
            "Produce JSON {\"memory\": {\"distorted_memory\": string, \"reliability_delta\": float between -0.2 and 0}}. "
            "The distorted memory should reflect the agent's cognitive bias and emotional state.\n"
            f"Scenario: {simulation.scenario}\n"
            f"Agent bias: {agent.cognitive_bias}\n"
            f"Agent emotional state: {agent.emotional_state}\n"
            f"Most recent action: {action.action_summary}\n"
            f"Action detail: {action.detailed_log}"
        )

    def _fate_prompt(self, simulation: Simulation, user_prompt: str | None) -> str:
        recent = [
            {
                "turn": event.turn,
                "actor": event.actor_id,
                "summary": event.summary,
            }
            for event in simulation.events[-5:]
        ]
        return (
            "You introduce a dramatic Fate Weaver twist for the Cogniverse simulation. "
            "Return JSON {\"fate\": {\"title\": string, \"description\": string, \"impact\": string}}. "
            "The description should be vivid and influence future actions. "
            f"Scenario: {simulation.scenario}\n"
            f"Recent events: {json.dumps(recent)}\n"
            f"Fate Weaver input: {user_prompt or 'None'}"
        )

    # ------------------------------------------------------------------ #
    #  Defaults                                                          #
    # ------------------------------------------------------------------ #
    def _build_agent_profile(self, agent_data: Dict, index: int) -> AgentProfile:
        position = Position(
            x=random.uniform(-25.0, 25.0),
            y=random.uniform(-25.0, 25.0),
        )
        traits = self._normalise_traits(
            agent_data.get("traits"), fallback=agent_data.get("role", "operative")
        )
        profile_kwargs = {
            "name": agent_data.get("name", f"Agent {index + 1}"),
            "role": agent_data.get("role", "operative"),
            "persona": agent_data.get(
                "persona",
                "Pragmatic thinker adapting quickly to the unfolding scenario.",
            ),
            "cognitive_bias": agent_data.get("cognitive_bias", "status quo bias"),
            "emotional_state": agent_data.get("emotional_state", "focused"),
            "mbti": agent_data.get("mbti"),
            "skills": agent_data.get("skills") or ["strategy"],
            "biography": agent_data.get(
                "biography", "Experienced specialist with a shrouded past."
            ),
            "constraints": agent_data.get("constraints") or [
                "Avoid collateral damage"
            ],
            "quirks": agent_data.get("quirks") or ["Hums when thinking"],
            "motivation": agent_data.get(
                "motivation",
                "Seeks to secure a lasting advantage for their allies.",
            ),
            "secret_agenda": agent_data.get(
                "secret_agenda",
                "Acquire leverage tied to the scenario's central conflict.",
            ),
            "agenda_progress": float(agent_data.get("agenda_progress", 0.0)),
            "memory": agent_data.get(
                "memory", "Remembers the briefing with calm confidence."
            ),
            "position": position,
            "traits": traits,
        }
        return AgentProfile(**profile_kwargs)

    def _default_agents(self, count: int, scenario: str) -> List[AgentProfile]:
        defaults = []
        for idx in range(count):
            defaults.append(
                AgentProfile(
                    name=f"Fallback Agent {idx + 1}",
                    role="generalist",
                    persona=f"Resilient problem-solver forged by {scenario[:40] or 'unknown events'}.",
                    cognitive_bias="anchoring bias",
                    emotional_state="alert",
                    mbti="INTJ",
                    skills=["analysis", "negotiation"],
                    biography="Veteran operative called in when plans unravel.",
                    constraints=["Report all anomalies"],
                    quirks=["Stacks mission tokens in perfect rows"],
                    motivation="Uncover the truth hidden beneath conflicting reports.",
                    secret_agenda="Secure evidence that exposes hidden agendas.",
                    memory="Remembers the latest mission briefing with precision.",
                    position=Position(
                        x=random.uniform(-20.0, 20.0),
                        y=random.uniform(-20.0, 20.0),
                    ),
                    traits={"specialty": "stabiliser"},
                )
            )
        return defaults

    @staticmethod
    def _normalise_traits(
        value: Dict | List | str | int | float | None, *, fallback: str
    ) -> Dict[str, str | List[str]]:
        if isinstance(value, dict):
            normalised: Dict[str, str | List[str]] = {}
            for key, raw in value.items():
                if raw is None:
                    continue
                if isinstance(raw, (list, tuple, set)):
                    items = [str(item) for item in raw if item is not None]
                    if items:
                        normalised[key] = items
                else:
                    normalised[key] = str(raw)
            if normalised:
                return normalised
        elif value is not None:
            return {"value": str(value)}
        fallback_value = fallback or "operative"
        return {"specialty": fallback_value}
