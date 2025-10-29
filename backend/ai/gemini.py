from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, List

import requests

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
REQUEST_COUNTER = 0
MAX_OUTPUT_TOKENS = int(os.getenv("COGNIVERSE_GEMINI_MAX_OUTPUT_TOKENS", "8192"))


class GeminiAIProvider(AIProvider):
    """
    Google Gemini 2.5 Flash backed provider.

    Required environment variables:
      - GOOGLE_API_KEY (from Google AI Studio)
      - Optional: COGNIVERSE_GEMINI_MODEL (defaults to gemini-2.5-flash)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 45.0,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not configured; cannot initialize Gemini provider."
            )
        self.model = model or os.getenv("COGNIVERSE_GEMINI_MODEL", "gemini-2.5-flash")
        self.timeout = timeout
        self.session = requests.Session()
        # Ignore system proxy settings to avoid blocked corporate proxies unless explicitly configured.
        self.session.trust_env = False
        self.base_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )

    # ------------------------------------------------------------------ #
    # Helper utilities                                                   #
    # ------------------------------------------------------------------ #
    def _invoke(
        self,
        prompt: str,
        *,
        max_output_tokens: int | None = None,
        attempt: int = 1,
    ) -> str:
        global REQUEST_COUNTER
        REQUEST_COUNTER += 1
        logger.info("Gemini requests so far: %s", REQUEST_COUNTER)
        tokens = min(max_output_tokens or MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.75,
                "topP": 0.9,
                "maxOutputTokens": tokens,
                "responseMimeType": "application/json",
            },
        }
        params = {"key": self.api_key}
        response = self.session.post(
            self.base_url, params=params, json=payload, timeout=self.timeout
        )
        if response.status_code != 200:
            logger.error("Gemini API error %s: %s", response.status_code, response.text)
            response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        for candidate in candidates:
            finish_reason = candidate.get("finishReason")
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            for part in parts:
                text = part.get("text")
                if isinstance(text, str):
                    return text.strip()
            if finish_reason == "MAX_TOKENS" and tokens < MAX_OUTPUT_TOKENS and attempt < 3:
                logger.warning(
                    "Gemini output hit MAX_TOKENS at %s tokens on attempt %s; retrying with %s tokens.",
                    tokens,
                    attempt,
                    min(tokens + 256, MAX_OUTPUT_TOKENS),
                )
                return self._invoke(
                    prompt,
                    max_output_tokens=min(tokens + 256, MAX_OUTPUT_TOKENS),
                    attempt=attempt + 1,
                )
        raise ValueError(f"Gemini response missing text payload: {data!r}")

    @staticmethod
    def _clean_json(text: str) -> Dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            sections = cleaned.split("\n", 1)
            cleaned = sections[1] if len(sections) > 1 else sections[0]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini JSON output: %s", cleaned)
            raise ValueError("Gemini response did not contain valid JSON") from exc

    def _call_json(
        self,
        prompt: str,
        *,
        max_output_tokens: int | None,
        log_label: str,
        retries: int = 3,
    ) -> Dict:
        """Invoke Gemini and ensure valid JSON, retrying with stricter instructions."""
        base_prompt = prompt
        extra_instruction = (
            "\n\nCRITICAL: Output strict JSON only. Do not include backticks, "
            "comments, or partial objects. Ensure all quotes and braces are closed."
        )
        for attempt in range(1, retries + 1):
            raw = self._invoke(
                prompt,
                max_output_tokens=max_output_tokens,
                attempt=1,
            )
            try:
                return self._clean_json(raw)
            except ValueError as exc:
                logger.warning(
                    "%s attempt %s failed to produce valid JSON (%s).",
                    log_label,
                    attempt,
                    exc,
                )
                if attempt == retries:
                    raise
                prompt = f"{base_prompt}{extra_instruction}"
                if max_output_tokens is not None:
                    max_output_tokens = min(
                        max_output_tokens + 256, MAX_OUTPUT_TOKENS
                    )

    @staticmethod
    def _apply_customisations(
        profiles: List[AgentProfile], customisations: List[AgentCustomization]
    ) -> List[AgentProfile]:
        overrides = {item.slot: item for item in customisations}
        for idx, profile in enumerate(profiles):
            override = overrides.get(idx)
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
        return profiles

    @staticmethod
    def _summarise_events(events, limit: int = 3, max_chars: int = 220) -> List[str]:
        """Return concise string summaries for recent events."""
        summaries: List[str] = []
        for event in events[-limit:]:
            summary = f"{event.type.value.upper()} turn {event.turn}: {event.summary or ''}"
            details = (event.details or "").strip()
            if details:
                summary = f"{summary} — {details}"
            if len(summary) > max_chars:
                summary = summary[: max_chars - 3] + "..."
            summaries.append(summary)
        return summaries

    # ------------------------------------------------------------------ #
    # Prompt builders                                                    #
    # ------------------------------------------------------------------ #
    def _agent_prompt(
        self, scenario: str, customisations: List[AgentCustomization]
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
            for item in customisations
        ]
        return (
            "You are the agent architect for the Cogniverse narrative simulator. "
            "Return JSON ONLY in the form {\"agents\": [five agent objects]}. "
            "Each agent object MUST include: name, role, persona, cognitive_bias, "
            "emotional_state, mbti, skills (array), biography, constraints (array), "
            "quirks (array), motivation, secret_agenda, agenda_progress (float), "
            "memory (first-person sentence), traits (object). "
            f"\nScenario: {scenario}\nUser customisations: {json.dumps(custom)}"
        )

    def _action_prompt(self, request: AgentActionRequest) -> str:
        agent = request.agent
        agent_brief = {
            "name": agent.name,
            "role": agent.role,
            "persona": agent.persona,
            "cognitive_bias": agent.cognitive_bias,
            "emotional_state": agent.emotional_state,
            "agenda_progress": round(agent.agenda_progress, 3),
            "last_action": agent.last_action,
            "memory": agent.corroded_memory or agent.memory,
            "position": {
                "x": round(agent.position.x, 2),
                "y": round(agent.position.y, 2),
            },
        }
        events = self._summarise_events(request.recent_events, limit=3)
        events_text = "\n".join(f"- {line}" for line in events) or "- No recent events."
        return (
            "You simulate a single agent turn in the Cogniverse world. "
            "Respond with JSON {\"action\": { ... }} containing: "
            "action_summary (sentence), detailed_log (2-3 sentences), emotional_state, "
            "agenda_progress_delta (float -0.2..0.2), new_position {\"x\": float, \"y\": float}. "
            "Stick to the agent's persona and the recent events.\n"
            f"Scenario: {request.scenario}\n"
            f"Turn index: {request.turn_index}\n"
            f"Agent profile summary: {json.dumps(agent_brief)}\n"
            f"Recent events:\n{events_text}"
        )

    def _memory_prompt(
        self, simulation: Simulation, agent: AgentProfile, action: AgentActionResult
    ) -> str:
        recent = self._summarise_events(simulation.events, limit=2)
        recent_text = "\n".join(f"- {line}" for line in recent) or "- None"
        return (
            "You apply memory corrosion to an agent in Cogniverse. "
            "Return JSON {\"memory\": {\"distorted_memory\": string, \"reliability_delta\": float}}. "
            "The distorted memory should reflect the cognitive bias and emotional state.\n"
            f"Scenario: {simulation.scenario}\n"
            f"Cognitive bias: {agent.cognitive_bias}\n"
            f"Emotional state: {agent.emotional_state}\n"
            f"Recent action: {action.action_summary}\n"
            f"Details: {action.detailed_log}\n"
            f"Recent context:\n{recent_text}"
        )

    def _fate_prompt(self, simulation: Simulation, user_prompt: str | None) -> str:
        recent = self._summarise_events(simulation.events, limit=4)
        recent_text = "\n".join(f"- {line}" for line in recent) or "- No major events yet."
        return (
            "You craft a Fate Weaver twist for Cogniverse. "
            "Return JSON {\"fate\": {\"title\": string, \"description\": string, \"impact\": string}}. "
            "Ensure the description is dramatic and the impact guides future actions.\n"
            f"Scenario: {simulation.scenario}\n"
            f"Recent events:\n{recent_text}\n"
            f"Fate Weaver input: {user_prompt or 'None'}"
        )

    # ------------------------------------------------------------------ #
    # AIProvider interface                                               #
    # ------------------------------------------------------------------ #
    def generate_agents(
        self, scenario: str, customizations: List[AgentCustomization]
    ) -> List[AgentProfile]:
        prompt = self._agent_prompt(scenario, customizations)
        payload = self._call_json(
            prompt, max_output_tokens=MAX_OUTPUT_TOKENS, log_label="generate_agents"
        )
        agents = payload.get("agents") or []
        profiles: List[AgentProfile] = []
        for idx, agent_data in enumerate(agents[:5]):
            profile = self._build_profile(agent_data, idx, scenario)
            profiles.append(profile)
        if len(profiles) < 5:
            profiles.extend(self._default_agents(5 - len(profiles), scenario))
        return self._apply_customisations(profiles[:5], customizations)

    def generate_agent_action(self, request: AgentActionRequest) -> AgentActionResult:
        prompt = self._action_prompt(request)
        payload = self._call_json(
            prompt, max_output_tokens=MAX_OUTPUT_TOKENS, log_label="agent_action"
        )
        data = payload.get("action") or {}
        position = data.get("new_position") or {}
        return AgentActionResult(
            action_summary=data.get(
                "action_summary", f"{request.agent.name} reflects before moving."
            ),
            detailed_log=data.get(
                "detailed_log",
                "The agent adapts their strategy in response to emerging details.",
            ),
            emotional_state=data.get("emotional_state", request.agent.emotional_state),
            agenda_progress_delta=float(data.get("agenda_progress_delta", 0.0)),
            new_position=Position(
                x=float(position.get("x", request.agent.position.x)),
                y=float(position.get("y", request.agent.position.y)),
            ),
        )

    def corrode_memory(
        self, simulation: Simulation, agent: AgentProfile, action: AgentActionResult
    ) -> MemoryCorrosionResult:
        prompt = self._memory_prompt(simulation, agent, action)
        payload = self._call_json(
            prompt, max_output_tokens=MAX_OUTPUT_TOKENS, log_label="memory_corrosion"
        )
        memory = payload.get("memory") or {}
        distorted = memory.get(
            "distorted_memory",
            f"{agent.name} insists events happened exactly as planned.",
        )
        reliability = float(memory.get("reliability_delta", -0.05))
        return MemoryCorrosionResult(
            distorted_memory=distorted, reliability_delta=reliability
        )

    def generate_fate_event(
        self, simulation: Simulation, user_prompt: str | None = None
    ) -> FateEventResult:
        prompt = self._fate_prompt(simulation, user_prompt)
        payload = self._call_json(
            prompt, max_output_tokens=MAX_OUTPUT_TOKENS, log_label="fate_event"
        )
        fate = payload.get("fate") or {}
        return FateEventResult(
            title=fate.get("title", "Catalyst Surge"),
            description=fate.get(
                "description",
                "A jolt of unforeseen intelligence forces alliances to reshuffle.",
            ),
            impact=fate.get(
                "impact",
                "Subsequent actions should reference the destabilised field of play.",
            ),
        )

    # ------------------------------------------------------------------ #
    # Defaults                                                           #
    # ------------------------------------------------------------------ #
    def _build_profile(
        self, agent_data: Dict, index: int, scenario: str
    ) -> AgentProfile:
        position = Position(
            x=random.uniform(-25.0, 25.0),
            y=random.uniform(-25.0, 25.0),
        )
        traits = self._normalise_traits(
            agent_data.get("traits"), fallback=agent_data.get("role", "strategist")
        )
        return AgentProfile(
            name=agent_data.get("name", f"Agent {index + 1}"),
            role=agent_data.get("role", "operative"),
            persona=agent_data.get(
                "persona",
                "Observant strategist who adapts quickly to shifts in the mission.",
            ),
            cognitive_bias=agent_data.get("cognitive_bias", "anchoring bias"),
            emotional_state=agent_data.get("emotional_state", "composed"),
            mbti=agent_data.get("mbti"),
            skills=agent_data.get("skills") or ["analysis"],
            biography=agent_data.get(
                "biography", "Veteran mission lead drawn back into the fold."
            ),
            constraints=agent_data.get("constraints") or ["Report anomalies"],
            quirks=agent_data.get("quirks") or ["Keeps a precise mission journal"],
            motivation=agent_data.get(
                "motivation", "Wants the team to stay three steps ahead."
            ),
            secret_agenda=agent_data.get(
                "secret_agenda",
                "Secure leverage that protects their allies from external pressure.",
            ),
            agenda_progress=float(agent_data.get("agenda_progress", 0.0)),
            memory=agent_data.get(
                "memory", "Remembers the morning briefing with quiet confidence."
            ),
            position=position,
            traits=traits,
        )

    def _default_agents(self, count: int, scenario: str) -> List[AgentProfile]:
        defaults: List[AgentProfile] = []
        for idx in range(count):
            defaults.append(
                AgentProfile(
                    name=f"Fallback Agent {idx + 1}",
                    role="generalist",
                    persona=f"Versatile operative shaped by {scenario[:40] or 'classified events'}.",
                    cognitive_bias="confirmation bias",
                    emotional_state="alert",
                    mbti="ENTJ",
                    skills=["coordination", "rapid analysis"],
                    biography="Cross-trained specialist assigned to stabilise the mission.",
                    constraints=["Maintain civilian safety"],
                    quirks=["Sketches timeline diagrams mid-briefing"],
                    motivation="Keep the team aligned around the long-game objective.",
                    secret_agenda="Collect proof of hidden saboteurs.",
                    memory="Mentally replays briefing highlights before acting.",
                    position=Position(
                        x=random.uniform(-15.0, 15.0),
                        y=random.uniform(-15.0, 15.0),
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
        fallback_value = fallback or "strategist"
        return {"specialty": fallback_value}
