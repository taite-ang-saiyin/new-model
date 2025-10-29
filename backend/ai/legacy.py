from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class LegacySystemProvider(AIProvider):
    """
    AIProvider implementation that stitches together the existing Model Architect,
    Narrative Weaver, and Action Engine modules that ship with CogniVerse.

    This adapter avoids remote calls and reuses the on-disk assets for routing,
    narrative analysis, and movement heuristics.
    """

    def __init__(self, seed: int | None = None):
        self._rand = random.Random(seed)
        self._setup_paths()
        self._router = self._init_router()
        self._analyze_story = self._load_story_analyzer()
        self._memory_tracker_cls = self._load_memory_tracker()
        self._movement_adapter = self._init_movement_adapter()
        self._verb_templates = [
            ("investigates", "evidence of"),
            ("negotiates with", "a faction about"),
            ("reconfigures", "critical systems for"),
            ("interrogates", "hidden motives tied to"),
            ("shields", "the team from"),
            ("models", "a risky projection for"),
        ]
        self._bias_phrases = {
            "confirmation bias": "only notices details that confirm prior beliefs.",
            "anchoring bias": "clings to the first interpretation, ignoring new input.",
            "loss aversion": "fixates on potential setbacks over gains.",
            "status quo bias": "prefers keeping the mission unchanged.",
            "overconfidence effect": "overstates personal success and control.",
        }

    def _setup_paths(self):
        root = Path(__file__).resolve().parents[2]
        model_architect = root / "model_architect"
        narrative_weaver = root / "narrative_weaver" / "src"
        action_engine = root / "action_engine"
        for path in (model_architect, narrative_weaver, action_engine):
            sys.path.insert(0, str(path))

    def _init_router(self):
        try:
            from optimized_router import OptimizedRouter  # type: ignore

            root = Path(__file__).resolve().parents[2]
            experts_dir = root / "model_architect" / "expert_training_logs"
            return OptimizedRouter(experts_dir=str(experts_dir))
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise RuntimeError("Failed to initialize Model Architect router.") from exc

    def _load_story_analyzer(self):
        try:
            from narrative_weaver.analysis_tools import analyze_story  # type: ignore

            return analyze_story
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import Narrative Weaver analysis tools."
            ) from exc

    def _load_memory_tracker(self):
        try:
            from narrative_weaver.memory_tracker import MemoryTracker  # type: ignore

            return MemoryTracker
        except Exception:
            return None

    def _init_movement_adapter(self):
        try:
            from action_engine.envs.character_env import MultiObjectiveCharacterEnv  # type: ignore

            env = MultiObjectiveCharacterEnv(state_dim=8, memory_len=4, max_steps=50)
            obs, _ = env.reset()
            return {"env": env, "obs": obs}
        except Exception:
            return None

    # --------------------------------------------------------------------- #
    # Agent creation                                                        #
    # --------------------------------------------------------------------- #
    def generate_agents(
        self, scenario: str, customizations: List[AgentCustomization]
    ) -> List[AgentProfile]:
        custom_map: Dict[int, AgentCustomization] = {
            item.slot: item for item in customizations
        }
        personas = [
            ("Coordinated Strategist", "systems coordination and long-range planning"),
            ("Empathic Mediator", "reading tensions between factions"),
            ("Unconventional Engineer", "rapid prototyping and field repairs"),
            ("Data Skeptic", "auditing hidden patterns and deception"),
            ("Rogue Explorer", "scouting hazardous zones alone"),
        ]
        biases = [
            "confirmation bias",
            "anchoring bias",
            "loss aversion",
            "status quo bias",
            "overconfidence effect",
        ]
        mbti_cycle = ["INTJ", "ENFJ", "ISTP", "ENTP", "INFJ"]
        skill_catalog = [
            ["macro-planning", "stacked simulations"],
            ["empathic sensing", "influence weaving"],
            ["rapid prototyping", "hazard containment"],
            ["anomaly auditing", "threat modeling"],
            ["terrain scouting", "stealth traversal"],
        ]
        biography_templates = [
            "Architected supply lines during the first orbital collapse.",
            "Brokered peace between rival enclaves under impossible odds.",
            "Built black-market rigs that kept fringe colonies breathing.",
            "Exposed doctored telemetry that nearly doomed a convoy.",
            "Mapped unstable corridors before any beacon network existed.",
        ]
        constraint_catalog = [
            ["Bound to share intel with command", "Must conserve drone assets"],
            ["Cannot escalate conflicts", "Requires consensus for big moves"],
            ["Needs access to fabrication bays", "Avoids politically sensitive tech"],
            ["Needs verifiable data", "Limited to cold channels"],
            ["Navigation relies on solar cues", "Communication blackout during jumps"],
        ]
        quirk_catalog = [
            ["Annotates every holo-map", "Speaks in mission codes even off-duty"],
            ["Carries antique envoy coins", "Mirrors conversation partners"],
            ["Collects broken circuit shards", "Whistles during calibrations"],
            ["Stacks dossiers by perceived truthfulness", "Clicks stylus when skeptical"],
            ["Marks waypoints with UV chalk", "Talks to drones like pets"],
        ]
        motivation_cycle = [
            "Prove that coordinated planning still matters.",
            "Shield bystanders from becoming leverage.",
            "Redeem a prototype that failed spectacularly.",
            "Reveal whatever truth is being suppressed.",
            "Chart a safe passage before the storm closes.",
        ]

        profiles: List[AgentProfile] = []
        for idx, (title, focus) in enumerate(personas):
            prompt = f"{scenario}. Focus area: {focus}."
            experts = self._router.route(prompt, top_k=3)
            features = self._router.semantic_analyzer.analyze(prompt)
            base_name = f"Agent {idx + 1}"
            profile_data: Dict[str, object] = {
                "name": base_name,
                "role": experts[0][0] if experts else "Generalist",
                "persona": f"{title} specialising in {focus}.",
                "cognitive_bias": biases[idx % len(biases)],
                "emotional_state": self._sentiment_to_emotion(
                    features.sentiment, default="focused"
                ),
                "mbti": mbti_cycle[idx % len(mbti_cycle)],
                "skills": skill_catalog[idx % len(skill_catalog)],
                "biography": biography_templates[idx % len(biography_templates)],
                "constraints": constraint_catalog[idx % len(constraint_catalog)],
                "quirks": quirk_catalog[idx % len(quirk_catalog)],
                "motivation": motivation_cycle[idx % len(motivation_cycle)],
                "secret_agenda": self._craft_secret_agenda(focus, experts),
                "memory": f"Remembers the {scenario.split()[0] if scenario else 'operation'} briefing with emphasis on {focus}.",
                "agenda_progress": round(self._rand.uniform(0.0, 0.15), 3),
                "traits": {
                    "dominant_experts": [expert for expert, _ in experts],
                    "intent": features.intent_type,
                    "domain": features.domain,
                },
                "position": Position(
                    x=self._rand.uniform(-20.0, 20.0),
                    y=self._rand.uniform(-20.0, 20.0),
                ),
            }
            if idx in custom_map:
                overrides = custom_map[idx]
                scalar_overrides = (
                    "name",
                    "role",
                    "persona",
                    "cognitive_bias",
                    "emotional_state",
                    "mbti",
                    "biography",
                    "motivation",
                )
                list_overrides = ("skills", "constraints", "quirks")
                for attr in scalar_overrides:
                    value = getattr(overrides, attr)
                    if value:
                        profile_data[attr] = value
                for attr in list_overrides:
                    value = getattr(overrides, attr)
                    if value is not None:
                        profile_data[attr] = value
            profiles.append(AgentProfile(**profile_data))
        return profiles

    def _sentiment_to_emotion(self, sentiment: float, default: str = "neutral") -> str:
        if sentiment > 0.4:
            return "optimistic"
        if sentiment > 0.1:
            return "focused"
        if sentiment < -0.4:
            return "defensive"
        if sentiment < -0.1:
            return "uneasy"
        return default

    def _craft_secret_agenda(self, focus: str, experts: List[Tuple[str, float]]) -> str:
        if not experts:
            return f"Advance personal leverage tied to {focus}."
        top = experts[0][0]
        return f"Leverage {top.lower()} insights to gain an edge in {focus}."

    # --------------------------------------------------------------------- #
    # Turn advancement                                                      #
    # --------------------------------------------------------------------- #
    def generate_agent_action(self, request: AgentActionRequest) -> AgentActionResult:
        agent = request.agent
        recent_text = " ".join(event.summary for event in request.recent_events[-4:])
        context_prompt = f"{request.scenario} Recent events: {recent_text}"
        experts = self._router.route(context_prompt or request.scenario, top_k=2)
        verb, obj = self._rand.choice(self._verb_templates)
        primary_expert = experts[0][0] if experts else agent.role
        action_summary = f"{agent.name} {verb} {obj} {primary_expert.lower()} expertise to counter unfolding tension."

        analysis = self._analyze_story([recent_text + " " + action_summary], [])
        tension = self._extract_tension(analysis)
        emotional_state = self._choose_emotion(agent.emotional_state, tension)
        agenda_delta = self._agenda_delta(agent, tension)
        new_position = self._move_agent(agent, tension)

        detailed_log = (
            f"{agent.name} reflects on the {primary_expert.lower()} guidance while {verb} {obj}, "
            f"shifting into a {emotional_state} mindset."
        )
        return AgentActionResult(
            action_summary=action_summary,
            detailed_log=detailed_log,
            emotional_state=emotional_state,
            agenda_progress_delta=agenda_delta,
            new_position=new_position,
        )

    def _extract_tension(self, analysis: Dict) -> float:
        try:
            basic = analysis.get("basic_analysis", {})
            first_act = next(iter(basic.values()), {})
            return float(first_act.get("tension_score", 0.4))
        except Exception:
            return 0.4

    def _choose_emotion(self, current: str, tension: float) -> str:
        if tension > 0.75:
            return "strained"
        if tension > 0.55:
            return "urgent"
        if tension < 0.25:
            return "composed"
        return current

    def _agenda_delta(self, agent: AgentProfile, tension: float) -> float:
        base = 0.05 if tension < 0.4 else -0.02
        modifier = 0.03 if "overconfidence" in agent.cognitive_bias else 0.0
        return round(base + modifier, 3)

    def _move_agent(self, agent: AgentProfile, tension: float) -> Position:
        if self._movement_adapter:
            env = self._movement_adapter["env"]
            obs = self._movement_adapter["obs"]
            try:
                action = 4 if tension > 0.6 else 2
                obs, reward, terminated, truncated, _ = env.step(action)
                self._movement_adapter["obs"] = obs
                dx = float(obs["state"][0]) * 3.0
                dy = float(obs["state"][1]) * 3.0
                return Position(
                    x=self._clamp(agent.position.x + dx, -100.0, 100.0),
                    y=self._clamp(agent.position.y + dy, -100.0, 100.0),
                )
            except Exception:
                pass
        # Fallback deterministic move
        delta = 6.0 if tension > 0.6 else 3.0
        dx = (
            delta
            * float(self._rand.choice([-1, 1]))
            * abs(self._rand.uniform(0.3, 1.0))
        )
        dy = (
            delta
            * float(self._rand.choice([-1, 1]))
            * abs(self._rand.uniform(0.3, 1.0))
        )
        return Position(
            x=self._clamp(agent.position.x + dx, -100.0, 100.0),
            y=self._clamp(agent.position.y + dy, -100.0, 100.0),
        )

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    # --------------------------------------------------------------------- #
    # Memory corrosion                                                      #
    # --------------------------------------------------------------------- #
    def corrode_memory(
        self, simulation: Simulation, agent: AgentProfile, action: AgentActionResult
    ) -> MemoryCorrosionResult:
        phrase = self._bias_phrases.get(
            agent.cognitive_bias.lower(), "reframes the story to feel in control."
        )
        distorted = (
            f"{agent.name} remembers the turn differently: {agent.name} {phrase} "
            f"They insist the latest move ({action.action_summary}) unfolded exactly as envisioned."
        )
        delta = -0.05 if "overconfidence" in agent.cognitive_bias.lower() else -0.04
        return MemoryCorrosionResult(
            distorted_memory=distorted,
            reliability_delta=delta,
        )

    # --------------------------------------------------------------------- #
    # Fate events                                                           #
    # --------------------------------------------------------------------- #
    def generate_fate_event(
        self, simulation: Simulation, user_prompt: Optional[str] = None
    ) -> FateEventResult:
        recent = " ".join(event.summary for event in simulation.events[-5:])
        experts = self._router.route(
            f"{simulation.scenario} Fate hook: {recent}", top_k=1
        )
        anchor = experts[0][0] if experts else "Unknown"
        title = f"{anchor} Resonance"
        description = f"A surge of {anchor.lower()} signals ripples through the scenario, forcing agents to reconcile hidden motives."
        if user_prompt:
            description = f"{description} Fate Weaver decree: {user_prompt.strip()}."
        impact = "Subsequent actions should factor in destabilised alliances."
        return FateEventResult(
            title=title,
            description=description,
            impact=impact,
        )
