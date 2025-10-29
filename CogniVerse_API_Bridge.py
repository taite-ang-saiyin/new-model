#!/usr/bin/env python3
"""
CogniVerse Unified API Bridge
Coordinates Model Architect, Narrative Weaver, and Action Engine
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add all system paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "model_architect"))
sys.path.append(str(current_dir / "narrative_weaver" / "src"))
sys.path.append(str(current_dir / "action_engine"))

logger = logging.getLogger("cogniverse_bridge")
logging.basicConfig(level=logging.INFO)


@dataclass
class CogniVerseRequest:
    """Unified request structure for all CogniVerse operations"""

    prompt: str
    operation_type: str  # "narrative", "action", "routing", "full_pipeline"
    genre: Optional[str] = None
    tone: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    expert_types: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


@dataclass
class CogniVerseResponse:
    """Unified response structure"""

    success: bool
    result: Any
    metadata: Dict[str, Any]
    error: Optional[str] = None


class CogniVerseBridge:
    """Main orchestrator for all CogniVerse systems"""

    def __init__(self):
        self.model_architect = None
        self.narrative_weaver = None
        self.action_engine = None
        self._initialize_systems()

    def _initialize_systems(self):
        """Initialize all available systems"""
        logger.info("Initializing CogniVerse systems...")

        # Initialize Model Architect (Router)
        try:
            from optimized_router import OptimizedRouter

            experts_dir = current_dir / "model_architect" / "expert_training_logs"
            if experts_dir.exists():
                self.model_architect = OptimizedRouter(experts_dir=str(experts_dir))
                logger.info("✅ Model Architect (Router) initialized")
            else:
                logger.warning("⚠️ Model Architect expert logs not found")
        except Exception as e:
            logger.warning(f"⚠️ Model Architect initialization failed: {e}")

        # Initialize Narrative Weaver
        try:
            from narrative_weaver.story_generator import get_generator

            self.narrative_weaver = get_generator()
            logger.info("✅ Narrative Weaver initialized")
        except Exception as e:
            logger.warning(f"⚠️ Narrative Weaver initialization failed: {e}")

        # Initialize Action Engine (if available)
        try:
            # Try to import Action Engine components
            from models.policy_networks import PolicyNetwork
            from envs.character_env import CharacterEnvironment

            self.action_engine = {"policy": PolicyNetwork, "env": CharacterEnvironment}
            logger.info("✅ Action Engine initialized")
        except Exception as e:
            logger.warning(f"⚠️ Action Engine initialization failed: {e}")

    def process_request(self, request: CogniVerseRequest) -> CogniVerseResponse:
        """Process unified requests across all systems"""
        try:
            if request.operation_type == "narrative":
                return self._handle_narrative_request(request)
            elif request.operation_type == "routing":
                return self._handle_routing_request(request)
            elif request.operation_type == "action":
                return self._handle_action_request(request)
            elif request.operation_type == "full_pipeline":
                return self._handle_full_pipeline(request)
            else:
                return CogniVerseResponse(
                    success=False,
                    result=None,
                    metadata={},
                    error=f"Unknown operation type: {request.operation_type}",
                )
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return CogniVerseResponse(
                success=False,
                result=None,
                metadata={"error_type": type(e).__name__},
                error=str(e),
            )

    def _handle_narrative_request(
        self, request: CogniVerseRequest
    ) -> CogniVerseResponse:
        """Handle narrative generation requests"""
        if not self.narrative_weaver:
            return CogniVerseResponse(
                success=False,
                result=None,
                metadata={},
                error="Narrative Weaver not available",
            )

        try:
            # Use Narrative Weaver's generate_story function
            from narrative_weaver.story_generator import generate_story

            story, acts, memory_summary = generate_story(
                prompt=request.prompt,
                genre=request.genre or "general",
                tone=request.tone or "neutral",
            )

            return CogniVerseResponse(
                success=True,
                result={"story": story, "acts": acts, "memory_summary": memory_summary},
                metadata={
                    "system": "narrative_weaver",
                    "acts_count": len(acts),
                    "story_length": len(story),
                },
            )
        except Exception as e:
            return CogniVerseResponse(
                success=False,
                result=None,
                metadata={},
                error=f"Narrative generation failed: {e}",
            )

    def _handle_routing_request(self, request: CogniVerseRequest) -> CogniVerseResponse:
        """Handle expert routing requests"""
        if not self.model_architect:
            return CogniVerseResponse(
                success=False,
                result=None,
                metadata={},
                error="Model Architect (Router) not available",
            )

        try:
            top_k = len(request.expert_types) if request.expert_types else 3
            selected_experts = self.model_architect.route(request.prompt, top_k=top_k)

            return CogniVerseResponse(
                success=True,
                result={
                    "selected_experts": selected_experts,
                    "routing_decision": (
                        self.model_architect.routing_history[-1]
                        if self.model_architect.routing_history
                        else None
                    ),
                },
                metadata={
                    "system": "model_architect",
                    "total_experts": len(selected_experts),
                },
            )
        except Exception as e:
            return CogniVerseResponse(
                success=False, result=None, metadata={}, error=f"Routing failed: {e}"
            )

    def _handle_action_request(self, request: CogniVerseRequest) -> CogniVerseResponse:
        """Handle action/decision requests"""
        if not self.action_engine:
            return CogniVerseResponse(
                success=False,
                result=None,
                metadata={},
                error="Action Engine not available",
            )

        # Placeholder for Action Engine integration
        return CogniVerseResponse(
            success=True,
            result={"action": "placeholder_action", "confidence": 0.8},
            metadata={
                "system": "action_engine",
                "note": "Action Engine integration pending",
            },
        )

    def _handle_full_pipeline(self, request: CogniVerseRequest) -> CogniVerseResponse:
        """Handle full pipeline: Route → Generate → Act"""
        pipeline_results = {}

        # Step 1: Route to experts
        if self.model_architect:
            try:
                routing_result = self._handle_routing_request(request)
                if routing_result.success:
                    pipeline_results["routing"] = routing_result.result
                    # Extract expert guidance for narrative generation
                    selected_experts = routing_result.result["selected_experts"]
                    expert_names = [expert for expert, score in selected_experts]
                    logger.info(f"Pipeline routing selected: {expert_names}")
            except Exception as e:
                logger.warning(f"Routing step failed: {e}")

        # Step 2: Generate narrative
        if self.narrative_weaver:
            try:
                narrative_result = self._handle_narrative_request(request)
                if narrative_result.success:
                    pipeline_results["narrative"] = narrative_result.result
            except Exception as e:
                logger.warning(f"Narrative step failed: {e}")

        # Step 3: Action planning (placeholder)
        if self.action_engine:
            try:
                action_result = self._handle_action_request(request)
                if action_result.success:
                    pipeline_results["action"] = action_result.result
            except Exception as e:
                logger.warning(f"Action step failed: {e}")

        return CogniVerseResponse(
            success=len(pipeline_results) > 0,
            result=pipeline_results,
            metadata={
                "system": "full_pipeline",
                "steps_completed": list(pipeline_results.keys()),
            },
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all systems"""
        return {
            "model_architect": self.model_architect is not None,
            "narrative_weaver": self.narrative_weaver is not None,
            "action_engine": self.action_engine is not None,
            "timestamp": str(Path(__file__).stat().st_mtime),
        }


# Global bridge instance
_bridge_instance = None


def get_cogniverse_bridge() -> CogniVerseBridge:
    """Get singleton bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = CogniVerseBridge()
    return _bridge_instance


def main():
    """CLI interface for CogniVerse Bridge"""
    import argparse

    parser = argparse.ArgumentParser(description="CogniVerse Unified API Bridge")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument(
        "--operation",
        choices=["narrative", "routing", "action", "full_pipeline"],
        default="full_pipeline",
        help="Operation type",
    )
    parser.add_argument("--genre", help="Story genre")
    parser.add_argument("--tone", help="Story tone")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    bridge = get_cogniverse_bridge()

    request = CogniVerseRequest(
        prompt=args.prompt,
        operation_type=args.operation,
        genre=args.genre,
        tone=args.tone,
    )

    response = bridge.process_request(request)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "request": {
                        "prompt": request.prompt,
                        "operation_type": request.operation_type,
                        "genre": request.genre,
                        "tone": request.tone,
                    },
                    "response": {
                        "success": response.success,
                        "result": response.result,
                        "metadata": response.metadata,
                        "error": response.error,
                    },
                },
                f,
                indent=2,
            )
        print(f"Results saved to {args.output}")
    else:
        print(
            json.dumps(
                {
                    "success": response.success,
                    "result": response.result,
                    "metadata": response.metadata,
                    "error": response.error,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
