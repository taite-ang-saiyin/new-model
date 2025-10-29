from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .ai import (
    GeminiAIProvider,
    HuggingFaceSpaceProvider,
    LegacySystemProvider,
    MockAIProvider,
    VertexAIProvider,
)
from .config import (
    AIProviderType,
    auto_advance_enabled,
    get_auto_advance_interval,
    get_ai_provider_type,
)
from .repository import SimulationRepository
from .schemas import (
    AdvanceSimulationRequest,
    CreateSimulationRequest,
    FateWeaverRequest,
    SimulationResponse,
)
from .simulation import SimulationEngine
from .simulation.auto_runner import SimulationAutoRunner

logger = logging.getLogger("cogniverse.backend")


def _build_engine() -> SimulationEngine:
    provider_type = get_ai_provider_type()
    if provider_type == AIProviderType.LEGACY:
        try:
            provider = LegacySystemProvider()
            logger.info(
                "Using LegacySystemProvider (Model Architect + Narrative Weaver + Action Engine)."
            )
        except Exception as exc:
            logger.warning(
                "Legacy provider failed to initialize (%s); reverting to MockAIProvider.",
                exc,
            )
            provider = MockAIProvider()
    elif provider_type == AIProviderType.REMOTE:
        try:
            provider = HuggingFaceSpaceProvider()
            logger.info("Using HuggingFaceSpaceProvider (remote Hugging Face Space).")
        except Exception as exc:
            logger.warning(
                "Remote provider failed to initialize (%s); reverting to MockAIProvider.",
                exc,
            )
            provider = MockAIProvider()
    elif provider_type == AIProviderType.GEMINI:
        try:
            provider = GeminiAIProvider()
            logger.info("Using GeminiAIProvider (Google Gemini 2.5 Flash).")
        except Exception as exc:
            logger.warning(
                "Gemini provider failed to initialize (%s); reverting to MockAIProvider.",
                exc,
            )
            provider = MockAIProvider()
    elif provider_type == AIProviderType.VERTEX:
        try:
            provider = VertexAIProvider()
            logger.info("Using VertexAIProvider (Google Vertex AI Generative).")
        except Exception as exc:
            logger.warning(
                "Vertex provider failed to initialize (%s); reverting to MockAIProvider.",
                exc,
            )
            provider = MockAIProvider()
    else:
        provider = MockAIProvider()
        logger.info("Using MockAIProvider (offline mode).")
    repository = SimulationRepository()
    return SimulationEngine(provider=provider, repository=repository)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Cogniverse Backend",
        description="Backend simulation engine for Cogniverse narrative sandbox.",
        version="0.1.0",
    )
    engine = _build_engine()
    app.state.engine = engine
    if auto_advance_enabled():
        app.state.auto_runner = SimulationAutoRunner(
            engine=engine, interval=get_auto_advance_interval()
        )
        logger.info(
            "Auto-advance enabled (interval %.2fs).", get_auto_advance_interval()
        )
    else:
        app.state.auto_runner = None
        logger.info("Auto-advance disabled; use manual advance endpoint/button.")

    allowed_origins = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5500",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5500",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def _shutdown_autorunner():
        runner: SimulationAutoRunner | None = getattr(app.state, "auto_runner", None)
        if runner:
            await runner.stop_all()

    app.add_event_handler("shutdown", _shutdown_autorunner)

    @app.post(
        "/simulations",
        response_model=SimulationResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_simulation(payload: CreateSimulationRequest):
        try:
            simulation = await asyncio.to_thread(
                app.state.engine.create_simulation,
                scenario=payload.scenario,
                customizations=payload.custom_agents,
                agent_profiles=payload.agent_profiles,
                relationships=payload.relationships,
            )
            runner: SimulationAutoRunner | None = getattr(app.state, "auto_runner", None)
            if runner:
                await runner.start(simulation.id)
            return SimulationResponse(simulation=simulation)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Failed to create simulation")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    @app.get("/simulations/{simulation_id}", response_model=SimulationResponse)
    async def get_simulation(simulation_id: str):
        try:
            simulation = await asyncio.to_thread(
                app.state.engine.get_simulation, simulation_id
            )
            return SimulationResponse(simulation=simulation)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc

    @app.post("/simulations/{simulation_id}/advance", response_model=SimulationResponse)
    async def advance_simulation(
        simulation_id: str, payload: AdvanceSimulationRequest
    ):
        try:
            simulation = await asyncio.to_thread(
                app.state.engine.advance_turns,
                simulation_id,
                payload.steps,
            )
            return SimulationResponse(simulation=simulation)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.exception("Failed to advance simulation")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    @app.post("/simulations/{simulation_id}/fate", response_model=SimulationResponse)
    async def trigger_fate(simulation_id: str, payload: FateWeaverRequest):
        try:
            simulation = await asyncio.to_thread(
                app.state.engine.trigger_fate_event,
                simulation_id,
                payload.prompt,
            )
            return SimulationResponse(simulation=simulation)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.exception("Failed to trigger fate event")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    return app


# Instantiate default application for uvicorn entry point
app = create_app()
