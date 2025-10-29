from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Dict

from ..models import SimulationStatus
from .engine import SimulationEngine

logger = logging.getLogger(__name__)


class SimulationAutoRunner:
    """Background task manager that advances simulations at fixed intervals."""

    def __init__(self, engine: SimulationEngine, interval: float = 4.0):
        self.engine = engine
        self.interval = max(0.5, interval)
        self._tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def start(self, simulation_id: str):
        """Ensure an auto-advance task is running for the simulation."""
        async with self._lock:
            if simulation_id in self._tasks:
                return
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._run(simulation_id))
            self._tasks[simulation_id] = task

    async def stop(self, simulation_id: str):
        """Cancel the auto-advance task for a simulation."""
        async with self._lock:
            task = self._tasks.pop(simulation_id, None)
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def stop_all(self):
        """Cancel all auto-advance tasks."""
        async with self._lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, simulation_id: str):
        """Background coroutine advancing turns until the simulation stops."""
        try:
            while True:
                await asyncio.sleep(self.interval)
                try:
                    simulation = await asyncio.to_thread(
                        self.engine.advance_turn, simulation_id
                    )
                except KeyError:
                    logger.info(
                        "Auto-runner stopped: simulation %s no longer exists.",
                        simulation_id,
                    )
                    break
                except RuntimeError as exc:
                    logger.info(
                        "Auto-runner halted for %s: %s", simulation_id, exc
                    )
                    break
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception(
                        "Auto-runner encountered error for %s: %s",
                        simulation_id,
                        exc,
                    )
                    continue

                if simulation.status != SimulationStatus.RUNNING:
                    logger.info(
                        "Auto-runner completed: simulation %s reached status %s.",
                        simulation_id,
                        simulation.status,
                    )
                    break
        except asyncio.CancelledError:
            pass
        finally:
            async with self._lock:
                self._tasks.pop(simulation_id, None)
