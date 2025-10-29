#!/usr/bin/env python3
"""
Simple CLI for exercising the CogniVerse simulation API.

Usage:
    1. Start the FastAPI backend (e.g. `uvicorn backend.api:app --reload`).
    2. Run this script: `python scripts/simulation_cli.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class SimulationSession:
    base_url: str
    timeout: float
    simulation_id: Optional[str] = None

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = requests.request(method, url, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response.json()

    def create(self, scenario: str) -> Dict[str, Any]:
        payload = {
            "scenario": scenario,
            "custom_agents": [],
            "agent_profiles": [],
            "relationships": [],
        }
        data = self._request("POST", "/simulations", json=payload)
        self.simulation_id = data["simulation"]["id"]
        return data

    def fetch(self) -> Dict[str, Any]:
        if not self.simulation_id:
            raise RuntimeError("No simulation created yet.")
        return self._request("GET", f"/simulations/{self.simulation_id}")

    def advance(self, steps: int) -> Dict[str, Any]:
        if not self.simulation_id:
            raise RuntimeError("No simulation created yet.")
        payload = {"steps": steps}
        return self._request(
            "POST", f"/simulations/{self.simulation_id}/advance", json=payload
        )

    def fate(self, prompt: Optional[str]) -> Dict[str, Any]:
        if not self.simulation_id:
            raise RuntimeError("No simulation created yet.")
        payload = {"prompt": prompt}
        return self._request(
            "POST", f"/simulations/{self.simulation_id}/fate", json=payload
        )


def pretty_events(events: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for event in events:
        turn = event.get("turn")
        summary = event.get("summary", "")
        lines.append(f"- Turn {turn}: {summary}")
        details = event.get("details")
        if details:
            lines.append(f"    {details}")
    return "\n".join(lines) if lines else "  (no events yet)"


def print_simulation(simulation: Dict[str, Any]) -> None:
    sim = simulation["simulation"]
    print(f"\nSimulation {sim['id']} (turn {sim['turn']})")
    print(f"Scenario: {sim['scenario']}")
    print("\nAgents:")
    for idx, agent in enumerate(sim["agents"], start=1):
        print(f"  {idx}. {agent['name']} ({agent['role']}) – {agent['persona']}")
    print("\nRecent events:")
    print(pretty_events(sim.get("events", [])))
    print("\n---\n")


def interactive_loop(session: SimulationSession) -> None:
    menu = (
        "\nCommands:\n"
        "  create  - start a new simulation (requires scenario text)\n"
        "  show    - display the current simulation state\n"
        "  advance - advance the simulation by N turns\n"
        "  fate    - trigger a fate event (optional prompt)\n"
        "  exit    - quit\n"
    )
    print(menu)

    while True:
        try:
            command = input("command> ").strip().lower()
        except EOFError:
            print()
            break

        if command in {"exit", "quit"}:
            break
        if command == "create":
            scenario = input("Scenario description: ").strip()
            if not scenario:
                print("Scenario cannot be empty.")
                continue
            try:
                data = session.create(scenario)
                print_simulation(data)
            except Exception as exc:  # pragma: no cover - CLI convenience
                print(f"Create failed: {exc}")
        elif command == "show":
            try:
                data = session.fetch()
                print_simulation(data)
            except Exception as exc:
                print(f"Show failed: {exc}")
        elif command.startswith("advance"):
            parts = command.split()
            if len(parts) == 2 and parts[1].isdigit():
                steps = int(parts[1])
            else:
                raw = input("Steps to advance: ").strip()
                if not raw.isdigit():
                    print("Please provide a positive integer.")
                    continue
                steps = int(raw)
            try:
                data = session.advance(steps)
                print_simulation(data)
            except Exception as exc:
                print(f"Advance failed: {exc}")
        elif command == "fate":
            prompt = input("Optional fate prompt (leave blank for default): ").strip()
            try:
                data = session.fate(prompt or None)
                print_simulation(data)
            except Exception as exc:
                print(f"Fate failed: {exc}")
        else:
            print("Unknown command.")
            print(menu)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CLI helper for the CogniVerse Vertex-backed simulation API."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running backend (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout for API requests in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--scenario",
        help="Optionally create a simulation immediately with the provided scenario.",
    )
    args = parser.parse_args(argv)

    session = SimulationSession(base_url=args.base_url.rstrip("/"), timeout=args.timeout)

    if args.scenario:
        try:
            data = session.create(args.scenario)
            print_simulation(data)
        except Exception as exc:
            print(f"Failed to create simulation: {exc}")
            return 1

    try:
        interactive_loop(session)
    except KeyboardInterrupt:  # pragma: no cover - manual exit
        print("\nInterrupted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
