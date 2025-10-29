#!/usr/bin/env python3
"""
CogniVerse Shared Configuration
Centralized configuration for all systems
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class CogniVerseConfig:
    """Centralized configuration manager"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "cogniverse_config.json"
        self.config = self._load_default_config()
        self._load_config_file()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "systems": {
                "model_architect": {
                    "enabled": True,
                    "experts_dir": "model_architect/expert_training_logs",
                    "router_type": "optimized",
                    "top_k": 3,
                },
                "narrative_weaver": {
                    "enabled": True,
                    "model": {
                        "hf_model": "gpt2",
                        "max_context_tokens": 2048,
                        "default_temperature": 0.8,
                        "default_top_p": 0.9,
                    },
                    "generation": {
                        "scenes_per_act": 1,
                        "beats_per_scene": 3,
                        "max_length_per_beat": 160,
                    },
                },
                "action_engine": {
                    "enabled": True,
                    "model_path": "action_engine/models",
                    "config_path": "action_engine/config",
                },
            },
            "api": {"host": "localhost", "port": 8080, "timeout": 30, "max_retries": 3},
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "cogniverse.log",
            },
            "cache": {"enabled": True, "max_size": 1000, "ttl": 3600},
        }

    def _load_config_file(self):
        """Load configuration from file if it exists"""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                    self._merge_config(file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")

    def _merge_config(self, file_config: Dict[str, Any]):
        """Recursively merge file config with default config"""

        def merge_dict(default: dict, override: dict):
            for key, value in override.items():
                if (
                    key in default
                    and isinstance(default[key], dict)
                    and isinstance(value, dict)
                ):
                    merge_dict(default[key], value)
                else:
                    default[key] = value

        merge_dict(self.config, file_config)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'systems.narrative_weaver.model.hf_model')"""
        keys = key_path.split(".")
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split(".")
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def save(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config file {self.config_file}: {e}")

    def get_system_config(self, system_name: str) -> Dict[str, Any]:
        """Get configuration for a specific system"""
        return self.get(f"systems.{system_name}", {})

    def is_system_enabled(self, system_name: str) -> bool:
        """Check if a system is enabled"""
        return self.get(f"systems.{system_name}.enabled", False)

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for Narrative Weaver"""
        return self.get("systems.narrative_weaver.model", {})

    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration for Narrative Weaver"""
        return self.get("systems.narrative_weaver.generation", {})

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.get("api", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get("logging", {})


# Global config instance
_config_instance = None


def get_config() -> CogniVerseConfig:
    """Get singleton config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = CogniVerseConfig()
    return _config_instance


def setup_environment():
    """Setup environment variables from config"""
    config = get_config()

    # Set Narrative Weaver environment variables
    model_config = config.get_model_config()
    if model_config.get("hf_model"):
        os.environ.setdefault("HF_MODEL", model_config["hf_model"])
    if model_config.get("max_context_tokens"):
        os.environ.setdefault(
            "MAX_CONTEXT_TOKENS", str(model_config["max_context_tokens"])
        )

    # Set logging level
    logging_config = config.get_logging_config()
    if logging_config.get("level"):
        os.environ.setdefault("LOG_LEVEL", logging_config["level"])


def create_sample_config():
    """Create a sample configuration file"""
    config = CogniVerseConfig()
    config.save()
    print(f"Sample configuration created: {config.config_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CogniVerse Configuration Manager")
    parser.add_argument(
        "--create-sample", action="store_true", help="Create sample config file"
    )
    parser.add_argument("--get", help="Get config value (dot notation)")
    parser.add_argument(
        "--set", nargs=2, metavar=("KEY", "VALUE"), help="Set config value"
    )

    args = parser.parse_args()

    if args.create_sample:
        create_sample_config()
    elif args.get:
        config = get_config()
        value = config.get(args.get)
        print(json.dumps(value, indent=2))
    elif args.set:
        config = get_config()
        key, value = args.set
        # Try to parse value as JSON, fallback to string
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        config.set(key, parsed_value)
        config.save()
        print(f"Set {key} = {parsed_value}")
    else:
        config = get_config()
        print(json.dumps(config.config, indent=2))
