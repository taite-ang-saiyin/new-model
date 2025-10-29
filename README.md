# CogniVerse - Unified AI System

A unified API bridge that coordinates three powerful AI systems:
- **Model Architect**: Expert routing and Mixture of Experts (MoE) system
- **Narrative Weaver**: Advanced story generation with dramatic structure
- **Action Engine**: Reinforcement learning for decision making

## 🚀 Quick Start

### 1. Setup
```bash
# Run the setup script (installs all dependencies)
python setup.py

# Or manually install requirements
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the System
```bash
# Interactive mode (recommended for first time)
python main.py interactive

# Quick test of all systems
python main.py test

# Direct API usage
python CogniVerse_API_Bridge.py --prompt "A detective investigates a crime" --operation full_pipeline
```

## 📁 System Architecture

```
CogniVerse1/
├── CogniVerse_API_Bridge.py    # Main orchestrator
├── main.py                     # Interactive CLI
├── config.py                   # Shared configuration
├── setup.py                    # Installation script
├── requirements.txt            # Unified dependencies
├── docs/                       # Project documentation (PDF, reports)
├── logs/                       # Runtime and quality logs
├── action_engine/              # Reinforcement learning
├── model_architect/            # Expert routing system
├── narrative_weaver/           # Story generation
├── backend/                    # Simulation API backend
└── cache/                      # Cached assets
```

## 🎯 Usage Examples

### Interactive Mode
```bash
python main.py interactive
```
Choose from:
1. Generate narrative story
2. Route to experts  
3. Full pipeline (route + generate)
4. System status
5. Exit

### API Bridge Usage
```python
from CogniVerse_API_Bridge import get_cogniverse_bridge, CogniVerseRequest

bridge = get_cogniverse_bridge()

# Generate a story
request = CogniVerseRequest(
    prompt="A detective investigates a mysterious crime",
    operation_type="narrative",
    genre="mystery",
    tone="dramatic"
)
response = bridge.process_request(request)
print(response.result["story"])

# Route to experts
request = CogniVerseRequest(
    prompt="How should I handle this emotional situation?",
    operation_type="routing"
)
response = bridge.process_request(request)
print(response.result["selected_experts"])
```

### Command Line Usage
```bash
# Generate narrative
python CogniVerse_API_Bridge.py --prompt "A hero's journey" --operation narrative --genre fantasy --tone epic

# Route to experts
python CogniVerse_API_Bridge.py --prompt "Technical problem solving" --operation routing

# Full pipeline
python CogniVerse_API_Bridge.py --prompt "A business decision" --operation full_pipeline --output results.json
```

## ⚙️ Configuration

The system uses a centralized configuration file (`cogniverse_config.json`):

```json
{
  "systems": {
    "model_architect": {
      "enabled": true,
      "experts_dir": "model_architect/expert_training_logs",
      "top_k": 3
    },
    "narrative_weaver": {
      "enabled": true,
      "model": {
        "hf_model": "gpt2",
        "default_temperature": 0.8
      }
    },
    "action_engine": {
      "enabled": true
    }
  }
}
```

### Configuration Management
```bash
# View current config
python config.py

# Get specific value
python config.py --get systems.narrative_weaver.model.hf_model

# Set value
python config.py --set systems.narrative_weaver.model.hf_model "microsoft/phi-2"

# Create sample config
python config.py --create-sample
```

## 🔧 System Integration

### How the Systems Work Together

1. **Model Architect** analyzes input and routes to appropriate experts
2. **Narrative Weaver** generates content using expert guidance
3. **Action Engine** provides decision-making capabilities (when available)

### Expert Routing Integration
The Narrative Weaver automatically uses Model Architect's router when available:
- Each story beat is analyzed by the router
- Selected experts influence generation parameters
- Expert guidance is added to prompts
- Temperature/top_p are adjusted based on expert types

### Fallback Behavior
- If Model Architect is unavailable: Narrative Weaver runs normally
- If Action Engine is unavailable: System continues without action planning
- If OpenAI API key is missing: Falls back to local HuggingFace models

## 🎭 Available Operations

### `narrative`
Generate a complete story with dramatic structure
- **Input**: prompt, genre, tone
- **Output**: story, acts, memory summary

### `routing` 
Route input to appropriate experts
- **Input**: text prompt
- **Output**: selected experts with scores

### `action`
Get action recommendations (placeholder)
- **Input**: decision context
- **Output**: action suggestions

### `full_pipeline`
Complete workflow: route → generate → act
- **Input**: prompt, genre, tone
- **Output**: routing results + generated story + action suggestions

## 🛠️ Development

### Adding New Systems
1. Add system configuration to `config.py`
2. Implement initialization in `CogniVerseBridge._initialize_systems()`
3. Add request handler in `CogniVerseBridge._handle_*_request()`
4. Update operation types and CLI options

### Customizing Expert Routing
Modify the expert-to-guidance mapping in `narrative_weaver/src/narrative_weaver/story_generator.py`:
```python
def _map_experts_to_guidance(selected: List[Tuple[str, float]]) -> Tuple[str, float, float]:
    # Add your custom expert mappings here
```

## 📊 System Status

Check which systems are available:
```python
bridge = get_cogniverse_bridge()
status = bridge.get_system_status()
print(status)
```

## 🐛 Troubleshooting

### Common Issues

1. **"Model Architect not found"**
   - Ensure `model_architect/optimized_router.py` exists
   - Check that `expert_training_logs` directory is present

2. **"Narrative Weaver initialization failed"**
   - Install requirements: `pip install -r requirements.txt`
   - Download spaCy model: `python -m spacy download en_core_web_sm`

3. **"Action Engine not available"**
   - This is expected if Action Engine dependencies aren't installed
   - System will continue without action planning

4. **Import errors**
   - Run setup: `python setup.py`
   - Check Python version (3.8+ required)

### Logs
- System logs: `cogniverse.log`
- Individual system logs in respective directories

## 📝 License

This project combines multiple AI systems for research and development purposes.

## 🤝 Contributing

1. Keep the three systems separate but coordinated
2. Use the unified API bridge for integration
3. Follow the configuration system for settings
4. Test with `python main.py test` before submitting changes




## Backend Simulation API (No Frontend Required)

The narrative sandbox backend described in Cogniverse.pdf lives under `backend/` and runs as a FastAPI service.

### Start the service

```bash
uvicorn backend.api:app --reload
```

This launches the API on http://localhost:8000 with an in-memory simulation store.

### Core endpoints

- `POST /simulations` - create a simulation from a scenario plus optional agent customisations (slots `0-4`).
- `GET /simulations/{id}` - retrieve live state including agents, coordinates, agendas, and event log.
- `POST /simulations/{id}/advance` - progress one or more turns; Memory Corrosion runs after each action.
- `POST /simulations/{id}/fate` - inject a Fate Weaver twist, optionally adding your own dramatic prompt.

#### Custom agent payload

Each object in `custom_agents` targets one of the five agent slots and can override narrative-facing metadata. Arrays accept up to a handful of short entries.

| Field | Type | Notes |
| --- | --- | --- |
| `slot` | int | Which default agent (0-4) to override. |
| `name` | string | Display name shown in the HUD/logs. |
| `role` | string | Summary of the agent’s functional focus. |
| `persona` | string | Optional flavor text applied to introductions. |
| `cognitive_bias` | string | Guides memory corrosion tone. |
| `emotional_state` | string | Sets the starting mood label. |
| `mbti` | string | 16-type shorthand (e.g., `INTJ`). |
| `skills` | string[] | Comma-separated on the frontend, but sent as an array (e.g., `["diplomacy","systems thinking"]`). |
| `biography` | string | One paragraph that explains the agent’s backstory. |
| `constraints` | string[] | Things the agent refuses to violate. |
| `quirks` | string[] | Fun behavioral ticks for UI display. |
| `motivation` | string | Single sentence on what drives them. |

Example `POST /simulations` body:

```json
{
  "scenario": "A rogue AI citadel fractures the orbital blockade.",
  "custom_agents": [
    {
      "slot": 0,
      "name": "Lyra Vance",
      "role": "Strategic Diplomat",
      "mbti": "INTJ",
      "skills": ["mediations", "protocol design"],
      "biography": "Former envoy who diffused the Delta Pact mutiny.",
      "constraints": ["Never break parley", "Keep civilian losses near zero"],
      "quirks": ["Quotes treaties verbatim", "Collects antique datachips"],
      "motivation": "Rebuild trust between the splintered factions."
    }
  ]
}
```

### AI providers

By default the backend activates `MockAIProvider` for deterministic local play.

To connect the simulation to the full legacy stack (Model Architect + Narrative Weaver + Action Engine) set:

```bash
export COGNIVERSE_AI_PROVIDER=legacy
```

This mode reuses the existing subsystems and falls back to `mock` if initialisation fails.
### Backend provider modes

- `mock` (default): offline heuristics without legacy systems.
- `legacy`: uses Model Architect routing, Narrative Weaver analysis, and Action Engine heuristics.

Switch with `export COGNIVERSE_AI_PROVIDER=legacy` (or leave unset for `mock`).
