# :brain: vibemind-shared

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Multi-provider LLM client factory for the VibeMind OS ecosystem.**

A pip-installable Python package that provides a unified interface for creating LLM clients across multiple providers. Configure once, use everywhere.

## Supported Providers

| Provider | Status |
|----------|--------|
| OpenAI | :white_check_mark: |
| Anthropic | :white_check_mark: |
| OpenRouter | :white_check_mark: |
| Google Gemini | :white_check_mark: |
| Groq | :white_check_mark: |
| Ollama (local) | :white_check_mark: |

## Installation

```bash
# From PyPI
pip install vibemind-shared

# Editable / development install
git clone https://github.com/Flissel/vibemind-shared.git
cd vibemind-shared
pip install -e .
```

## Configuration

Create a `llm_config.yml` in your project root:

```yaml
default:
  provider: openai
  model: gpt-4o
  temperature: 0.7

roles:
  planner:
    provider: anthropic
    model: claude-sonnet-4-20250514
  coder:
    provider: openrouter
    model: deepseek/deepseek-coder
  local:
    provider: ollama
    model: llama3
```

Role-based overrides let each component in your system use the best model for its task.

## Usage

```python
from vibemind_shared import get_client, get_model

# Default client
client = get_client()

# Role-specific client
planner_client = get_client(role="planner")
model = get_model(role="planner")

response = planner_client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello from VibeMind"}]
)
```

## Project Structure

```
vibemind-shared/
  vibemind_shared/
    __init__.py
    client_factory.py
    config_loader.py
    providers/
  llm_config.yml
  setup.py
  pyproject.toml
```

## License

MIT -- Felix Baumann ([@Flissel](https://github.com/Flissel))
