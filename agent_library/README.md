# Agent library (implementations)

Concrete agents live here (`entity_extraction/`, …). The **runtime framework**
(base, registry, executor) lives in `../agent_runtime/agent_library/`.

Both directories share the Python package name `agent_library`:

- **Framework** → `agent_runtime/agent_library/` (base, registry, executor)
- **Agents** → `agent_library/<agent_name>/` (this folder)

Tests merge them via `entity_extraction/tests/conftest.py` (appends this folder to `agent_library.__path__`).

## Run entity extraction tests

From PowerShell:

```powershell
cd backend\agent_runtime
.\.venv\Scripts\Activate.ps1   # or backend\.venv if shared
pip install -e ".[dev]"
pytest ..\agent_library\entity_extraction\tests\ -v
```

Do **not** run `python test_agent.py` directly — use `pytest` so paths are set.

## Add a new agent

1. Create `backend/agent_library/<your_agent>/` with `__init__.py` and `agent.py`.
2. Subclass `BaseAgent` / `LLMAgent` from `from agent_library import ...`.
3. Register via `AgentLoader().load_package("agent_library.<your_agent>")` or app auto-discovery.
