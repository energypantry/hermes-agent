## Policy Bias Engine Plan

### Codebase map

- Main agent loop: `run_agent.py`
  - `AIAgent.run_conversation()` drives the conversation/tool loop.
  - `_execute_tool_calls_sequential()` and `_execute_tool_calls_concurrent()` run tool calls.
- LLM call path: `run_agent.py`
  - API payload is assembled inside `run_conversation()` before `_interruptible_api_call()`.
- System prompt construction: `run_agent.py`
  - `_build_system_prompt()` builds the cached base prompt.
  - Per-turn prompt additions are appended just before the API call.
- Memory read/write:
  - `run_agent.py` initializes `tools.memory_tool.MemoryStore`.
  - Memory is injected via `format_for_system_prompt("memory"/"user")`.
  - Writes happen through the `memory` tool path in `_invoke_tool()` / sequential tool execution.
- Tool selection / execution:
  - `model_tools.py:get_tool_definitions()` builds the tool surface.
  - `model_tools.py:handle_function_call()` dispatches registry tools.
  - `run_agent.py` already has policy-aware hooks around tool ordering and risk gating.

### Implementation approach

1. Add the required minimal bias layer files under `agent/policy_bias/`:
   - `bias_model.py`
   - `bias_store.py`
   - `bias_retriever.py`
2. Use the existing policy-bias engine as the integration point instead of introducing a second runtime path.
3. Seed 3 hardcoded initial biases on engine startup:
   - inspect before modifying code
   - avoid repeating failed tool actions
   - search before answering ambiguous fresh-info queries
4. Feed only the top 3 retrieved biases into a short `Decision Priors` block before each API call.
5. Ensure behavior changes are deterministic, not prompt-only:
   - inspection bias reorders tools and blocks mutating actions before inspection
   - retry-avoidance bias penalizes repeated failing tools
   - search-first bias boosts search tools for ambiguous fresh-answer requests
6. Add simple moment logging to `moment_log.json` for context/action/success snapshots.
7. Add focused tests that prove repo modification requests read before write/edit.
