PYTHON := python3
UV := uv

LLM_PROVIDER ?= lmstudio
OPENROUTER_MODEL ?= meta-llama/llama-3.1-8b-instruct

.PHONY: help sync check lint run-01 run-02 run-03 run-04 run-05 \
	run-openrouter-01 run-openrouter-05 clean-memory show-memory

help:
	@printf '%s\n' \
		'make sync                - install dependencies with uv sync' \
		'make check               - run py_compile on shared code and exercises' \
		'make lint                - run Ruff static checks' \
		'make run-01              - run Exercise 01' \
		'make run-02              - run Exercise 02' \
		'make run-03              - run Exercise 03' \
		'make run-04              - run Exercise 04' \
		'make run-05              - run Exercise 05' \
		'make run-openrouter-01   - run Exercise 01 with OpenRouter' \
		'make run-openrouter-05   - run Exercise 05 with OpenRouter' \
		'make clean-memory        - remove persistent summary memory' \
		'make show-memory         - print the saved summary memory file'

sync:
	$(UV) sync

check:
	$(PYTHON) -m py_compile \
		hello_lang_graph/__init__.py \
		hello_lang_graph/config.py \
		hello_lang_graph/memory.py \
		hello_lang_graph/tool_fallback.py \
		exercises/01_minimal_chat/chat.py \
		exercises/02_langgraph_state/stateful_chat.py \
		exercises/03_tools/tool_chat.py \
		exercises/04_memory/memory_chat.py \
		exercises/05_resident/tiny_claw.py

lint:
	$(UV) run ruff check .

run-01:
	$(UV) run python exercises/01_minimal_chat/chat.py

run-02:
	$(UV) run python exercises/02_langgraph_state/stateful_chat.py

run-03:
	$(UV) run python exercises/03_tools/tool_chat.py

run-04:
	$(UV) run python exercises/04_memory/memory_chat.py

run-05:
	$(UV) run python exercises/05_resident/tiny_claw.py

run-openrouter-01:
	LLM_PROVIDER=openrouter OPENROUTER_MODEL=$(OPENROUTER_MODEL) $(UV) run python exercises/01_minimal_chat/chat.py

run-openrouter-05:
	LLM_PROVIDER=openrouter OPENROUTER_MODEL=$(OPENROUTER_MODEL) $(UV) run python exercises/05_resident/tiny_claw.py

clean-memory:
	rm -f memory_store/summary_memory.json

show-memory:
	@if [ -f memory_store/summary_memory.json ]; then \
		cat memory_store/summary_memory.json; \
	else \
		printf '%s\n' 'memory_store/summary_memory.json not found'; \
	fi
