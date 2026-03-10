# Jarvis — Multi-Agent GAIA Benchmark Solver

A multi-agent LLM system built with [LangGraph](https://github.com/langchain-ai/langgraph) for solving the [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA). A manager agent analyses each question and delegates it to one of three specialised worker agents, each powered by a different model suited to its task type.

---

## Architecture

```
                        ┌─────────────────────┐
                        │     User / Runner   │
                        └──────────┬──────────┘
                                   │ question + optional file
                                   ▼
                        ┌─────────────────────┐
                        │    prepare_input    │  detect file type,
                        │                     │  base64-encode media
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │      manager        │  DeepSeek-R1 8B (Ollama)
                        │   (router/brain)    │  structured JSON output
                        └──────────┬──────────┘
                                   │ routes to one of:
                  ┌────────────────┼─────────────────┐
                  ▼                ▼                  ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │  text_agent  │  │ multimodal_  │  │  web_agent   │
        │              │  │    agent     │  │              │
        │ Llama3.1 8B  │  │ Gemini-2.0-  │  │ Gemini-2.0-  │
        │  (Ollama)    │  │   Flash      │  │   Flash      │
        └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
               │                 │                  │
               └─────────────────┼──────────────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │   extract_answer    │  Gemini-2.0-Flash
                      │                     │  short exact answer
                      └─────────────────────┘
```

### Nodes

| Node | Model | Role |
|------|-------|------|
| `prepare_input` | — | Detects attached file type, base64-encodes images/audio for multimodal input |
| `manager` | DeepSeek-R1 8B (Ollama) | Analyses the question and routes it to the right specialist |
| `text_agent` | Llama3.1 8B (Ollama) | Handles document analysis, math, coding, and logical reasoning |
| `multimodal_agent` | Gemini-2.0-Flash (API) | Handles image, audio, and video understanding |
| `web_agent` | Gemini-2.0-Flash (API) | Handles web search and real-time information retrieval |
| `extract_answer` | Gemini-2.0-Flash (API) | Distils verbose agent output into a short exact answer |

### State

All nodes share a `GaiaState` object (extends LangGraph's `MessagesState`) that carries:

- `messages` — full conversation history (accumulated by all agents)
- `task_id`, `question`, `file_path`, `file_name`, `file_type` — task metadata
- `routed_to` — which agent was selected (`"text"` | `"multimodal"` | `"web"`)
- `agent_output` — raw text from the worker agent
- `final_answer` — the extracted concise answer

---

## Task Routing

The manager uses **two-stage routing** to decide which agent handles each question:

### Stage 1 — Rule-based pre-filters (no LLM cost)

These fire immediately for unambiguous cases:

| Condition | Routes to |
|-----------|-----------|
| Attached file is an image (`.jpg`, `.png`, `.gif`, `.webp`, …) | `multimodal` |
| Attached file is audio (`.mp3`, `.wav`, `.ogg`, …) or video (`.mp4`, `.mov`, …) | `multimodal` |
| Question contains: `website`, `url`, `http`, `current`, `latest`, `today`, `news`, `search online` | `web` |

### Stage 2 — LLM routing via DeepSeek-R1 8B

For cases the rules don't cover, the manager invokes DeepSeek-R1 8B with a structured JSON output prompt. The model returns:

```json
{"reasoning": "The question asks about a PDF document", "agent": "text"}
```

DeepSeek-R1's tool-calling is unreliable at 8B scale, so structured JSON output (`ChatOllama(format="json")`) is used instead of native tool-calling. If JSON parsing fails, a regex fallback extracts the agent name from the raw response. Ultimate fallback is `"text"`.

### Agent tools

| Agent | Tools |
|-------|-------|
| `text_agent` | `load_pdf`, `load_excel`, `load_csv`, `read_text_file`, `python_repl` |
| `multimodal_agent` | `load_pdf`, `load_excel`, `python_repl` + native image/audio input |
| `web_agent` | `tavily_search`, `fetch_page`, `python_repl` |

Images and audio files are embedded directly into the `HumanMessage` content as base64 data URIs — Gemini handles them natively without a separate tool call.

---

## Project Structure

```
jarvis/
├── config/
│   └── settings.py              # Pydantic settings (env vars, model names)
├── src/jarvis/
│   ├── cli.py                   # Typer CLI: ask, evaluate, submit
│   ├── graph/
│   │   ├── state.py             # GaiaState definition
│   │   ├── orchestrator.py      # LangGraph StateGraph (nodes + edges)
│   │   └── router.py            # DeepSeek-R1 routing logic
│   ├── agents/
│   │   ├── text.py              # Llama3.1 ReAct agent
│   │   ├── multimodal.py        # Gemini multimodal ReAct agent
│   │   └── web.py               # Gemini web browsing ReAct agent
│   ├── tools/
│   │   ├── file_loaders.py      # PDF, Excel, CSV, text reading (@tool functions)
│   │   ├── search.py            # Tavily search wrapper
│   │   ├── web_fetch.py         # Page fetcher (requests + BeautifulSoup)
│   │   ├── code_exec.py         # Python REPL
│   │   └── multimodal.py        # Base64 encoding for images/audio
│   ├── data/
│   │   ├── loader.py            # HuggingFace dataset download + split loading
│   │   └── schemas.py           # GaiaTask Pydantic model
│   ├── evaluation/
│   │   ├── scorer.py            # Quasi-exact match scoring + metrics
│   │   ├── runner.py            # Validation set evaluation loop
│   │   └── submission.py        # Test set JSONL generation
│   └── utils/
│       ├── answer_extract.py    # Concise answer extraction from agent output
│       └── normalize.py         # Answer normalisation helpers
├── tests/
│   ├── test_scorer.py
│   ├── test_router.py
│   └── test_tools.py
├── data/                        # GAIA dataset cache (gitignored)
├── outputs/                     # Submission files and eval results (gitignored)
├── pyproject.toml
└── .env.example
```

---

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- A Google AI API key (for Gemini-2.0-Flash)
- A Tavily API key (for web search)
- A HuggingFace token (the GAIA dataset is gated)

### 1. Pull the local models

```bash
ollama pull deepseek-r1:8b
ollama pull llama3.1:8b
```

### 2. Clone and install

```bash
git clone <repo-url>
cd jarvis

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### 3. Configure API keys

```bash
mkdir .env
```

Edit `.env`:

```env
JARVIS_GOOGLE_API_KEY=your-google-api-key
JARVIS_TAVILY_API_KEY=your-tavily-api-key
JARVIS_HUGGINGFACE_TOKEN=your-hf-token
```

Get your keys here:
- Google AI key: https://aistudio.google.com/apikey
- Tavily key: https://app.tavily.com
- HuggingFace token: https://huggingface.co/settings/tokens (needs read access to gated repos — request GAIA access at https://huggingface.co/datasets/gaia-benchmark/GAIA)

### 4. Verify setup

```bash
# Check Ollama models are available
ollama list

# Run the test suite (no API keys needed)
pytest tests/ -v
```

---

## Usage

All commands are available via the `jarvis` CLI after activating the virtual environment.

### Ask a single question

```bash
jarvis ask "What is the tallest mountain in Africa?"

# With an attached file
jarvis ask "How many rows are in this spreadsheet?" --file data/myfile.xlsx

# Show the full agent reasoning (not just the final answer)
jarvis ask "What year was the Eiffel Tower built?" --verbose
```

### Run local evaluation (validation set)

The GAIA validation set contains 165 questions with ground truth answers. The dataset is downloaded automatically on first run.

```bash
# Evaluate all 165 questions
jarvis evaluate

# Evaluate only Level 1 questions (easiest)
jarvis evaluate --level 1

# Quick smoke test — first 5 questions only
jarvis evaluate --level 1 --max-tasks 5

# Save results to a custom path
jarvis evaluate --output outputs/my_run.json
```

Results are printed to the console as a per-level accuracy table and saved as JSON.

### Generate a test set submission

```bash
# Run on all test questions and write submission.jsonl
jarvis submit

# Level-filtered partial submission
jarvis submit --level 1 --output outputs/submission_level1.jsonl
```

The output JSONL format matches the GAIA leaderboard requirements:

```jsonl
{"task_id": "abc123", "model_answer": "42"}
{"task_id": "def456", "model_answer": "Paris"}
```

Upload the file at: https://huggingface.co/spaces/gaia-benchmark/leaderboard

---

## Configuration

All settings can be overridden via environment variables (prefix: `JARVIS_`) or by editing `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `JARVIS_GOOGLE_API_KEY` | — | Google AI API key for Gemini |
| `JARVIS_TAVILY_API_KEY` | — | Tavily search API key |
| `JARVIS_HUGGINGFACE_TOKEN` | — | HuggingFace token for dataset access |
| `JARVIS_MANAGER_MODEL` | `deepseek-r1:8b` | Ollama model used for routing |
| `JARVIS_TEXT_MODEL` | `llama3.1:8b` | Ollama model for text agent |
| `JARVIS_MULTIMODAL_MODEL` | `gemini-2.0-flash` | Gemini model for multimodal agent |
| `JARVIS_WEB_MODEL` | `gemini-2.0-flash` | Gemini model for web agent |
| `JARVIS_EXTRACTOR_MODEL` | `gemini-2.0-flash` | Gemini model for answer extraction |
| `JARVIS_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `JARVIS_MAX_AGENT_ITERATIONS` | `15` | Max ReAct steps per agent |
| `JARVIS_DATA_CACHE_DIR` | `data/gaia` | Local path for GAIA dataset cache |
| `JARVIS_OUTPUT_DIR` | `outputs` | Directory for results and submissions |

---

## GAIA Benchmark

[GAIA](https://arxiv.org/abs/2311.12983) (General AI Assistants) is a benchmark of 450+ questions that are conceptually simple for humans but challenging for AI — requiring real tool use, multi-step reasoning, and handling of attached files (PDFs, images, audio, spreadsheets).

Questions are divided into three difficulty levels:

| Level | Description | Typical steps |
|-------|-------------|---------------|
| 1 | Solvable with minimal tool use | < 5 |
| 2 | Requires coordinating multiple tools | 5–10 |
| 3 | Long-horizon planning across many tools | 10+ |

Answers are evaluated by **quasi-exact match**: both the predicted and ground truth answers are normalised (lowercased, articles removed, numbers standardised) before comparison.
