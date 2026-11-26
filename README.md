# LLM Council

![llmcouncil](header.jpg)

The idea of this repo is that instead of asking a question to your favorite LLM provider (e.g. OpenAI GPT 5.1, Google Gemini 3.0 Pro, Anthropic Claude Sonnet 4.5, xAI Grok 4, eg.c), you can group them into your "LLM Council". This repo is a simple, local web app that essentially looks like ChatGPT except it uses OpenRouter to send your query to multiple LLMs, it then asks them to review and rank each other's work, and finally a Chairman LLM produces the final response.

In a bit more detail, here is what happens when you submit a query:

1. **Stage 1: First opinions**. The user query is given to all LLMs individually, and the responses are collected. The individual responses are shown in a "tab view", so that the user can inspect them all one by one.
2. **Stage 2: Review**. Each individual LLM is given the responses of the other LLMs. Under the hood, the LLM identities are anonymized so that the LLM can't play favorites when judging their outputs. The LLM is asked to rank them in accuracy and insight.
3. **Stage 3: Final response**. The designated Chairman of the LLM Council takes all of the model's responses and compiles them into a single final answer that is presented to the user.

## Vibe Code Alert

This project was 99% vibe coded as a fun Saturday hack because I wanted to explore and evaluate a number of LLMs side by side in the process of [reading books together with LLMs](https://x.com/karpathy/status/1990577951671509438). It's nice and useful to see multiple responses side by side, and also the cross-opinions of all LLMs on each other's outputs. I'm not going to support it in any way, it's provided here as is for other people's inspiration and I don't intend to improve it. Code is ephemeral now and libraries are over, ask your LLM to change it in whatever way you like.

## Setup

### 1. Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for project management.

**Backend:**
```bash
uv sync
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

Get your API key at [openrouter.ai](https://openrouter.ai/). Make sure to purchase the credits you need, or sign up for automatic top up.

### 3. Configure Models (Optional)

Edit `backend/config.py` to customize the council:

```python
COUNCIL_MODELS = [
  "meta-llama/llama-3.3-70b-instruct:free",
  "google/gemma-3-27b-it:free",
  "qwen/qwen3-32b:free",
  "deepseek/deepseek-r1-0528:free",
]

CHAIRMAN_MODEL = "google/gemma-3-27b-it:free"
```

You can also define these in `council_config.json` or `config.yaml`.

## Running the Application

**Option 1: Use the start script**
```bash
./start.sh
```

**Option 2: Run manually**

Terminal 1 (Backend):
```bash
uv run python -m backend.main
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## Custom API Keys (Optional)

By default every model call goes through OpenRouter.  If you have accounts
with individual providers you can supply their keys directly — the app will
call that provider's native API and skip OpenRouter entirely for those models.

### Supported providers

| Provider  | Env variable        | SDK package            | Model prefix examples              |
|-----------|---------------------|------------------------|------------------------------------|
| OpenAI    | `OPENAI_API_KEY`    | `openai`               | `openai/gpt-4o`, `openai/gpt-5.1` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic`            | `anthropic/claude-3-5-sonnet`      |
| Google    | `GOOGLE_API_KEY`    | `google-generativeai`  | `google/gemini-1.5-pro`            |
| Groq      | `GROQ_API_KEY`      | `groq`                 | `groq/llama-3.3-70b-versatile`     |

Any provider **not** listed above (e.g. `x-ai/grok-4`, `mistralai/...`) will
always route through OpenRouter regardless of what keys are set.

### Key precedence

Provider keys are resolved in this order:

1. CLI flags
2. Config file (`council_config.json` or `config.yaml`)
3. Environment variables (`.env`)

### How to enable

1. Add the key(s) to your `.env` file (see `.env.example` for all options):

```bash
# Keep this for providers you haven't set up directly, and as a fallback
OPENROUTER_API_KEY=sk-or-v1-...

# Add whichever provider keys you have
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```

2. Install the corresponding SDK(s):

```bash
uv add openai               # for OpenAI direct calls
uv add anthropic            # for Anthropic direct calls
uv add google-generativeai  # for Google direct calls
uv add groq                 # for Groq direct calls
```

3. (Optional) Provide keys and model overrides via config file:

`council_config.json` example:

```json
{
  "providers": {
    "openai": { "api_key": "sk-..." },
    "anthropic": { "api_key": "sk-ant-..." },
    "google": { "api_key": "AIza..." },
    "groq": { "api_key": "gsk_..." },
    "openrouter": { "api_key": "sk-or-v1-..." },
    "fallback_to_openrouter": true
  },
  "openrouter": {
    "free_models_only": true
  },
  "council_models": [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "qwen/qwen3-32b:free"
  ],
  "chairman_model": "google/gemma-3-27b-it:free"
}
```

4. (Optional) Override from CLI when launching backend:

```bash
uv run python -m backend.main \
  --config ./council_config.json \
  --openai-key sk-... \
  --google-key AIza... \
  --groq-key gsk_... \
  --provider-fallback-to-openrouter true
```

   If a key is set but the SDK is not installed, the call will fail with a
   clear error message (and fall back to OpenRouter if
   `PROVIDER_FALLBACK_TO_OPENROUTER` is not disabled).

### Fallback behaviour

- If a direct provider call **succeeds** → the response is returned immediately.
- If it **fails** (bad key, quota exceeded, network error, etc.) → a warning is
  logged to the console and the request is automatically retried via OpenRouter.
- To disable the fallback and surface the error instead:

```bash
PROVIDER_FALLBACK_TO_OPENROUTER=false
```

### OpenRouter free-only mode

By default OpenRouter requests are constrained to free models only. If a
non-free model id is selected for an OpenRouter-routed call, the backend logs
a warning and swaps it to a fallback `:free` model.

To disable this policy:

```bash
OPENROUTER_FREE_MODELS_ONLY=false
```

### Routing logic

All routing lives in `backend/llm_client.py`.  The provider is inferred from
the model name prefix (`openai/…` → OpenAI, `anthropic/…` → Anthropic,
`google/…` → Google, `groq/…` → Groq).  The function
`infer_provider(model_name)` can be called
directly if you want to verify how a particular model string will be routed.

## Tech Stack

- **Backend:** FastAPI (Python 3.10+), async httpx, OpenRouter API
- **Frontend:** React + Vite, react-markdown for rendering
- **Storage:** JSON files in `data/conversations/`
- **Package Management:** uv for Python, npm for JavaScript
