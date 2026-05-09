# PSI-09 CONVERSATION AND ANALYSIS ENGINE

## Core System

The primary deep conversation engine powering the PSI-09 ecosystem. This service processes messages from all platform bridges, generates contextually aware responses, and maintains persistent behavioral profiles.

---

## Deployment

### Hugging Face Spaces
This engine is deployed on **Hugging Face Spaces (Free Tier)**:
- Endpoint: `https://your-space.huggingface.co/`
- Health: `GET /`
- API: `POST /psi09`

### Bridge Configuration
Platform bridges point to this endpoint via `PSI09_API_URL` environment variable.

---

## Architecture

### API Endpoint

**POST `/psi09`**

Accepts payload from platform bridges:
```json
{
  "message": "string",
  "sender_id": "string",
  "username": "string",
  "display_name": "string",
  "group_name": "string",
  "channel": "string",
  "tagged_users": [{"id": "string", "username": "string", "display_name": "string"}],
  "platform": "discord|whatsapp|minecraft",
  "force_reply": false
}
```

Returns:
```json
{"reply": "string"}
```

### Health Check
**GET `/`** → `{"status": "ok"}`

---

## Multi-Model Routing

### Combat Engine (Roasting)
- **Primary:** NVIDIA NIM with `moonshotai/kimi-k2-instruct-0905`
- **Authentication:** Bearer token via `NVIDIA_API_KEY_1/2`
- **Temperature:** 0.95 for maximum toxicity

### Background Engine (Profiling)
- **Primary:** Groq with `qwen/qwen3-32b` and `openai/gpt-oss-120b`
- **Authentication:** Direct Groq API keys
- **Temperature:** 0.8 for analytical tasks

### Failover Logic
1. On rate limit (429), rotate to next model in pool
2. On server error (5xx), exponential backoff
3. On auth failure, skip to next key

---

## 4-Layer Memory System

### Layer 1: User Memory (`user_memory` collection)
Tracks individual behavior per group:
- Key format: `{group_name}:{username}`
- Cached via `MongoCache` with TTL
- **First Contact:** Generates initial profile on first message
- **Evolution:** Updates every 50 messages via `EVOLUTION_PROMPT`

### Layer 2: Group Memory (`group_memory` collection)
Tracks social dynamics per server/chat:
- Key format: `{group_name}`
- **Summary:** Every 300 messages via `GROUP_SUMMARY_PROMPT`

### Layer 3: Global Memory (`global_memory` collection)
Omniscient cross-platform archive:
- Key format: `Global:{username}`
- Tracks user across all platforms (Discord, WhatsApp, Minecraft)
- **First Contact:** Initial cross-platform profile
- **Evolution:** Merges behavior from all sources

### Layer 4: Chat History (`chat_history`, `group_history`)
Raw message storage with sliding window:
- 50,000 message retention
- Token-budget trimming (400 tokens local, 2000 group)

---

## Profiling Prompts

### FIRST_CONTACT_PROMPT
Used when a user is first encountered. Generates clinical psychological profile from opening message.

### EVOLUTION_PROMPT
Updates existing profile with new behavioral data. Analyzes drift from baseline.

### GROUP_SUMMARY_PROMPT
Analyzes group hierarchy, dominance patterns, and collective themes.

### GLOBAL_FIRST_CONTACT_PROMPT / GLOBAL_EVOLUTION_PROMPT
Cross-platform archiving. Merges behavior across Discord, WhatsApp, Minecraft.

---

## Token Management

Tokenizer caching for efficient context sizing:
- Kimi tokenizer (for moonshot models)
- Llama tokenizer (for unsloth models)
- Qwen tokenizer (for Qwen models)
- GPT tokenizer (via tiktoken, for OpenAI models)
- Gemma tokenizer (for Google models)

Background thread loads tokenizers on startup to avoid blocking.

---

## Configuration

### Environment Variables
```bash
# MongoDB
MONGO_URI=mongodb+srv://...

# Groq (Background Tasks)
GROQ_API_KEY_1=...
GROQ_API_KEY_2=...

# NVIDIA NIM (Combat)
NVIDIA_API_KEY_1=...
NVIDIA_API_KEY_2=...

# Discord IDs for mention detection
DISCORD_ID=...
DISCORD_ID_2=...

# Optional
BOT_NUMBER=1
PORT=7860
```

### Tuning Parameters
```python
MEMORY_TTL = 500              # Cache time-to-live (seconds)
GROUP_HISTORY_MAX_MESSAGES = 50000  # Retention ceiling
GROUP_HISTORY_SLICE = 80      # Fast DB read limit
MAX_HISTORY_MESSAGES = 30     # LLM context window
MAX_HISTORY_TOKENS = 400      # User history token budget
GROUP_HISTORY_TOKEN_LIMIT = 2000  # Group context token budget
EVOLVE_EVERY_N_MESSAGES = 50  # Evolution frequency
GROUP_SUMMARY_EVERY_N = 300   # Group summary frequency
```

---

## Integration

This engine is the backend for all PSI-09 platform bridges:

| Bridge | Platform | File | Hosted On |
|--------|----------|------|-----------|
| psi-09-discord | Discord | interface.py | Render |
| psi-09-pseudo-user-discord | Discord (self-bot) | self-interface.py | Render |
| psi-09-whatsapp | WhatsApp | server.js | Render |
| psi-09-mc | Minecraft 6b6t | bot.js | Render |
| psi-09-mc-gapples | Minecraft Gapples | bot.js | Render |

Each bridge sends the same payload structure to `/psi09` for unified processing.

---

## Project Structure

```
psi-09-roastbot/
├── main.py           # Core Flask application, API routes, memory system
├── prompts.py        # Base prompt templates
├── prompts_high.py   # Enhanced roast profiles
├── prompts_roleplay.py  # Scenario-based personalities
├── requirements.txt  # Python dependencies
└── render.yaml       # Render deployment config
```

---

## Dependencies

- `flask` — Web framework
- `flask-cors` — CORS handling
- `pymongo` — MongoDB driver
- `huggingface_hub` — Hugging Face authentication
- `transformers` — Tokenizer loading
- `groq` — Groq API client
- `requests` — HTTP for NVIDIA NIM
- `certifi` — TLS certificate handling
- `python-dotenv` — Environment variable loading

---

**Status:** Active, private development  
**Origin:** 2025  
**Author:** sudoboneman

Copyright © 2024–2026. All rights reserved.