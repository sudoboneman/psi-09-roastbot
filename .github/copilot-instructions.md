# Copilot Instructions for PSI-09 Roastbot

## Project Overview
PSI-09 Roastbot is a Flask-based AI insult generator designed for WhatsApp-style group and personal chat interactions. It leverages OpenAI's GPT models to deliver creative, psychologically sharp roasts, storing chat history and user memory in MongoDB for context-aware responses.

## Architecture
- **Flask API** (`main.py`): Exposes endpoints for roast generation (`/psi09`), health checks (`/health`), and status.
- **MongoDB**: Stores chat history (`chat_history` collection) and user memory summaries (`user_memory` collection).
- **OpenAI GPT**: Used for generating roasts and summarizing chat history/user profiles.
- **BufferedWriter**: Batches and asynchronously writes chat history to MongoDB for performance.
- **MemoryCache**: Caches user memory summaries with TTL to reduce DB reads.

## Key Patterns & Conventions
- **Roast Generation**: System prompts (`ROAST_PROMPT`, `STATUS_PROMPT`) enforce a unique, cold, and creative insult style. See `get_roast_response()` for orchestration.
- **Group vs Personal Logic**: Group roasts are triggered by bot mentions or keywords; personal roasts default to hard, short insults.
- **Chat History Trimming**: Uses token counting and summarization to fit within model limits. Summaries are generated with a sarcastic, hostile style.
- **Async Writes**: All chat history writes are buffered and flushed in batches for efficiency.
- **Error Handling**: Centralized Flask error handler logs and returns structured errors.
- **Config via `.env`**: Sensitive keys and model settings are loaded from environment variables.

## Developer Workflows
- **Run Locally**: `python main.py` (Flask server on port 5000 by default)
- **Environment**: Requires `.env` file with `MONGO_URI` and `OPENAI_API_KEY`.
- **Health Check**: GET `/health` endpoint verifies DB and cache status.
- **Debug Logging**: All major actions and errors are logged to stdout.

## Integration Points
- **OpenAI API**: Change model or API key via `.env` and `Config` dataclass.
- **MongoDB**: Connection settings in `.env` and `Config`.
- **CORS**: Enabled for all routes for easy frontend integration.

## Example Files
- `main.py`: All core logic, including API, DB, AI, and buffering.
- `.env`: Required for secrets/config (not committed).

## Special Notes
- **Roast Style**: All AI output must be short, creative, and psychologically sharp. Avoid generic insults.
- **Buffered Writes**: Do not write directly to MongoDB; use `BufferedWriter` for chat history.
- **Memory Summaries**: Use `MemoryCache` for user profiles; update only via summarization logic.

---
For questions or unclear conventions, review `main.py` for implementation details or ask for clarification.
