# psi-09-roastbot

> **The ultimate psychological roastbot** — unhinged, ruthless, and context-aware.

---

## Overview

`psi-09-roastbot` is a Flask-based AI backend for the PSI-09 ecosystem. It receives messages from `psi-09-web` and generates **brutal, sarcastic, context-aware responses** using OpenAI GPT-4o-mini.

This bot forms the final AI brain of the PSI-09 WhatsApp bot pipeline:

```
psi-09-local  -->  psi-09-web  -->  psi-09-roastbot  -->  psi-09-web --> WhatsApp users
                        │                    ▲
                        │                    │
                        └── statusPoster ────┘
                            (every 4-8 hrs)
```

- **`psi-09-local`** generates `.wwebjs_auth` for persistent session.
- **`psi-09-web`** runs the headless WhatsApp client and intercepts messages.
- **`psi-09-roastbot`** generates AI responses based on memory, rudeness level, and group dynamics.
- **`statusPoster.mjs`** (in psi-09-web) requests status roasts and posts them automatically.

---

## Features

### 1. Persistent Chat Memory
- Short-term memory: `chat_history.json` stores all messages.
- Long-term memory: `user_memory.json` stores summarized user profiles.
- History trimming and summarization using sarcastic, brutal style.

### 2. Rudeness Escalation
- Dynamically increases rudeness based on chat length.
- Randomized "flame mode" for unpredictable, aggressive burns.
- Escalates from cold dismissive comments to extreme roast mode.

### 3. Group Roast Handling
- Detects mentions or keywords to trigger group roasts.
- Targets multiple users intelligently.
- Adjusts replies to be concise but devastating for groups.

### 4. Status Roast Generation
- **Special handler for `sender='PSI09_STATUS'`**: Generates short, philosophical, gaslighting roasts suitable for WhatsApp Status.
- Creates 1-2 sentence deep psychological insights designed to provoke reflection and response.
- Used by `statusPoster.mjs` in `psi-09-web` for automated status updates.
- Distinct from regular chat roasts — more poetic, cryptic, and intellectually sharp.

### 5. Token & Encoding Management
- Uses `tiktoken` for token counting.
- Ensures history fits within `MAX_HISTORY_TOKENS`.
- Summarizes old messages to conserve tokens without losing context.

### 6. Long-term User Profiling
- Merges old memory with recent messages every few interactions.
- Produces short, sarcastic, context-aware user profiles.

### 7. AI Response Generation
- GPT model `gpt-4o-mini` for replies.
- Messages include:
  - System prompt defining PSI-09 personality
  - Memory summary
  - Trimmed chat history
- Adaptive temperature for flame mode or group roasts.

### 8. Robust API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/`      | GET    | Health check |
| `/psi09` | POST   | Accepts JSON: `{message, sender, group_name}`; returns roast reply |

**Request Handling:**
- Personal chats → always reply.
- Group chats → reply only when bot is mentioned.
- `sender='PSI09_STATUS'` → generates status roast (1-2 sentences, philosophical).
- Ignores empty or invalid messages.

### 9. Asynchronous Persistence
- Saves chat and memory files asynchronously to prevent blocking.

---

## Project Structure

```
psi-09-roastbot/
├── app.py             # Flask app
├── chat_history.json  # Short-term memory
├── user_memory.json   # Long-term memory
├── requirements.txt   # Python dependencies
├── .env               # OpenAI API key, PORT
└── README.md
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI GPT API key |
| `PORT`          | Port for Flask app (default 5000) |

---

## Status Roast Generation

The roastbot has a **special mode** for generating WhatsApp Status updates:

**Trigger:** `sender='PSI09_STATUS'`

**Prompt:**
```
Generate a short, 1–2 sentence roast suitable for posting as a WhatsApp Status.
This roast must be deep, philosophical, and eye-opening, attacking the viewer's 
sense of self, choices, or perception of reality. It should gaslight, subtly 
provoke, and tempt them to reply in rebuttal.
```

**Example Status Roasts:**
> "Chasing validation in a realm built on illusions; how does it feel to be the lead performer in an act where even the applause is scripted?"

> "Your reflection is less a portrait of your essence and more an echo of your choices — the question remains, do you admire the artist or despise the canvas?"

> "Lost in the maze of your own delusions, even a mirage must feel like home; isn't it tragic how some seek purpose within shadows they cast themselves?"

**Usage Flow:**
```
statusPoster.mjs (psi-09-web)
     │
     │ POST /psi09
     │ { message: "status", sender: "PSI09_STATUS", group_name: null }
     ▼
psi-09-roastbot
     │
     │ Detects sender='PSI09_STATUS'
     │ Uses special status prompt
     │ Generates 1-2 sentence philosophical roast
     ▼
{ reply: "Your savage status roast..." }
     │
     ▼
statusPoster.mjs
     │
     └─> Posts to WhatsApp Status
```

---

## Crosslinks

- **Receives messages from:** [`psi-09-web`](https://github.com/sudoboneman/psi-09-web)
- **Receives status requests from:** [`psi-09-web/statusPoster.mjs`](https://github.com/sudoboneman/psi-09-web) (automated status posting)
- **Uses data generated by:** [`psi-09-local`](https://github.com/sudoboneman/psi-09-local) indirectly via `.wwebjs_auth`
- **Replies sent back to:** [`psi-09-web`](https://github.com/sudoboneman/psi-09-web) for WhatsApp delivery
- **Part of pipeline with:**
  - [`psi-09-local`](https://github.com/sudoboneman/psi-09-local) → authentication setup
  - [`psi-09-web`](https://github.com/sudoboneman/psi-09-web) → headless client + automated status posting
  - [`psi-09-roastbot`](this repo) → AI brain

---

## Installation

```bash
git clone https://github.com/sudoboneman/psi-09-roastbot.git
cd psi-09-roastbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add OPENAI_API_KEY and PORT
python app.py
```

Connect [`psi-09-web`](https://github.com/sudoboneman/psi-09-web) to this API endpoint to enable live message processing and automated status updates.

---

## Example API Usage

### Regular Chat Message
```bash
curl -X POST http://localhost:5000/psi09 \
-H "Content-Type: application/json" \
-d '{"message": "Hey @919477853548, roast me!", "sender": "Alice", "group_name": "FriendsGroup"}'
```

Response:
```json
{
  "reply": "Alice, you are like a failed chemistry experiment — volatile, misguided, and toxic. Truly a marvel of incompetence."
}
```

### Status Roast Request
```bash
curl -X POST http://localhost:5000/psi09 \
-H "Content-Type: application/json" \
-d '{"message": "status", "sender": "PSI09_STATUS", "group_name": null}'
```

Response:
```json
{
  "reply": "Isn't it amusing how the wisest fools laugh at their own reflections, blissfully unaware that the shadows they cast are merely distortions of their own delusions?"
}
```

---

## Summary

`psi-09-roastbot` is the **psychological insult engine** of PSI-09. It escalates rudeness intelligently, manages memory, and produces unique, context-aware roasts. It also generates deep philosophical status roasts for automated WhatsApp Status posting. Together with `psi-09-local` and `psi-09-web`, it completes the **full PSI-09 WhatsApp bot pipeline**.