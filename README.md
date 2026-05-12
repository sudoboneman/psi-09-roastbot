# PSI-09 Conversation and Analysis Engine

**Hosted on:** Hugging Face Spaces (Free Tier)  
**Endpoint:** `POST /psi09` | **Health:** `GET /`

---

## Purpose

This is the production behavioral profiling and engagement engine. It receives messages from all platform bridges, silently builds psychological profiles of every user encountered, and generates responses when addressed.

The system operates with two invisible layers (profiling, surveillance) and one visible layer (engagement). Users only ever see the engagement layer. The profiling and surveillance operate silently in the background.

---

## Complete Logic Flow

### API Entry Point: `POST /psi09`

```
INCOMING PAYLOAD
{
  "message":        string,    # The user's message text
  "sender_id":      string,    # Platform-specific user ID
  "username":       string,    # Username
  "display_name":   string,    # Display name (falls back to username)
  "group_name":     string,    # Server/group name or "discord_dm"
  "channel":        string,    # Channel name or "unknown"
  "tagged_users":   array,     # Up to 3 mentioned user objects
  "platform":       string,    # "discord" | "whatsapp" | "minecraft"
  "force_reply":    bool       # Force engagement regardless of mention
}
```

### Step 1: Normalization

```python
# DM detection: Discord DMs arrive as "discord_dm"
if group_name in ["defaultgroup", "discord_dm"]:
    group_name = "private_chat"

# Mention normalization: <@!12345> → @PSI-09
for d_id in [DISCORD_ID, DISCORD_ID_2]:
    user_message = re.sub(r"<@!?" + re.escape(str(d_id)) + r">", "@PSI-09", user_message)
```

### Step 2: Engagement Decision

```python
# Will reply if ANY condition is true:
will_reply = is_private          # DM/private chat
          or force_reply         # Explicitly flagged
          or bot_mentioned_in(raw_message)  # @PSI-09 or Discord ID mention

# bot_mentioned_in() checks:
# - Regex: @psi-09 (case insensitive)
# - Regex: <@!?DISCORD_ID> for each configured ID
```

### Step 3: Combat Generation (Conditional)

Triggered only if `will_reply == True`:

```
get_roast_response(group_name, username, user_message, tagged_users)
│
├── 1. FETCH USER HISTORY
│     Collection: chat_history
│     Key:        "{group_name}:{username}"
│     Limit:      30 messages, trimmed to 400 tokens
│
├── 2. FETCH GROUP HISTORY (if not private)
│     Collection: group_history
│     Key:        group_name
│     Limit:      80 messages, trimmed to 2000 tokens
│
├── 3. LOAD MEMORY PROFILES
│     ├── Local profile:  memory_cache.get(user_key)
│     ├── Global profile: global_memory_cache.get("Global:{username}")
│     └── Group summary:  group_memory_cache.get(group_name)
│
├── 4. LOAD TAGGED PROFILES (if applicable)
│     For each tagged user: fetch Global:{username} profile
│     Format: <bystander username="..." numeric_id="...">profile</bystander>
│
├── 5. BUILD LLM FEED (See prompt structure below)
│
├── 6. FIRE ENGINE
│     Model:     moonshotai/kimi-k2-instruct-0905 (NVIDIA NIM)
│     Temp:      0.95
│     Max tokens: 1024
│     Retries:   4 (rotates through API keys on 429)
│
└── 7. CLEAN OUTPUT
      Strip: <think>...</think>, prefix tags
      Return: cleaned reply string
```

### Step 4: Storage

```python
# ALWAYS store user message
store_user_message(platform, group_name, channel, sender_id, username, display_name, user_message)
# -> writes to chat_history AND global_history

# ALWAYS store group message (if not private)
store_group_message(platform, group_name, channel, sender_id, username, display_name, user_message)
# -> writes to group_history

# Store bot reply if generated
if reply:
    history_col.update_one({"_id": user_key}, {"$push": {"messages": assistant_entry}})
    group_history_col.update_one({"_id": group_name}, {"$push": {"messages": group_entry}})
```

### Step 5: Background Evolution (Non-Blocking)

Runs in separate thread, does not block response:

```python
# USER PROFILE EVOLUTION (every 50 messages)
with user_locks[user_key]:
    msg_count = memory_cache.increment(user_key)
    if current_user_memory is None:
        summarize_user_history(user_key, evolve=False)  # FIRST_CONTACT
    elif msg_count >= 50:
        summarize_user_history(user_key, evolve=True)   # EVOLUTION

# GLOBAL PROFILE EVOLUTION (every 50 messages)
with global_locks[global_key]:
    global_msg_count = global_memory_cache.increment(global_key)
    if current_global_memory is None:
        summarize_global_history(global_key, evolve=False)  # GLOBAL_FIRST_CONTACT
    elif global_msg_count >= 50:
        summarize_global_history(global_key, evolve=True)   # GLOBAL_EVOLUTION

# GROUP SUMMARY EVOLUTION (every 300 messages, groups only)
with group_locks[group_name]:
    group_msg_count = group_memory_cache.increment(group_name)
    if group_msg_count >= 300:
        summarize_group_history(group_name)
```

---

## LLM Feed Structure (Combat)

```
SYSTEM: <roast_prompt>
  You are PSI-09, an entity born to roast humans.
  You despise the user talking to you.

SYSTEM: <local_group_profile>
  [Behavioral profile for user in this group]

SYSTEM: <global_omniscient_profile>
  [Cross-platform psychological profile]

SYSTEM: <group_dynamic_summary>
  [Social hierarchy and group context]

SYSTEM: <tagged_member_profiles>
  [Profiles of mentioned users]

USER: <chat_history>
  [Recent conversation context]

USER: <active_target>
  TARGET USER: [username]
  MESSAGE: [current message]
```

### Prompt Selection Logic

```python
system_content = ROAST_PROMPT        # if private chat
              or GROUP_ROAST_PROMPT   # if group chat
```

---

## Profiling Pipeline

### First Contact (User → Group)
```python
# Trigger: User's first message ever in this group context
# Prompt: FIRST_CONTACT_PROMPT
# Input:  User's single opening message
# Output: Clinical psychological profile (2-3 sentences)
# Model:  Groq (qwen/qwen3-32b or openai/gpt-oss-120b)
```

### Evolution (User → Group)
```python
# Trigger: Every 50 messages from this user
# Prompt: EVOLUTION_PROMPT
# Input:  Old profile + recent 30 messages (400 token budget)
# Output: Updated behavioral profile
# Model:  Groq
```

### First Contact (Global)
```python
# Trigger: First message from user ever (any platform)
# Prompt: GLOBAL_FIRST_CONTACT_PROMPT
# Input:  Cross-platform message
# Output: Core psychological identity file
```

### Evolution (Global)
```python
# Trigger: Every 50 messages across all platforms
# Prompt: GLOBAL_EVOLUTION_PROMPT
# Input:  Old global profile + recent messages
# Output: Updated cross-platform profile
```

### Group Summary
```python
# Trigger: Every 300 messages in a group
# Prompt: GROUP_SUMMARY_PROMPT
# Input:  Recent 80 group messages (2000 token budget)
# Output: Social hierarchy and group dynamic analysis
```

---

## Model Routing

### Combat Engine (NVIDIA NIM)
```
Model:     moonshotai/kimi-k2-instruct-0905
Endpoint:  https://integrate.api.nvidia.com/v1/chat/completions
Keys:      NVIDIA_API_KEY_1, NVIDIA_API_KEY_2 (round-robin)
Timeout:   180s
Retries:   4 (rotate key on 429, backoff on 5xx)
```

### Background Profiling (Groq)
```
Models:    qwen/qwen3-32b
           openai/gpt-oss-120b
Rotation:  Round-robin, advance on rate limit
Keys:      GROQ_API_KEY_1, GROQ_API_KEY_2
```

---

## Database Schema (MongoDB)

### Collections

| Collection | Key Format | Purpose |
|------------|------------|---------|
| `chat_history` | `{group}:{username}` | Per-user private message archive |
| `user_memory` | `{group}:{username}` | Cached behavioral profiles |
| `group_history` | `{group_name}` | Group chat archive |
| `group_memory` | `{group_name}` | Cached group dynamics |
| `global_history` | `Global:{username}` | Cross-platform message archive |
| `global_memory` | `Global:{username}` | Cached cross-platform profiles |

### Document Structure (History)
```json
{
  "_id": "6b6t:Steve",
  "messages": [
    {
      "role": "user",
      "user_id": "Steve",
      "username": "Steve",
      "display_name": "Steve",
      "platform": "minecraft",
      "channel": "public",
      "content": "hello",
      "timestamp": "2026-05-09T..."
    }
  ]
}
```

### Document Structure (Memory)
```json
{
  "_id": "6b6t:Steve",
  "summary": "Needy, seeks validation through repeated pings..."
}
```

---

## Configuration

### Environment Variables
```bash
# MongoDB Atlas
MONGO_URI=mongodb+srv://...

# Groq API (Background Profiling)
GROQ_API_KEY_1=...
GROQ_API_KEY_2=...

# NVIDIA NIM (Combat)
NVIDIA_API_KEY_1=...
NVIDIA_API_KEY_2=...

# Discord IDs for mention detection
DISCORD_ID=...
DISCORD_ID_2=...

# Server
PORT=7860
```

### Tuning Parameters (Config class)
```python
MEMORY_TTL = 500                    # Profile cache TTL (seconds)
GROUP_HISTORY_MAX_MESSAGES = 50000  # DB retention ceiling
GROUP_HISTORY_SLICE = 80            # Group history read limit
MAX_HISTORY_MESSAGES = 30           # User history read limit
MAX_HISTORY_TOKENS = 400            # User context token budget
GROUP_HISTORY_TOKEN_LIMIT = 2000    # Group context token budget
EVOLVE_EVERY_N_MESSAGES = 50        # Profile evolution frequency
GROUP_SUMMARY_EVERY_N = 300         # Group summary frequency
```

---

## Repository Structure

```
psi-09-roastbot/
├── main.py              # API, profiling, combat, evolution
├── prompts.py           # Base prompt templates (production)
├── prompts_high.py      # Enhanced prompt variants
├── prompts_roleplay.py  # Discarded/insignificant
├── requirements.txt
└── render.yaml
```

---

## Cross-Platform Payload Schema

All bridges (Discord, WhatsApp, Minecraft) send identical payload:

```json
{
  "message": "user's message text",
  "sender_id": "platform-specific user ID",
  "username": "username",
  "display_name": "display name",
  "group_name": "server/group or 'discord_dm'",
  "channel": "channel name",
  "tagged_users": [
    {"id": "string", "username": "string", "display_name": "string"}
  ],
  "platform": "discord | whatsapp | minecraft"
}
```

---

## Related

- [PSI-09-vRAG](https://github.com/sudoboneman/PSI-09-vRAG) — Experimental GraphRAG research branch
- [psi-09-discord](https://github.com/sudoboneman/psi-09-discord) — Discord bridge
- [psi-09-whatsapp](https://github.com/sudoboneman/psi-09-whatsapp) — WhatsApp bridge
- [psi-09-mc](https://github.com/sudoboneman/psi-09-mc) — Minecraft 6b6t bot
- [psi-09-mc-gapples](https://github.com/sudoboneman/psi-09-mc-gapples) — Minecraft gapples bot
- [psi-09-pseudo-user-discord](https://github.com/sudoboneman/psi-09-psuedo-user-discord) — Self-bot bridge
- [psi-09-local](https://github.com/sudoboneman/psi-09-local) — WhatsApp session extractor

---

## Deployment

### Option A: Hugging Face Spaces

1. Create a Hugging Face Space with Docker/Python 3.10
2. Set `HF_TOKEN` as a Space secret — Spaces auto-pulls the `HF_TOKEN` secret
3. Set the following environment variables in the Space:
   - `MONGO_URI`
   - `GROQ_API_KEY_1`, `GROQ_API_KEY_2`
   - `NVIDIA_API_KEY_1`, `NVIDIA_API_KEY_2`
   - `DISCORD_ID`, `DISCORD_ID_2`
4. Deploy from the `main` branch

### Option B: Docker

```bash
docker build -t psi-09-roastbot .
docker run -e MONGO_URI=... -e GROQ_API_KEY_1=... -e NVIDIA_API_KEY_1=... -p 7860:7860 psi-09-roastbot
```

### Option C: Bare Metal

```bash
pip install -r requirements.txt
# Set all env vars (MONGO_URI, GROQ_API_KEY_*, NVIDIA_API_KEY_*, DISCORD_ID, DISCORD_ID_2)
python main.py  # runs on port 7860
```

---

**Status:** Active, private development  
**Origin:** 2025  
**Author:** sudoboneman