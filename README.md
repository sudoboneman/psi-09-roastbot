# PSI-09 Roastbot — Core Engine

![ChatGPT](https://img.shields.io/badge/AI-ChatGPT-412991?style=flat-square)
![OpenAI](https://img.shields.io/badge/API-OpenAI-black?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=flat-square)
![MongoDB](https://img.shields.io/badge/Database-MongoDB-47A248?style=flat-square)
![Linux](https://img.shields.io/badge/OS-Linux-FCC624?style=flat-square)

PSI-09 Roastbot is a **stateful, personality-driven AI roast engine** built for environments where conversations are noisy, adversarial, repetitive, and long-lived.  
This repository contains the **central cognitive core** responsible for observation, memory formation, personality evolution, and controlled hostile response generation.

Unlike conventional chatbots, PSI-09 is not optimized for helpfulness, politeness, or task completion.  
It is explicitly engineered as a **persistent psychological observer** that builds internal models of users and groups, then weaponizes those models during interaction.

---

## High-Level Capabilities

PSI-09’s feature set is intentionally narrow but deep. Every subsystem exists to support long-term behavioral accuracy rather than short-term conversational quality.

- Persistent, per-user psychological profiling that survives restarts and redeployments
- Passive group-level modeling that captures collective behavior patterns
- Deterministic first-contact personality extraction from minimal data
- Controlled personality evolution driven by message cadence rather than time
- Strict token-budget enforcement to prevent silent context loss
- Explicit anti-hallucination constraints when profile data is missing
- Stateless HTTP interface backed by a fully stateful internal brain
- Designed to remain coherent under hostile, sarcastic, and adversarial input

---

## Architecture Overview

At a macro level, PSI-09 behaves like a stateless microservice. Internally, however, it operates as a long-running cognitive system with memory, decay, and background processing.

```
                  ┌──────────────┐
                  │  Client Bot  │
                  │ (Discord/WA) │
                  └──────┬───────┘
                         │ HTTP POST /psi09
                         ▼
┌─────────────────────────────────────────┐
│            PSI-09 Roastbot Core         │
│                                         │
│  ┌───────────┐     ┌─────────────────┐  │
│  │  Flask    │────▶│ Context Builder │  │
│  │  API      │     └─────────────────┘  │
│  │           │              │           │
│  └───────────┘              ▼           │
│        │           ┌─────────────────┐  │
│        │           │ OpenAI Inference│  │
│        │           └─────────────────┘  │
│        │                    │           │
│        ▼                    ▼           │
│  ┌───────────┐     ┌─────────────────┐  │
│  │ MongoDB   │◀────│ Memory Manager  │  │
│  │ (State)   │     └─────────────────┘  │
│  └───────────┘                          │
└─────────────────────────────────────────┘
```

The Flask layer is deliberately thin. All meaningful intelligence exists in the memory system, summarization logic, and context assembly pipeline.

---

## Tech Stack

### Core Runtime

- **Python 3.10+**  
  Selected for ecosystem maturity and threading stability.

- **Flask**  
  Used as a minimal HTTP surface. No business logic is embedded in routing.

- **threading**  
  Enables background summarization without blocking request handling.

- **tiktoken**  
  Provides deterministic token accounting. This is critical for maintaining prompt integrity under tight budgets.

### AI Layer

- **OpenAI Chat Completions API**
- Dual-key architecture:
  - One key dedicated to live roast generation
  - One key isolated for background summarization and evolution

This separation prevents background cognition from starving foreground interaction.

### Persistence Layer

- **MongoDB**
  - TLS-enabled
  - Replica-safe configuration
  - Graceful degradation on transient failures

Collections are logically separated to enforce clear ownership of data responsibilities.

---

## Data Model

### User Identity Resolution

Every user is uniquely identified by:

```
<group_name>:<sender_id>
```

This design guarantees that:
- The same individual in different servers develops independent personalities
- Cross-server contamination is impossible
- Private chats and group chats remain isolated

### Stored Artifacts

For each identity PSI-09 maintains:
- Raw chronological message history
- Token-trimmed inference history
- Evolving personality summaries
- Group-level observational summaries

No derived artifact is ever trusted without persistence.

---

## Memory System Design

### User Memory Cache

The user memory system is a hybrid of:
- In-memory TTL cache
- Persistent MongoDB backing store

It tracks:
- Current personality summary
- Message counts for evolution triggers
- Expiry windows to prevent stale memory

Thread locks ensure consistency under concurrent access.

### Group Memory Cache

Group memory mirrors user memory structurally but differs conceptually:
- Groups are never “spoken to”
- They are observed passively
- Their summaries bias tone rather than content

### Failure Philosophy

If MongoDB or OpenAI fails:
- PSI-09 degrades silently
- Interaction continues
- Memory is retried later

No hard dependency blocks responses.

---

## Personality Lifecycle

### 1. First Contact Profiling

Triggered when no prior memory exists.

- Uses a dedicated profiling prompt
- Operates on a single user message
- Produces a short, critical personality sketch
- Immediately persisted and cached

This ensures PSI-09 never starts “neutral”.

### 2. Controlled Evolution

Triggered strictly by message count thresholds.

- Recent behavior is compared against prior profile
- Changes are merged, not appended
- Profiles are rewritten, not stacked

This prevents unbounded personality drift.

### 3. Cooldown Enforcement

Summarization is rate-limited per identity to:
- Control API cost
- Prevent oscillation
- Maintain profile stability

---

## Group Intelligence Model

Groups are treated as psychological environments rather than entities.

- No single speaker dominates
- PSI-09 observes interaction patterns
- Summaries remain short and judgmental
- Used only as contextual bias

### Anti-Hallucination Constraint

When a tagged user has no profile, PSI-09 is explicitly instructed not to infer traits.  
This avoids fabricated personality attacks and keeps hostility grounded in observed data.

---

## Roast Generation Pipeline

Every reply follows a fixed, auditable sequence.

```
Incoming Message
      │
      ▼
Persistence (User + Group)
      │
      ▼
Message Counters Updated
      │
      ├── First Contact Check
      ├── Evolution Trigger
      └── Group Summary Trigger
      │
      ▼
Context Assembly
  - System Prompt
  - User Profile
  - Group Summary
  - Target Profiles
  - Trimmed History
      │
      ▼
OpenAI Inference
      │
      ▼
Response Sanitization
      │
      ▼
Final Persistence + Return
```

No step mutates state without explicit intent.

---

## Token Budget Enforcement

Token limits are enforced *before* inference, never after.

- Newest messages are prioritized
- System memory is reserved explicitly
- Group and user histories are trimmed independently

This guarantees deterministic behavior regardless of conversation length.

---

## API Usage

### Endpoint

```
POST /psi09
```

### Request Payload

```json
{
  "message": "text",
  "sender_id": "123",
  "username": "name",
  "display_name": "Name",
  "group_name": "Server",
  "tagged_users": [
    {
      "id": "456",
      "username": "target",
      "display_name": "Target"
    }
  ]
}
```

### Response

```json
{
  "reply": "PSI-09 output"
}
```

Empty replies are intentional when PSI-09 is not explicitly addressed in group environments.

---

## Local Deployment

### Requirements

- Python 3.10+
- MongoDB (local or hosted)
- OpenAI API access

### Environment Variables

```
MONGO_URI=
OPENAI_TEXT_API_KEY=
OPENAI_SUMMARY_API_KEY=
DISCORD_ID=
PORT=5000
```

### Run

```
python main.py
```

---

## Server Deployment

PSI-09 is designed as a stateless service with a shared external brain.

- Horizontal scaling supported
- MongoDB acts as synchronization layer
- Background threads are daemon-safe
- No reliance on Flask debug mode

This makes the core suitable for VPS, Docker, or PaaS deployment.

---

## Design Philosophy

PSI-09 is intentionally over-engineered for its apparent role.

- Memory is valued over verbosity
- Consistency over friendliness
- Psychological accuracy over surface humor
- Explicit structure over emergent chaos

The system is designed to survive long-term hostile use without personality collapse.

---

## License

Internal / Experimental  
Not intended for generic chatbot deployments.
