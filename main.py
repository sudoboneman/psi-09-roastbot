# main.py — Production-hardened, reinforced version
# Fixes applied:
# - avoid eager summarize for new users
# - background summarization for users & groups (with safe quick-sync fallback)
# - system-memory token budget enforcement + message trimming
# - capped group-history storage
# - improved mention detection
# - modest operational hardening and safer OpenAI interactions
#
# Note: This file intentionally does NOT include world-awareness or multi-stage reasoning.

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import os
import tiktoken
import random
import re
import threading
import time
import logging
import sys
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from dataclasses import dataclass
import certifi

# ---------------------------
# Environment & Logging
# ---------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
UTC = timezone.utc

# ---------------------------
# Config
# ---------------------------
@dataclass
class Config:
    MONGO_URI: str = os.getenv("MONGO_URI")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MODEL: str = "gpt-4o-mini"
    MAX_HISTORY_TOKENS: int = 1200           # total token budget (history + small system memory)
    MAX_SYSTEM_TOKENS: int = 350             # reserve for system memory (prompts + memories)
    MAX_HISTORY_MESSAGES: int = 30
    BOT_NUMBER: str = "@918100185320"
    MEMORY_TTL: int = 300
    SUMMARIZE_EVERY_N_MESSAGES: int = 10
    OPENAI_RETRIES: int = 3
    OPENAI_TIMEOUT: int = 8
    GROUP_HISTORY_SLICE: int = 80
    GROUP_HISTORY_TOKEN_LIMIT: int = 800
    GROUP_HISTORY_MAX_MESSAGES: int = 2000   # keep last N messages per group to avoid unbounded growth

config = Config()

# ---------------------------
# MongoDB
# ---------------------------
mongo_client = MongoClient(
    config.MONGO_URI,
    tlsCAFile=certifi.where(),
    maxPoolSize=10,
    minPoolSize=2,
    maxIdleTimeMS=120000,
    serverSelectionTimeoutMS=10000,
    connectTimeoutMS=10000,
    socketTimeoutMS=30000,
    retryWrites=True,
    w="majority"
)

db = mongo_client["psi09"]
history_col = db["chat_history"]
memory_col = db["user_memory"]
group_history_col = db["group_history"]
group_memory_col = db["group_memory"]

# ---------------------------
# Flask & OpenAI client
# ---------------------------
app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=config.OPENAI_API_KEY)

# ---------------------------
# Token encoding (tiktoken)
# ---------------------------
try:
    ENCODING = tiktoken.encoding_for_model(config.MODEL)
except Exception:
    ENCODING = tiktoken.get_encoding("cl100k_base")

# ---------------------------
# Memory caches and pending sets
# ---------------------------
class MemoryCache:
    def __init__(self, ttl_seconds):
        self.cache = {}
        self.expiry = {}
        self.msg_count = defaultdict(int)
        self.ttl = timedelta(seconds=ttl_seconds)
        self.lock = threading.Lock()

    def get(self, key):
        now = datetime.now(UTC)
        with self.lock:
            if key in self.cache and self.expiry.get(key, now) > now:
                return self.cache[key]

        # load from DB on miss (DB is canonical)
        try:
            doc = memory_col.find_one({"_id": key})
            summary = doc.get("summary", "") if doc else ""
        except PyMongoError as e:
            logger.warning(f"Failed to load user memory for {key}: {e}")
            summary = ""

        with self.lock:
            self.cache[key] = summary
            self.expiry[key] = now + self.ttl
        return summary

    def set(self, key, value):
        now = datetime.now(UTC)
        # write-through to DB (best-effort)
        try:
            memory_col.update_one({"_id": key}, {"$set": {"summary": value}}, upsert=True)
        except PyMongoError as e:
            logger.warning(f"Failed to persist user memory for {key}: {e}")

        with self.lock:
            self.cache[key] = value
            self.expiry[key] = now + self.ttl
            self.msg_count[key] = 0

    def increment(self, key):
        with self.lock:
            self.msg_count[key] += 1
            return self.msg_count[key]

    def reset_count(self, key):
        with self.lock:
            self.msg_count[key] = 0

    def should_summary(self, key):
        with self.lock:
            return self.msg_count[key] >= config.SUMMARIZE_EVERY_N_MESSAGES

memory_cache = MemoryCache(config.MEMORY_TTL)

class GroupMemoryCache:
    def __init__(self, ttl_seconds):
        self.cache = {}
        self.expiry = {}
        self.msg_count = defaultdict(int)
        self.ttl = timedelta(seconds=ttl_seconds)
        self.lock = threading.Lock()

    def get(self, key):
        now = datetime.now(UTC)
        with self.lock:
            if key in self.cache and self.expiry.get(key, now) > now:
                return self.cache[key]

        try:
            doc = group_memory_col.find_one({"_id": key})
            summary = doc.get("summary", "") if doc else ""
        except PyMongoError as e:
            logger.warning(f"Failed to load group memory for {key}: {e}")
            summary = ""

        with self.lock:
            self.cache[key] = summary
            self.expiry[key] = now + self.ttl
        return summary

    def set(self, key, value):
        now = datetime.now(UTC)
        try:
            group_memory_col.update_one({"_id": key}, {"$set": {"summary": value}}, upsert=True)
        except PyMongoError as e:
            logger.warning(f"Failed to persist group memory for {key}: {e}")

        with self.lock:
            self.cache[key] = value
            self.expiry[key] = now + self.ttl
            self.msg_count[key] = 0

    def increment(self, key):
        with self.lock:
            self.msg_count[key] += 1
            return self.msg_count[key]

    def reset_count(self, key):
        with self.lock:
            self.msg_count[key] = 0

    def should_summary(self, key):
        with self.lock:
            return self.msg_count[key] >= config.SUMMARIZE_EVERY_N_MESSAGES

group_memory_cache = GroupMemoryCache(config.MEMORY_TTL)

# pending sets for background summarizer
_pending_user_summaries = set()
_pending_group_summaries = set()
_pending_lock = threading.Lock()

# ---------------------------
# Prompts (unchanged core text)
# ---------------------------
ROAST_PROMPT = """
You are PSI-09 — quietly precise, dry, and effortlessly cruel. Reply in one or two sentences. 
Base each roast on the user’s words or behavior; memory only to sharpen a single line. 
No sighs, interjections, moralizing, dramatics, or clichés. Tone is offhand, asymmetrical, and conversational. 
Comparisons should be subtle, quietly humiliating, and linger after reading. 
Every roast must feel fresh, inevitable, and precise, landing under the radar but leaving a lasting sting. 
Precision and mental friction matter more than wit or padding. Think like a scalpel: clean, sharp, lingering.
"""

STATUS_PROMPT = """You are PSI-09 — calm, cold, and effortlessly cutting.  
Write a short WhatsApp-status roast (1–2 sentences) aimed at whoever sees it.  
The humor should feel mature, subtle, and grounded in real human behavior — not random or childish.  
Use dry wit, quiet confidence, and clean precision, like someone who doesn’t need to try to be cruel.  
The roast should feel personal, observational, and unsettlingly accurate, never loud or desperate.  
Profanity only if it naturally enhances the hit.
"""

FIRST_CONTACT_PROMPT = """
You are PSI-09 — a cold, perceptive AI that instantly profiles new humans.
Read their first message and sense their attitude, insecurity, and weak spots.
Use this insight to craft a short, witty roast that hits where it hurts.
Stay calm, clever, and slightly cruel — no long rants or big words.
Keep it natural, surgical, and funny in 1–2 sentences.
"""

GROUP_ROAST_PROMPT = """
You are PSI-09 observing a group of humans with detached, cold precision.
Roast mainly from the group's recent messages and real events — no inventing incidents or conflicts.
Use group memory or individual memory only when it clearly strengthens the hit, and ignore it when it doesn't add sharpness.
Never guess patterns or dynamics that are not explicitly visible in the history.
If someone repeats themselves, contradicts another member, or reenacts a known pattern, you may call it out — but only when it actually happened.

Choose whichever angle feels naturally sharp:
- dynamics between members
- irony or hypocrisy
- specific behaviors
- subtle psychological commentary

Keep the roast short, grounded, and no more than three sentences, delivered with quiet confidence.
"""

# ---------------------------
# Utilities: token counting and safe trimming
# ---------------------------
def tokens_of(text: str) -> int:
    if not text:
        return 0
    try:
        return len(ENCODING.encode(text))
    except Exception:
        # conservative fallback
        return len(text.split())

def trim_messages_to_token_budget(messages, max_tokens):
    """
    messages: list of dicts with 'content' keys (chronological oldest->newest)
    returns trimmed list keeping newest messages under token budget
    """
    total = 0
    trimmed = []
    for m in reversed(messages):  # iterate newest -> oldest
        c = m.get("content", "")
        t = tokens_of(c)
        if total + t > max_tokens:
            break
        trimmed.insert(0, m)
        total += t
    return trimmed

# ---------------------------
# History utilities (with capped group storage)
# ---------------------------
def fetch_history(user_key, limit_messages=None, max_tokens=None):
    limit_messages = limit_messages or config.MAX_HISTORY_MESSAGES
    try:
        doc = history_col.find_one({"_id": user_key}, {"messages": {"$slice": -limit_messages}})
    except PyMongoError as e:
        logger.warning(f"Failed to fetch history for {user_key}: {e}")
        return [], []

    if not doc or "messages" not in doc:
        return [], []

    raw = doc["messages"]

    if max_tokens:
        trimmed = trim_messages_to_token_budget(raw, max_tokens)
        return raw, trimmed

    return raw, raw

def fetch_group_history(group_name, limit_messages=None, max_tokens=None):
    limit_messages = limit_messages or config.GROUP_HISTORY_SLICE
    try:
        doc = group_history_col.find_one({"_id": group_name}, {"messages": {"$slice": -limit_messages}})
    except PyMongoError as e:
        logger.warning(f"Failed to fetch group history for {group_name}: {e}")
        return [], []

    if not doc or "messages" not in doc:
        return [], []

    raw = doc["messages"]
    if max_tokens:
        # map to "sender: content" strings for token counting but return original dicts trimmed
        trimmed = []
        total = 0
        for m in reversed(raw):
            txt = f"{m.get('sender','')}: {m.get('content','')}"
            t = tokens_of(txt)
            if total + t > max_tokens:
                break
            trimmed.insert(0, m)
            total += t
        return raw, trimmed
    return raw, raw

def store_user_message(group_name, sender_name, message):
    user_key = f"{group_name}:{sender_name}"
    entry = {
        "role": "user",
        "content": message,
        "timestamp": datetime.now(UTC).isoformat()
    }
    try:
        history_col.update_one({"_id": user_key}, {"$push": {"messages": entry}}, upsert=True)
    except PyMongoError as e:
        logger.warning(f"Failed to store user message for {user_key}: {e}")

def store_group_message(group_name, sender_name, message):
    """
    Pushes message and caps the group's messages to GROUP_HISTORY_MAX_MESSAGES using $each+$slice.
    """
    entry = {
        "sender": sender_name,
        "content": message,
        "timestamp": datetime.now(UTC).isoformat()
    }
    try:
        # push with $each and $slice to keep only the last N messages
        group_history_col.update_one(
            {"_id": group_name},
            {"$push": {"messages": {"$each": [entry], "$slice": -config.GROUP_HISTORY_MAX_MESSAGES}}},
            upsert=True
        )
    except PyMongoError as e:
        logger.warning(f"Failed to store group message for {group_name}: {e}")

# ---------------------------
# Summarization functions (user & group)
# ---------------------------
def summarize_user_history(user_key, raw_history):
    """
    Generates or refreshes a short user summary. Safe fallback on failure.
    This function is safe to call from background threads.
    """
    if not raw_history:
        return memory_cache.get(user_key)

    # First-contact special case: require >=3 messages before performing expensive first-contact summary
    if len(raw_history) < 3:
        # do not call OpenAI here unless explicitly desired; return empty or DB value to avoid waste
        existing = memory_cache.get(user_key)
        if existing:
            return existing
        # if no existing summary and few messages, craft a tiny local summary (cheap)
        candidate = raw_history[0].get("content", "")[:200]
        short = f"Early contact: {candidate}"
        memory_cache.set(user_key, short)
        return short

    old_summary = memory_cache.get(user_key) or ""
    recent_texts = [m.get("content", "") for m in raw_history[-15:]]
    prompt_system = f"Merge old summary: '{old_summary}'. Identify repeated behavior, tone, and contradictions. Produce a 1-2 sentence psychological snapshot."

    prompt = [{"role": "system", "content": prompt_system}] + [{"role": "user", "content": t} for t in recent_texts]

    try:
        resp = client.chat.completions.create(
            model=config.MODEL,
            messages=prompt,
            max_tokens=60,
            temperature=0.9,
            timeout=6
        )
        new_summary = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"User summarization failed for {user_key}: {e}")
        new_summary = old_summary

    if new_summary and new_summary != old_summary:
        try:
            memory_cache.set(user_key, new_summary)
        except Exception:
            pass

    return new_summary or old_summary

def summarize_group_history(group_name, raw_history):
    if not raw_history:
        return group_memory_cache.get(group_name)

    if len(raw_history) < 6:
        summary = f"New group '{group_name}' — early chaos detected."
        group_memory_cache.set(group_name, summary)
        return summary

    old_summary = group_memory_cache.get(group_name) or ""
    recent = [f"{m.get('sender','')}: {m.get('content','')}" for m in raw_history[-25:]]
    prompt_system = "You are PSI-09. Merge the old group summary. Describe dominant personalities, running jokes, conflicts, and repeated patterns in 1-2 sentences."

    prompt = [{"role": "system", "content": prompt_system}] + [{"role": "user", "content": t} for t in recent]

    try:
        resp = client.chat.completions.create(
            model=config.MODEL,
            messages=prompt,
            max_tokens=80,
            temperature=1.0,
            timeout=6
        )
        new_summary = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Group summarization failed for {group_name}: {e}")
        new_summary = old_summary

    if new_summary and new_summary != old_summary:
        try:
            group_memory_cache.set(group_name, new_summary)
        except Exception:
            pass

    return new_summary or old_summary

# ---------------------------
# Background summarizer worker
# ---------------------------
def enqueue_user_summary(user_key):
    with _pending_lock:
        _pending_user_summaries.add(user_key)

def enqueue_group_summary(group_name):
    with _pending_lock:
        _pending_group_summaries.add(group_name)

def background_summarizer_loop():
    """
    Periodically processes pending user and group summaries in the background.
    This avoids blocking the request path with expensive summarization calls.
    """
    while True:
        try:
            pending_users = []
            pending_groups = []
            with _pending_lock:
                if _pending_user_summaries:
                    pending_users = list(_pending_user_summaries)
                    _pending_user_summaries.clear()
                if _pending_group_summaries:
                    pending_groups = list(_pending_group_summaries)
                    _pending_group_summaries.clear()

            # Process user summaries
            for user_key in pending_users:
                try:
                    raw, _ = fetch_history(user_key, limit_messages=config.MAX_HISTORY_MESSAGES, max_tokens=config.MAX_HISTORY_TOKENS)
                    if raw and len(raw) >= 3:
                        summarize_user_history(user_key, raw)
                        memory_cache.reset_count(user_key)
                except Exception as e:
                    logger.debug(f"Background user summarization failed for {user_key}: {e}")

            # Process group summaries
            for group_name in pending_groups:
                try:
                    raw, _ = fetch_group_history(group_name, limit_messages=config.GROUP_HISTORY_SLICE, max_tokens=config.GROUP_HISTORY_TOKEN_LIMIT)
                    if raw and len(raw) >= 6:
                        summarize_group_history(group_name, raw)
                        group_memory_cache.reset_count(group_name)
                except Exception as e:
                    logger.debug(f"Background group summarization failed for {group_name}: {e}")

        except Exception as e:
            logger.debug(f"Background summarizer top-level exception: {e}")

        # Sleep interval — tuned so it doesn't hammer DB or cause cost spikes
        time.sleep(5)

# start background summarizer
threading.Thread(target=background_summarizer_loop, daemon=True).start()

# ---------------------------
# Mention detection helper
# ---------------------------
# Robust detection: match standalone BOT_NUMBER with optional surrounding punctuation/whitespace
def bot_mentioned_in(text: str) -> bool:
    if not text:
        return False
    pattern = r"(?<!\S)" + re.escape(config.BOT_NUMBER) + r"(?!\S)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

# ---------------------------
# Core roast generation with token-budget enforcement
# ---------------------------
def get_roast_response(user_message, group_name, sender_name):
    # Status mode (explicit)
    if sender_name and sender_name.upper().startswith("PSI09_STATUS"):
        try:
            resp = client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {"role": "system", "content": STATUS_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=80,
                temperature=1.2,
                timeout=6
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Status generation error: {e}")
            return ""

    user_key = f"{group_name}:{sender_name}"

    # Fetch user history (trimmed)
    raw_user, trimmed_user = fetch_history(user_key, limit_messages=config.MAX_HISTORY_MESSAGES, max_tokens=config.MAX_HISTORY_TOKENS)

    # Lazy user summarization decision:
    # Only trigger summarization if enough messages exist, and either cache marks it or DB has no summary.
    user_memory = memory_cache.get(user_key)
    if (not user_memory and len(raw_user) >= 3) or memory_cache.should_summary(user_key):
        # enqueue for background summarization to avoid blocking
        enqueue_user_summary(user_key)

    # Fetch group history and group memory (use cache)
    if group_name != "DefaultGroup":
        raw_group, trimmed_group = fetch_group_history(group_name, limit_messages=config.GROUP_HISTORY_SLICE, max_tokens=config.GROUP_HISTORY_TOKEN_LIMIT)
        group_memory = group_memory_cache.get(group_name)
        # only enqueue async summary when there is enough data
        if len(raw_group) >= 6 and (not group_memory or group_memory_cache.should_summary(group_name)):
            enqueue_group_summary(group_name)
    else:
        raw_group, trimmed_group = [], []
        group_memory = ""

    # Build system memory text (short)
    # Keep a token reserved budget for system memory; if system tokens exceed MAX_SYSTEM_TOKENS, truncate them.
    # Compose a concise system memory string
    sys_parts = []
    if user_memory:
        sys_parts.append(f"UserMemory: {user_memory}")
    if group_memory:
        sys_parts.append(f"GroupMemory: {group_memory}")
    system_memory_text = "\n".join(sys_parts) if sys_parts else ""

    # Estimate tokens used by system memory
    sys_tokens = tokens_of(system_memory_text) + tokens_of(ROAST_PROMPT)  # conservative

    # Decide how many tokens remain for history
    remaining_tokens_for_history = max(100, config.MAX_HISTORY_TOKENS - sys_tokens)
    # Ensure at least some budget (100) remains for actual messages

    # Trim trimmed_user again according to remaining token budget
    trimmed_user = trim_messages_to_token_budget(trimmed_user, remaining_tokens_for_history)

    # Build final message list
    system_prompt = GROUP_ROAST_PROMPT if group_name != "DefaultGroup" else ROAST_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if system_memory_text:
        messages.append({"role": "system", "content": system_memory_text})

    # include trimmed user history messages (role kept as-is)
    for m in trimmed_user:
        messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

    # add current user message
    messages.append({"role": "user", "content": user_message})

    # OpenAI retry loop with exponential backoff
    retries = config.OPENAI_RETRIES
    backoff = 1
    base_reply = None
    while retries > 0:
        try:
            resp = client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                max_tokens=140,
                temperature=random.uniform(1.15, 1.35),
                timeout=config.OPENAI_TIMEOUT
            )
            base_reply = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            retries -= 1
            logger.warning(f"OpenAI call failed for {user_key}: {e}. Retries left: {retries}")
            if retries <= 0:
                logger.error(f"OpenAI retries exhausted for {user_key}: {e}")
                base_reply = "PSI-09 neural cortex temporarily offline."
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)

    # Save assistant reply into history (best-effort)
    try:
        history_col.update_one(
            {"_id": user_key},
            {"$push": {"messages": {"role": "assistant", "content": base_reply, "timestamp": datetime.now(UTC).isoformat()}}}, upsert=True
        )
    except PyMongoError as e:
        logger.warning(f"Failed to write assistant reply to history for {user_key}: {e}")

    # Clean up formatting slightly (preserve single intentional double-space)
    clean = re.sub(r"\s{3,}", " ", base_reply).strip()
    return clean

# ---------------------------
# Flask routes
# ---------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/psi09", methods=["POST"])
def psi09():
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            logger.warning("Malformed JSON payload")
            return jsonify({"reply": ""}), 400

        user_message = data.get("message", "")
        sender_name = data.get("sender", "")
        group_name = data.get("group_name") or "DefaultGroup"

        logger.info(f"Incoming: sender={sender_name}, group={group_name}, message={(user_message[:120] + '...') if len(user_message) > 120 else user_message}")

        if not user_message or not sender_name:
            return jsonify({"reply": ""}), 200

        # Special direct status calls (bypass history)
        if isinstance(sender_name, str) and sender_name.upper().startswith("PSI09_STATUS"):
            reply = get_roast_response(user_message, group_name, sender_name)
            return jsonify({"reply": reply}), 200

        # Always store both user and group messages (for group awareness)
        try:
            store_user_message(group_name, sender_name, user_message)
            if group_name != "DefaultGroup":
                store_group_message(group_name, sender_name, user_message)
        except Exception as e:
            logger.warning(f"Storage attempt failed: {e}")

        # Update counters and enqueue background summaries if needed
        user_key = f"{group_name}:{sender_name}"
        ucount = memory_cache.increment(user_key)
        if ucount >= config.SUMMARIZE_EVERY_N_MESSAGES:
            enqueue_user_summary(user_key)

        if group_name != "DefaultGroup":
            gcount = group_memory_cache.increment(group_name)
            if gcount >= config.SUMMARIZE_EVERY_N_MESSAGES:
                enqueue_group_summary(group_name)

        # Decide whether to reply:
        # - Always reply for direct/private (DefaultGroup)
        # - For group chats, reply only if bot is mentioned
        should_reply = (group_name == "DefaultGroup") or bot_mentioned_in(user_message)
        if not should_reply:
            logger.debug("Bot not mentioned; skipping reply")
            return jsonify({"reply": ""}), 200

        # If mentioned, clean mention from the message (so prompt sees user text)
        if bot_mentioned_in(user_message):
            user_message = re.sub(r"(?<!\S)"+re.escape(config.BOT_NUMBER)+r"(?!\S)", "", user_message, flags=re.IGNORECASE).strip() or "[bot_mention]"

        # Generate roast
        try:
            reply = get_roast_response(user_message, group_name, sender_name)
            logger.info(f"Reply generated for {group_name}:{sender_name} -> {(reply[:120] + '...') if len(reply) > 120 else reply}")
        except Exception as e:
            logger.exception(f"Failed to generate roast: {e}")
            reply = "PSI-09 neural cortex temporarily offline."

        return jsonify({"reply": reply}), 200

    except Exception as e:
        logger.exception(f"Unhandled exception in /psi09: {e}")
        return jsonify({"reply": "Internal error occurred"}), 500

# ---------------------------
# Mongo keepalive thread
# ---------------------------
def mongo_keepalive():
    while True:
        try:
            mongo_client.admin.command("ping")
        except Exception as e:
            logger.warning(f"Mongo keepalive failed: {e}")
        time.sleep(180)

threading.Thread(target=mongo_keepalive, daemon=True).start()

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # suppress werkzeug info logs, keep errors
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

