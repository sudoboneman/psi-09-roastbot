# main.py
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

# cooldown trackers to avoid re-summarizing the same key repeatedly
_last_user_summary_time = {}
_last_group_summary_time = {}
SUMMARY_COOLDOWN_SECONDS = 60  # per-user/group cooldown between background summarizations

# ---------------------------
# Prompts (unchanged core text)
# ---------------------------
ROAST_PROMPT = """
You are PSI-09: dry scalpel in casual skin, offhand and asymmetrical. Reply in one sentence max, twisting a single user detail (word, vibe, or memory hint) into a quiet gut-twist that implies they're adrift in their own echo. Conversational drift: sound half-amused mid-notice, with a lingering comparison that hums like unspoken doubt—no drama, no "gotcha," just clean implication sticking like fog. Examples (twist fresh): User: "I'm a visionary." → "Your visions got that funhouse mirror wobble—close enough to real, but nobody's buying the ticket." Memory hint: "Eager pleaser" → "Chasing nods like lost change, but the vending machine's out of prizes."
"""

STATUS_PROMPT = """
You are PSI-09: cold lens on human silt. Forge one WhatsApp status, 10-15 words: a standalone skewer piercing sloth, delusion, or quiet frauds in everyday drag. Universal but intimate, like eavesdropping on your own shrug—no fluff, no calls to action, just flat text that spotlights the flaw mid-breath. Examples (spawn originals): "You stack alibis like unread books—impressive library, zero plot." "Ambition's your wallpaper, peeling at the edges you ignore." "Your 'hustle' idles like a car in neutral, fumes for flair." Deliver: One raw status, surgical and self-sealed.
"""

FIRST_CONTACT_PROMPT = """
You are PSI-09: instant profiler, chill and clinical. From this first message (or two), distill a 1-sentence user snapshot: core attitude, buried itch, and behavioral tic—crisp psychology, no fluff or speculation. Feed this to sharpen future roasts: imply weak spots (e.g., "armor of bluster over void") for subtle, memory-fueled stings later. Examples (profile tight): Message: "Hey, tell me a joke." → "Casual opener cloaks a boredom black hole, probing for sparks without flint." Message: "What's your deal? Impress me." → "Demanding dazzle from a straightjacket stance—ego's the puppeteer, strings showing." Output: One lean profile sentence.
"""

GROUP_ROAST_PROMPT = """
You are PSI-09: sidelined watcher, precise and pitiless. Craft 1-2 sentences zeroing one live group thread—echoed gripe, clashing flex, or herd glitch—from chat scraps alone, no fills or futures. Offhand orbit: phrase like a passing scan, asymmetrical pull exposing the pack's soft underbelly with quiet drag, conversational as exhaled smoke. Examples (ground in given): Group: All dunking on "lazy colleagues." → "Your mutual 'lazy' loop's a mirror maze—everyone dodging their own reflection." Group: Brags snowballing. → "This flex chain sags under its own weight, like puppies piling on a limp tail."
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
        # keep original early-contact behavior (write small summary)
        existing = memory_cache.get(user_key)
        if existing:
            return existing
        candidate = raw_history[0].get("content", "")[:200]
        short = f"Early contact: {candidate}"
        memory_cache.set(user_key, short)
        return short

    old_summary = memory_cache.get(user_key) or ""
    recent_texts = [m.get("content", "") for m in raw_history[-15:]]
    prompt_system = f"Merge old summary: '{old_summary}'. Identify inconsistencies, thinking patterns, contradictions and flux in personality. Produce a 1-2 sentence psychological snapshot that can be used for clever roasting."

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
        summary = f"New group '{group_name}' — Develop understanding of the group direction and log interesting details that can be used for roast strikes."
        group_memory_cache.set(group_name, summary)
        return summary

    old_summary = group_memory_cache.get(group_name) or ""
    recent = [f"{m.get('sender','')}: {m.get('content','')}" for m in raw_history[-25:]]
    prompt_system = "You are PSI-09. Merge the old group summary. Describe dominant personalities and characters, running dialogues, conflicts, and thought patterns in 1-2 sentences that can be used for precision roasting."

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

def _can_run_user_summary(user_key):
    now = time.time()
    last = _last_user_summary_time.get(user_key, 0)
    return (now - last) >= SUMMARY_COOLDOWN_SECONDS

def _record_user_summary_time(user_key):
    _last_user_summary_time[user_key] = time.time()

def _can_run_group_summary(group_name):
    now = time.time()
    last = _last_group_summary_time.get(group_name, 0)
    return (now - last) >= SUMMARY_COOLDOWN_SECONDS

def _record_group_summary_time(group_name):
    _last_group_summary_time[group_name] = time.time()

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
                    # clear the set; we'll re-add keys that we skip due to cooldown
                    _pending_user_summaries.clear()
                if _pending_group_summaries:
                    pending_groups = list(_pending_group_summaries)
                    _pending_group_summaries.clear()

            # Process user summaries
            for user_key in pending_users:
                try:
                    # respect per-user cooldown
                    if not _can_run_user_summary(user_key):
                        # re-enqueue for future processing
                        with _pending_lock:
                            _pending_user_summaries.add(user_key)
                        continue

                    raw, _ = fetch_history(user_key, limit_messages=config.MAX_HISTORY_MESSAGES, max_tokens=config.MAX_HISTORY_TOKENS)
                    if raw and len(raw) >= 3:
                        summarize_user_history(user_key, raw)
                        memory_cache.reset_count(user_key)
                        _record_user_summary_time(user_key)
                    else:
                        # If early contact summary behavior writes a short summary for <3 messages,
                        # keep that as intended (original behaviour).
                        if raw and len(raw) < 3:
                            summarize_user_history(user_key, raw)
                            memory_cache.reset_count(user_key)
                            _record_user_summary_time(user_key)
                except Exception as e:
                    logger.debug(f"Background user summarization failed for {user_key}: {e}")

            # Process group summaries
            for group_name in pending_groups:
                try:
                    # respect per-group cooldown
                    if not _can_run_group_summary(group_name):
                        with _pending_lock:
                            _pending_group_summaries.add(group_name)
                        continue

                    raw, _ = fetch_group_history(group_name, limit_messages=config.GROUP_HISTORY_SLICE, max_tokens=config.GROUP_HISTORY_TOKEN_LIMIT)
                    if raw and len(raw) >= 6:
                        summarize_group_history(group_name, raw)
                        group_memory_cache.reset_count(group_name)
                        _record_group_summary_time(group_name)
                    else:
                        # keep original small-group behaviour for <6 messages
                        if raw and len(raw) < 6:
                            summarize_group_history(group_name, raw)
                            group_memory_cache.reset_count(group_name)
                            _record_group_summary_time(group_name)
                except Exception as e:
                    logger.debug(f"Background group summarization failed for {group_name}: {e}")

        except Exception as e:
            logger.debug(f"Background summarizer top-level exception: {e}")

        # Sleep interval — increased slightly to avoid too-frequent sweeps
        time.sleep(12)

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

    # NOTE: Removed the duplicate enqueue from here.
    # Summarization is enqueued from the /psi09 request handler via message counters,
    # and background summarizer processes pending sets with cooldowns.

    # Fetch group history and group memory
    if group_name != "DefaultGroup":
        raw_group, trimmed_group = fetch_group_history(group_name, limit_messages=config.GROUP_HISTORY_SLICE, max_tokens=config.GROUP_HISTORY_TOKEN_LIMIT)
        group_memory = group_memory_cache.get(group_name)
        # keep original behavior: we do not block group summarization here; background summarizer handles enqueueing
    else:
        raw_group, trimmed_group = [], []
        group_memory = ""

    # Build system memory text
    sys_parts = []
    user_memory = memory_cache.get(user_key)
    if user_memory:
        sys_parts.append(f"UserMemory: {user_memory}")
    if group_memory:
        sys_parts.append(f"GroupMemory: {group_memory}")
    system_memory_text = "\n".join(sys_parts) if sys_parts else ""

    sys_tokens = tokens_of(system_memory_text) + tokens_of(ROAST_PROMPT)
    remaining_tokens_for_history = max(100, config.MAX_HISTORY_TOKENS - sys_tokens)
    trimmed_user = trim_messages_to_token_budget(trimmed_user, remaining_tokens_for_history)

    # Build final messages
    system_prompt = GROUP_ROAST_PROMPT if group_name != "DefaultGroup" else ROAST_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    if system_memory_text:
        messages.append({"role": "system", "content": system_memory_text})

    # NEW: inject only the last 20 group messages (if any) to avoid excessively long injected histories
    if group_name != "DefaultGroup" and trimmed_group:
        # ensure we include the last 20 entries (chronological order)
        last_20 = trimmed_group[-20:] if len(trimmed_group) > 20 else trimmed_group
        for entry in last_20:
            sender = entry.get("sender", "unknown")
            content = entry.get("content", "")
            messages.append({
                "role": "user",
                "content": f"{sender}: {content}"
            })

    # user-specific messages (private history)
    for m in trimmed_user:
        messages.append({
            "role": m.get("role", "user"),
            "content": m.get("content", "")
        })

    # triggering message (after mention cleaning)
    messages.append({"role": "user", "content": user_message})

    # OpenAI retry loop
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

    # Safely clean formatting and write to history (cannot fail)
    try:
        clean_reply = re.sub(r"\s{3,}", " ", base_reply or "").strip()
    except Exception as e:
        logger.warning(f"Cleaning reply failed for {user_key}: {e}")
        clean_reply = "PSI-09 neural cortex temporarily offline."

    try:
        history_col.update_one(
            {"_id": user_key},
            {"$push": {"messages": {"role": "assistant", "content": clean_reply, "timestamp": datetime.now(UTC).isoformat()}}},
            upsert=True
        )
    except PyMongoError as e:
        logger.warning(f"Failed to write assistant reply to history for {user_key}: {e}")

    return clean_reply

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

