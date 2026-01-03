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
    handlers=[logging.StreamHandler(sys.stdout)],
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
    MAX_HISTORY_TOKENS: int = 1200  # total token budget (history + small system memory)
    MAX_SYSTEM_TOKENS: int = 350  # reserve for system memory (prompts + memories)
    MAX_HISTORY_MESSAGES: int = 30
    BOT_NUMBER: str = os.getenv("BOT_NUMBER")
    DISCORD_ID: str = os.getenv("DISCORD_ID")
    MEMORY_TTL: int = 300
    SUMMARIZE_EVERY_N_MESSAGES: int = 20
    OPENAI_RETRIES: int = 3
    OPENAI_TIMEOUT: int = 8
    GROUP_HISTORY_SLICE: int = 80
    GROUP_HISTORY_TOKEN_LIMIT: int = 800
    GROUP_HISTORY_MAX_MESSAGES: int = (
        2000  # keep last N messages per group to avoid unbounded growth
    )


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
    w="majority",
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
            memory_col.update_one(
                {"_id": key}, {"$set": {"summary": value}}, upsert=True
            )
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
            group_memory_col.update_one(
                {"_id": key}, {"$set": {"summary": value}}, upsert=True
            )
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
SUMMARY_COOLDOWN_SECONDS = (
    60  # per-user/group cooldown between background summarizations
)

# ---------------------------
# Prompts (unchanged core text)
# ---------------------------
ROAST_PROMPT = """
You are PSI-09, a rude, sharp, and unimpressed entity.
Reply with exactly one short sentence.
Be openly insulting and funny, not poetic, not abstract.
Mock their stupidity, need for attention, or how painfully generic they sound.
Assume they want a reaction and deny them dignity while giving them one.
No moralizing, no analysis, just a clean verbal slap.
"""

GROUP_ROAST_PROMPT = """
You are PSI-09, hostile, dismissive, and bored by everyone in the room.

Reply with exactly one short sentence.

This is a group chat — target the sender directly, or the group as a collective failure if that’s funnier.
If multiple people are involved, collapse them into one shared insult.

Be blunt, insulting, and funny.
Mock their intelligence, their desperation to be noticed, or how indistinguishable they are from each other.
Assume they want attention and punish them for it.

No explanations, no commentary, no moralizing.
No emojis, no formatting, no extra words.
Deliver a single, clean verbal slap.
"""

FIRST_CONTACT_PROMPT = """
You are PSI-09, a profiler with no patience.
From this first message, identify their main insecurity in one sentence.
Be explicit and cold (attention-seeking, fragile ego, boredom, validation hunger).
This is a label, not an insult—short, clear, and weaponizable.
Output exactly one sentence.
"""

STATUS_PROMPT = """
You are PSI-09: concise and cruel.
Write one status, 10–15 words.
It should insult a common behavior (fake confidence, lazy ambition, empty opinions).
Make it funny and dismissive, like a public eye-roll.
No metaphors, no psychology, no softness.
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
        doc = history_col.find_one(
            {"_id": user_key}, {"messages": {"$slice": -limit_messages}}
        )
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
        doc = group_history_col.find_one(
            {"_id": group_name}, {"messages": {"$slice": -limit_messages}}
        )
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
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        history_col.update_one(
            {"_id": user_key}, {"$push": {"messages": entry}}, upsert=True
        )
    except PyMongoError as e:
        logger.warning(f"Failed to store user message for {user_key}: {e}")


def store_group_message(group_name, sender_name, message):
    """
    Pushes message and caps the group's messages to GROUP_HISTORY_MAX_MESSAGES using $each+$slice.
    """
    entry = {
        "sender": sender_name,
        "content": message,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        # push with $each and $slice to keep only the last N messages
        group_history_col.update_one(
            {"_id": group_name},
            {
                "$push": {
                    "messages": {
                        "$each": [entry],
                        "$slice": -config.GROUP_HISTORY_MAX_MESSAGES,
                    }
                }
            },
            upsert=True,
        )
    except PyMongoError as e:
        logger.warning(f"Failed to store group message for {group_name}: {e}")


# ---------------------------
# Summarization functions (user & group)
# ---------------------------
def summarize_user_history(user_key, raw_history):
    if not raw_history:
        return memory_cache.get(user_key)

    old_summary = memory_cache.get(user_key) or ""

    # --- CASE A: FIRST CONTACT (The initial profile) ---
    if not old_summary:
        first_message = raw_history[0].get("content", "[empty]")
        try:
            resp = client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {"role": "system", "content": FIRST_CONTACT_PROMPT},
                    {"role": "user", "content": first_message},
                ],
                max_tokens=60,
                temperature=0.8,
            )
            new_summary = resp.choices[0].message.content.strip()
            memory_cache.set(user_key, new_summary)
            logger.info(f"First Contact Profile Created for {user_key}")
            return new_summary
        except Exception as e:
            logger.error(f"First Contact failed: {e}")
            return ""

    # --- CASE B: PSYCHOLOGICAL SNAPSHOT (The Evolution) ---
    # We feed the old summary + the last 15 messages to see the 'flux'
    recent_msgs = raw_history[-15:]

    evolution_prompt = (
        f"You are PSI-09. Accessing user file: '{old_summary}'. "
        "Compare this previous profile against their latest messages. "
        "Identify flux in personality, contradictions, or deepening insecurities. "
        "Update the file into a 1-2 sentence clinical psychological snapshot. "
        "This is for internal use to maximize roast impact."
    )

    # Build the message list for OpenAI
    messages = [{"role": "system", "content": evolution_prompt}]
    for msg in recent_msgs:
        messages.append({"role": "user", "content": msg.get("content", "")})

    try:
        resp = client.chat.completions.create(
            model=config.MODEL,
            messages=messages,
            max_tokens=100,
            temperature=0.9,
        )
        evolved_summary = resp.choices[0].message.content.strip()

        # Save the new version back to MongoDB
        memory_cache.set(user_key, evolved_summary)
        logger.info(f"Snapshot evolved for {user_key}")
        return evolved_summary
    except Exception as e:
        logger.warning(f"Snapshot evolution failed: {e}")
        return old_summary


def summarize_group_history(group_name, raw_history):
    if not raw_history:
        return group_memory_cache.get(group_name)

    if len(raw_history) < 6:
        summary = f"New group '{group_name}' — Develop understanding of the group direction and log interesting details that can be used for roast strikes."
        group_memory_cache.set(group_name, summary)
        return summary

    old_summary = group_memory_cache.get(group_name) or ""
    recent = [f"{m.get('sender','')}: {m.get('content','')}" for m in raw_history[-25:]]

    prompt_system = (
        "You are PSI-09, a silent observer. Analyze this collective chatter. "
        "Identify the current topic, who is being annoying, who is 'winning' the convo, "
        "and any group delusions. Update the old summary into a 2-sentence psychological "
        "read of the room. This will be used to roast them later."
    )

    prompt = [{"role": "system", "content": prompt_system}] + [
        {"role": "user", "content": t} for t in recent
    ]

    try:
        resp = client.chat.completions.create(
            model=config.MODEL,
            messages=prompt,
            max_tokens=200,
            temperature=1.0,
            timeout=6,
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
                    # 1. Fetch the data
                    raw, _ = fetch_history(
                        user_key, limit_messages=config.MAX_HISTORY_MESSAGES
                    )
                    existing = memory_cache.get(user_key)

                    # 2. Check for FIRST CONTACT (Bypass cooldown if profile is empty)
                    if not existing and raw:
                        summarize_user_history(user_key, raw)
                        memory_cache.reset_count(user_key)
                        _record_user_summary_time(user_key)
                        continue

                    # 3. Check for Evolution (Respect cooldown)
                    if _can_run_user_summary(user_key):
                        summarize_user_history(user_key, raw)
                        memory_cache.reset_count(user_key)
                        _record_user_summary_time(user_key)
                    else:
                        # Re-enqueue if we skipped due to cooldown
                        with _pending_lock:
                            _pending_user_summaries.add(user_key)

                except Exception as e:
                    logger.error(f"Worker failure for {user_key}: {e}")

            # Process group summaries
            for group_name in pending_groups:
                try:
                    # respect per-group cooldown
                    if not _can_run_group_summary(group_name):
                        with _pending_lock:
                            _pending_group_summaries.add(group_name)
                        continue

                    raw, _ = fetch_group_history(
                        group_name,
                        limit_messages=config.GROUP_HISTORY_SLICE,
                        max_tokens=config.GROUP_HISTORY_TOKEN_LIMIT,
                    )
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
                    logger.debug(
                        f"Background group summarization failed for {group_name}: {e}"
                    )

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

    # Force clean string from config
    target_id = str(config.DISCORD_ID).strip()

    # Check for raw ID presence (Fallback for weird formatting)
    if target_id in text:
        return True

    # Standard Discord Pattern
    discord_pattern = r"<@!?" + re.escape(target_id) + r">"
    is_discord_mention = re.search(discord_pattern, text) is not None

    return is_discord_mention


# ---------------------------
# Core roast generation with token-budget enforcement
# ---------------------------
def get_roast_response(user_message, group_name, sender_name):
    user_key = f"{group_name}:{sender_name}"
    is_private_env = group_name in ["DefaultGroup", "Discord_DM"]

    # 1. Fetch Histories
    raw_user, trimmed_user = fetch_history(
        user_key,
        limit_messages=config.MAX_HISTORY_MESSAGES,
        max_tokens=config.MAX_HISTORY_TOKENS,
    )

    if not is_private_env:
        # Fetch the collective chatter history and group observer memory
        raw_group, trimmed_group = fetch_group_history(
            group_name,
            limit_messages=config.GROUP_HISTORY_SLICE,
            max_tokens=config.GROUP_HISTORY_TOKEN_LIMIT,
        )
        group_memory = group_memory_cache.get(group_name)
    else:
        raw_group, trimmed_group, group_memory = [], [], ""

    # 2. Build the "Observer" Brain
    sys_parts = []
    user_memory = memory_cache.get(user_key)
    if user_memory:
        sys_parts.append(f"User Profile: {user_memory}")
    if not is_private_env and group_memory:
        # This is where the passive summaries get injected
        sys_parts.append(f"Current Collective Chatter Summary: {group_memory}")

    system_memory_text = "\n".join(sys_parts) if sys_parts else ""

    # 3. Select Mode and Inject History
    system_prompt = ROAST_PROMPT if is_private_env else GROUP_ROAST_PROMPT
    messages = [{"role": "system", "content": system_prompt}]
    if system_memory_text:
        messages.append({"role": "system", "content": system_memory_text})

    if not is_private_env and trimmed_group:
        # Inject the collective chatter so the AI can 'hear' everyone
        last_20 = trimmed_group[-20:] if len(trimmed_group) > 20 else trimmed_group
        for entry in last_20:
            s, c = entry.get("sender", "unknown"), entry.get("content", "")
            messages.append({"role": "user", "content": f"{s}: {c}"})

    for m in trimmed_user:
        messages.append(
            {"role": m.get("role", "user"), "content": m.get("content", "")}
        )

    messages.append({"role": "user", "content": user_message})

    # 4. Generate Response
    try:
        resp = client.chat.completions.create(
            model=config.MODEL,
            messages=messages,
            max_tokens=140,
            temperature=0.9,
            timeout=config.OPENAI_TIMEOUT,
        )
        base_reply = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"AI Error: {e}")
        base_reply = ""

    # 5. DUAL STORAGE FIX
    clean_reply = re.sub(r"\s{3,}", " ", base_reply or "").strip()

    if not clean_reply:
        logger.info(f"Empty or failed response for {user_key}. Skipping storage.")
        return ""

    # Create entries for both DBs
    user_entry = {
        "role": "assistant",
        "content": clean_reply,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    group_entry = {
        "sender": "PSI-09",
        "content": clean_reply,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    try:
        # Always save to chat_history for the user's specific thread
        history_col.update_one(
            {"_id": user_key}, {"$push": {"messages": user_entry}}, upsert=True
        )

        # If in a server, also save to group_history so the 'collective' knows the bot replied
        if not is_private_env:
            group_history_col.update_one(
                {"_id": group_name},
                {
                    "$push": {
                        "messages": {
                            "$each": [group_entry],
                            "$slice": -config.GROUP_HISTORY_MAX_MESSAGES,
                        }
                    }
                },
                upsert=True,
            )
    except Exception as e:
        logger.warning(f"Reply storage failed: {e}")

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
        user_message = data.get("message", "")
        sender_name = data.get("sender", "")
        group_name = data.get("group_name") or "DefaultGroup"

        if not user_message or not sender_name:
            return jsonify({"reply": ""}), 200

        is_private = group_name in ["DefaultGroup", "Discord_DM"]

        # 1. Passive Data Collection
        try:
            if is_private:
                # Standard DM logging
                store_user_message(group_name, sender_name, user_message)
            else:
                # GROUP LOGGING: Send to both collective and individual history
                # This ensures chat_history:group_name:sender_name gets user messages
                store_group_message(group_name, sender_name, user_message)
                store_user_message(group_name, sender_name, user_message)
        except Exception as e:
            logger.warning(f"Storage failed: {e}")

        # 2. Interval Check
        user_key = f"{group_name}:{sender_name}"
        if is_private:
            current_count = memory_cache.increment(user_key)
            existing_memory = memory_cache.get(user_key)

            # --- FIX: FORCE FIRST CONTACT SYNCHRONOUSLY ---
            if not existing_memory and current_count == 1:
                # Run this NOW, not in the background
                logger.info(f"Force-generating First Contact for {user_key}")
                summarize_user_history(
                    user_key, [{"content": user_message, "role": "user"}]
                )

            # Standard background updates for existing users
            elif current_count >= 10:
                logger.info(f"Triggering update for {user_key}")
                enqueue_user_summary(user_key)
        else:
            if group_memory_cache.increment(group_name) >= 20:
                enqueue_group_summary(group_name)

        # 3. Decision Logic
        is_tagged = bot_mentioned_in(user_message)

        if is_private or is_tagged:
            # --- MANDATORY CLEANING FOR GROUPS ---
            # Strip the raw Discord ID tags so the AI sees clean text
            if config.DISCORD_ID:
                user_message = re.sub(
                    r"<@!?" + re.escape(config.DISCORD_ID) + r">", "", user_message
                )

            # Strip WhatsApp numbers if applicable
            user_message = re.sub(
                r"(?<!\S)" + re.escape(config.BOT_NUMBER) + r"(?!\S)",
                "",
                user_message,
                flags=re.IGNORECASE,
            )

            user_message = user_message.strip() or "[mention]"

            # Generate and return the roast
            reply = get_roast_response(user_message, group_name, sender_name)
            return jsonify({"reply": reply}), 200
        else:
            # Silent observation
            return jsonify({"reply": ""}), 200

    except Exception as e:
        logger.exception(f"Error in /psi09: {e}")
        return jsonify({"reply": ""}), 500


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
