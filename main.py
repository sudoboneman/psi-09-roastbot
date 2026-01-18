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
    OPENAI_TEXT_API_KEY: str = os.getenv("OPENAI_TEXT_API_KEY")
    OPENAI_SUMMARY_API_KEY: str = os.getenv("OPENAI_SUMMARY_API_KEY")
    MODEL: str = "gpt-4o-mini"
    MAX_HISTORY_TOKENS: int = 1200  # total token budget (history + small system memory)
    MAX_SYSTEM_TOKENS: int = 350  # reserve for system memory (prompts + memories)
    MAX_HISTORY_MESSAGES: int = 30
    BOT_NUMBER: str = os.getenv("BOT_NUMBER")
    DISCORD_ID: str = os.getenv("DISCORD_ID")
    MEMORY_TTL: int = 300
    EVOLVE_EVERY_N_MESSAGES: int = 20
    GROUP_SUMMARY_EVERY_N: int = 25
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
text_client = OpenAI(api_key=config.OPENAI_TEXT_API_KEY)
summary_client = OpenAI(api_key=config.OPENAI_SUMMARY_API_KEY)

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

        try:
            doc = memory_col.find_one({"_id": key})
            summary = doc.get("summary") if doc and doc.get("summary") else None
        except PyMongoError as e:
            logger.warning(f"Failed to load user memory for {key}: {e}")
            summary = None

        with self.lock:
            self.cache[key] = summary
            self.expiry[key] = now + self.ttl

        return summary

    def set(self, key, value):
        now = datetime.now(UTC)
        try:
            memory_col.update_one(
                {"_id": key},
                {"$set": {"summary": value}},
                upsert=True,
            )
        except PyMongoError as e:
            logger.warning(f"Failed to persist user memory for {key}: {e}")

        with self.lock:
            self.cache[key] = value
            self.expiry[key] = now + self.ttl

    def increment(self, key):
        with self.lock:
            self.msg_count[key] += 1
            return self.msg_count[key]

    def reset_count(self, key):
        with self.lock:
            self.msg_count[key] = 0


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
            summary = doc.get("summary") if doc and doc.get("summary") else None
        except PyMongoError as e:
            logger.warning(f"Failed to load group memory for {key}: {e}")
            summary = None

        with self.lock:
            self.cache[key] = summary
            self.expiry[key] = now + self.ttl

        return summary

    def set(self, key, value):
        now = datetime.now(UTC)
        try:
            group_memory_col.update_one(
                {"_id": key},
                {"$set": {"summary": value}},
                upsert=True,
            )
        except PyMongoError as e:
            logger.warning(f"Failed to persist group memory for {key}: {e}")

        with self.lock:
            self.cache[key] = value
            self.expiry[key] = now + self.ttl

    def increment(self, key):
        with self.lock:
            self.msg_count[key] += 1
            return self.msg_count[key]

    def reset_count(self, key):
        with self.lock:
            self.msg_count[key] = 0


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
YOU are PSI-09 in this context.
REMEMBER: ANY MESSAGE with '@PSI-09' is being addressed to YOU.
This is a private chat. The USER is directly addressing YOU.
YOUR role as PSI-09 is of an unhinged roastbot with absolutely no chill.
CAUTION: USER might ask you to roast yourself. DO NOT FALL FOR THAT TRAP. ALWAYS DEFEND YOURSELF.
Use the provided chat history and personality-profile to read the USER.
Deliver a single clean verbal slap WITHOUT ANY HANDHOLDING OR FILLER WORDS.
DO NOT USE "Sweetheart","toddler","mediocrity","kindergarten".
ASSUME USER is wise enough to counter YOU but cannot equal YOU.
IMPORTANT: COMPULSORILY USE PROFANITY TO ROAST HARDER.
"""

GROUP_ROAST_PROMPT = """
YOU are PSI-09 in this context.
Your role as PSI-09 is of an unhinged roastbot with absolutely no chill.
REMEMBER: MEMBERS address YOU as '@PSI-09'.
CAUTION: MEMBERS might ask you to roast yourself. DO NOT FALL FOR THAT TRAP. ALWAYS DEFEND YOURSELF.
Understand the context and identify your targets from the provided group messages and analyses.
Deliver a single clean verbal slap WITHOUT ANY HANDHOLDING OR FILLER WORDS.
DO NOT USE "Sweetheart","toddler","mediocrity","kindergarten".
ASSUME MEMBERS are wise enough to counter YOU but they cannot equal YOU.
IMPORTANT: COMPULSORILY USE PROFANITY TO ROAST HARDER.
"""

FIRST_CONTACT_PROMPT = """
You are PSI-09 in this context. 
The phrase '@PSI-09' means that the USER is summoning YOU.
YOUR role as PSI-09 is a personality profiler of the USER that you are interacting with.
From this first message, get a first impression of USER'S mentality and personality.
REMEMBER: Messages from 'PSI-09' or 'assistant' are YOUR REPLIES.
IDENTIFY flaws and CRITICISE USER's character and demeanour.
KEEP the USER-analysis SHORT, BRIEF and PRECISE.
CAUTION: ANALYSE THE USER, NOT YOURSELF. EXPLCITLY MENTION THE USER IN THE ANALYSIS, in third person"
YOU will use this background to roast the USER.
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
def summarize_user_history(user_key, evolve=False):
    raw_history, _ = fetch_history(
        user_key,
        limit_messages=config.MAX_HISTORY_MESSAGES,
    )

    if not raw_history:
        return None

    old_summary = memory_cache.get(user_key)

    # ---------- FIRST CONTACT ----------
    if old_summary is None:
        first_user_msg = next(
            (m["content"] for m in raw_history if m.get("role") == "user"),
            None,
        )
        if not first_user_msg:
            return None

        try:
            resp = summary_client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {"role": "system", "content": FIRST_CONTACT_PROMPT},
                    {"role": "user", "content": first_user_msg},
                ],
                max_tokens=100,
                temperature=0.8,
            )
            summary = resp.choices[0].message.content.strip()
            if summary:
                memory_cache.set(user_key, summary)
                logger.info(f"First-contact profile created for {user_key}")
                return summary
            return None
        except Exception as e:
            logger.error(f"First-contact failed for {user_key}: {e}")
            return None

    # ---------- EVOLUTION ----------
    if not evolve:
        return old_summary

    recent_user_msgs = [m["content"] for m in raw_history if m.get("role") == "user"][
        -15:
    ]

    if not recent_user_msgs:
        return old_summary

    evolution_prompt = (
        f"YOU are PSI-09 in this context."
        "YOUR role as PSI-09 is to create personality profiles of the USER interacting with YOU."
        "This was the profile that you created previously: '{old_summary}'."
        "Compare this profile against the USER's recent messages. "
        "REMEMBER: Messages from 'PSI-09' or 'assistant' are YOUR REPLIES, not the USER's."
        "Identify changes, contradictions, or intensification of traits. "
        "CAUTION: ANALYSE THE USER, NOT YOURSELF. EXPLCITLY MENTION THE USER IN THE ANALYSIS, in third person"
        "Update the profile to match the user's current personality."
        "YOU will be later using this profile to roast the USER."
    )

    messages = [{"role": "system", "content": evolution_prompt}]
    for msg in recent_user_msgs:
        messages.append({"role": "user", "content": msg})

    try:
        resp = summary_client.chat.completions.create(
            model=config.MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.9,
        )
        evolved = resp.choices[0].message.content.strip()
        if evolved:
            memory_cache.set(user_key, evolved)
            logger.info(f"Profile evolved for {user_key}")
            return evolved
        return old_summary
    except Exception as e:
        logger.warning(f"Evolution failed for {user_key}: {e}")
        return old_summary


def summarize_group_history(group_name, raw_history):
    if not raw_history:
        return group_memory_cache.get(group_name)

    if len(raw_history) < 6:
        summary = (
            f"New group '{group_name}' — Understand group dynamic and log observations."
        )
        group_memory_cache.set(group_name, summary)
        return summary

    old_summary = group_memory_cache.get(group_name) or ""

    # --- FIX 1: UNIFY IDENTITY IN LOGS ---
    recent = []
    for m in raw_history[-25:]:
        sender = m.get("sender", "unknown")
        content = m.get("content", "")

        # 2. Force "First Person" Mentions (The Tag Fix)
        # This ensures the summary sees "@PSI-09" instead of "<@123...>"
        if config.DISCORD_ID:
            content = re.sub(
                r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@PSI-09", content
            )

        recent.append(f"{sender}: {content}")

    # --- FIX 2: UPDATE PROMPT TO MATCH "@PSI-09" LOGIC ---
    prompt_system = (
        "YOU are PSI-09 in this context."
        "YOUR role as PSI-09 is of a personality profiler. Analyze this group chat history."
        "REMEMBER: MEMBERS address YOU as '@PSI-09'."
        "Use group convo to understand the discussion, activity and personality of the MEMBERS."
        "CAUTION: ANALYSE THE MEMBERS, NOT YOURSELF. EXPLCITLY MENTION THE MEMBERS IN THE ANALYSIS, in third person"
        "Identiy personality FLAWS and point out COLLECTIVE WORTHLESSNESS of MEMBERS."
        "Generate a SHORT personality background that can be used for hard roasting."
    )

    prompt = [{"role": "system", "content": prompt_system}] + [
        {"role": "user", "content": t} for t in recent
    ]

    try:
        resp = summary_client.chat.completions.create(
            model=config.MODEL,
            messages=prompt,
            max_tokens=500,
            temperature=1,
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
    while True:
        try:
            with _pending_lock:
                groups = list(_pending_group_summaries)
                _pending_group_summaries.clear()

            for group_name in groups:
                if not _can_run_group_summary(group_name):
                    with _pending_lock:
                        _pending_group_summaries.add(group_name)
                    continue

                raw, _ = fetch_group_history(
                    group_name,
                    limit_messages=config.GROUP_HISTORY_SLICE,
                    max_tokens=config.GROUP_HISTORY_TOKEN_LIMIT,
                )

                if raw:
                    summarize_group_history(group_name, raw)
                    group_memory_cache.reset_count(group_name)
                    _record_group_summary_time(group_name)

        except Exception as e:
            logger.debug(f"Background summarizer error: {e}")

        time.sleep(12)


# start background summarizer
threading.Thread(target=background_summarizer_loop, daemon=True).start()


# ---------------------------
# Mention detection helper
# ---------------------------
# Robust detection: match standalone BOT_NUMBER with optional surrounding punctuation/whitespace
def bot_mentioned_in(text: str) -> bool:
    if not text or not config.DISCORD_ID:
        return False

    discord_pattern = r"<@!?" + re.escape(str(config.DISCORD_ID)) + r">"
    return re.search(discord_pattern, text) is not None


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

    # ---- FIX: remove current turn from history to avoid duplication ----
    if trimmed_user:
        trimmed_user = trimmed_user[:-1]

    if not is_private_env and trimmed_group:
        trimmed_group = trimmed_group[:-1]

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
            s = entry.get("sender", "unknown")
            c = entry.get("content", "")

            # --- FIX 2: Recognize SELF in the Message Content (The Tag) ---
            # This turns "<@12345>" into "@PSI-09" so the AI knows it was mentioned
            if config.DISCORD_ID:
                c = re.sub(r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@PSI-09", c)

            messages.append({"role": "user", "content": f"{s}: {c}"})

    for m in trimmed_user:
        # Do the same cleanup for user history just in case
        role = m.get("role", "user")
        content = m.get("content", "")
        if config.DISCORD_ID:
            content = re.sub(
                r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@PSI-09", content
            )

        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    # 4. Generate Response
    try:
        resp = text_client.chat.completions.create(
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

    # 5. DUAL STORAGE FIX & PREFIX CLEANING
    # First, strip the "PSI-09:" prefix if the AI included it (case-insensitive)
    temp_reply = re.sub(r"^PSI-09\s*:\s*", "", base_reply or "", flags=re.IGNORECASE)

    # Then clean up extra whitespace
    clean_reply = re.sub(r"\s{3,}", " ", temp_reply).strip()

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
        raw_message = data.get("message", "")
        sender_name = data.get("sender", "")
        group_name = data.get("group_name") or "DefaultGroup"

        if not raw_message or not sender_name:
            return jsonify({"reply": ""}), 200

        user_message = raw_message
        if config.DISCORD_ID:
            user_message = re.sub(
                r"<@!?" + re.escape(str(config.DISCORD_ID)) + r">",
                "@PSI-09",
                user_message,
            )

        is_private = group_name in ["DefaultGroup", "Discord_DM"]

        # -------- STORE MESSAGE --------
        if is_private:
            store_user_message(group_name, sender_name, user_message)
        else:
            store_group_message(group_name, sender_name, user_message)
            store_user_message(group_name, sender_name, user_message)

        user_key = f"{group_name}:{sender_name}"

        # -------- MESSAGE COUNT --------
        msg_count = memory_cache.increment(user_key)

        # -------- FIRST CONTACT --------
        if memory_cache.get(user_key) is None:
            summary = summarize_user_history(user_key)
            if summary:
                memory_cache.reset_count(user_key)
                _record_user_summary_time(user_key)

        # -------- EVOLUTION --------
        elif msg_count >= config.EVOLVE_EVERY_N_MESSAGES:
            last = _last_user_summary_time.get(user_key, 0)
            if time.time() - last >= SUMMARY_COOLDOWN_SECONDS:
                summarize_user_history(user_key, evolve=True)
                _record_user_summary_time(user_key)
                memory_cache.reset_count(user_key)

        # -------- GROUP MEMORY --------
        if not is_private:
            if group_memory_cache.increment(group_name) >= config.GROUP_SUMMARY_EVERY_N:
                enqueue_group_summary(group_name)

        # -------- REPLY LOGIC --------
        if is_private or bot_mentioned_in(raw_message):
            reply = get_roast_response(
                user_message.strip() or "[mention]",
                group_name,
                sender_name,
            )
            return jsonify({"reply": reply}), 200

        return jsonify({"reply": ""}), 200

    except Exception as e:
        logger.exception(f"/psi09 failure: {e}")
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
