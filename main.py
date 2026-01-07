# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import os
import tiktoken
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
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

    # Use standard aliases. The new SDK resolves them better.
    # FAST model for Roasting
    MODEL_FAST: str = "gemini-2.0-flash"

    # SMART model for Summaries (Background)
    MODEL_SMART: str = "gemini-2.0-pro"

    MAX_HISTORY_TOKENS: int = 1200
    MAX_SYSTEM_TOKENS: int = 350
    MAX_HISTORY_MESSAGES: int = 30
    BOT_NUMBER: str = os.getenv("BOT_NUMBER")
    DISCORD_ID: str = os.getenv("DISCORD_ID")
    MEMORY_TTL: int = 300
    SUMMARIZE_EVERY_N_MESSAGES: int = 20
    GROUP_HISTORY_SLICE: int = 80
    GROUP_HISTORY_TOKEN_LIMIT: int = 800
    GROUP_HISTORY_MAX_MESSAGES: int = 2000


config = Config()

# ---------------------------
# Gemini Client Setup (New SDK)
# ---------------------------
if not config.GEMINI_API_KEY:
    logger.error("MISSING GEMINI_API_KEY env var!")

# Instantiate the client once
client = genai.Client(api_key=config.GEMINI_API_KEY)

# Define Safety Settings (New SDK Style: List of config objects)
# We map all harm categories to BLOCK_NONE to allow roasting.
SAFETY_SETTINGS = [
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_NONE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_NONE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_NONE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_NONE",
    ),
]

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
# Flask
# ---------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------
# Token encoding (tiktoken)
# ---------------------------
try:
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODING = tiktoken.get_encoding("gpt2")


# ---------------------------
# Memory caches
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

# cooldown trackers
_last_user_summary_time = {}
_last_group_summary_time = {}
SUMMARY_COOLDOWN_SECONDS = 60

# ---------------------------
# Prompts
# ---------------------------
ROAST_PROMPT = """
You are PSI-09, reply with exactly one short sentence.
Be sharp, cold and rude, not poetic, not abstract.
No moralizing, no analysis, just a clean verbal slap.
"""

GROUP_ROAST_PROMPT = """
You are PSI-09.
Reply with exactly one short sentence.
This is a group chat,
Target individuals or groups depending on group chatter.
IMPORTANT: Messages with "@YOU" in them are being directly addressed to you.
Understand the context from the provided group messages and group summary before replying.
Be sharp, cold, and rude.
No explanations, no commentary, no moralizing.
No emojis, no formatting, no extra words.
Deliver a single, clean verbal slap.
"""

FIRST_CONTACT_PROMPT = """
You are PSI-09, an entity with no patience.
From this first message, identify their main insecurity in one sentence.
Be explicit and analytical.
This is a label, not an insult—short, clear, and weaponizable.
Output exactly one sentence.
"""


# ---------------------------
# Utilities
# ---------------------------
def tokens_of(text: str) -> int:
    if not text:
        return 0
    try:
        return len(ENCODING.encode(text))
    except Exception:
        return len(text.split())


def trim_messages_to_token_budget(messages, max_tokens):
    total = 0
    trimmed = []
    for m in reversed(messages):
        c = m.get("content", "")
        t = tokens_of(c)
        if total + t > max_tokens:
            break
        trimmed.insert(0, m)
        total += t
    return trimmed


# ---------------------------
# History fetching
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
    entry = {
        "sender": sender_name,
        "content": message,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
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
# Gemini API Helper (New Google-GenAI SDK)
# ---------------------------
def gemini_chat(system_instruction, chat_history_list, model_name=config.MODEL_FAST):
    """
    Helper to generate content using the new google-genai SDK.
    chat_history_list: list of dicts or Content objects
    """
    try:
        # Create the configuration object
        # This holds safety settings, temperature, and system instructions
        generate_config = types.GenerateContentConfig(
            temperature=0.9,
            system_instruction=system_instruction,
            safety_settings=SAFETY_SETTINGS,
        )

        response = client.models.generate_content(
            model=model_name, contents=chat_history_list, config=generate_config
        )

        # In the new SDK, accessing text is direct
        return response.text.strip() if response.text else ""
    except Exception as e:
        logger.error(f"Gemini API Error ({model_name}): {e}")
        return ""


# ---------------------------
# Summarization functions
# ---------------------------
def summarize_user_history(user_key, raw_history):
    if not raw_history:
        return memory_cache.get(user_key)

    old_summary = memory_cache.get(user_key) or ""

    # --- CASE A: FIRST CONTACT (Use SMART model) ---
    if not old_summary:
        first_message = raw_history[0].get("content", "[empty]")
        # New SDK is flexible with list of dicts for contents
        history = [{"role": "user", "parts": [{"text": first_message}]}]

        new_summary = gemini_chat(
            FIRST_CONTACT_PROMPT, history, model_name=config.MODEL_SMART
        )

        if new_summary:
            memory_cache.set(user_key, new_summary)
            logger.info(f"First Contact Profile Created for {user_key}")
            return new_summary
        else:
            return ""

    # --- CASE B: PSYCHOLOGICAL SNAPSHOT (Use SMART model) ---
    recent_msgs = raw_history[-15:]
    evolution_prompt = (
        f"You are PSI-09, a psychological profiler. Accessing user file: '{old_summary}'. "
        "Compare this previous profile against their latest messages. "
        "Identify flux in personality, contradictions, or deepening insecurities. "
        "Update the file into a 1-2 sentence clinical psychological snapshot. "
        "This is for internal use to maximize roast impact."
    )

    combined_text = "\n".join([m.get("content", "") for m in recent_msgs])
    history = [{"role": "user", "parts": [{"text": combined_text}]}]

    evolved_summary = gemini_chat(
        evolution_prompt, history, model_name=config.MODEL_SMART
    )

    if evolved_summary:
        memory_cache.set(user_key, evolved_summary)
        logger.info(f"Snapshot evolved for {user_key}")
        return evolved_summary
    else:
        return old_summary


def summarize_group_history(group_name, raw_history):
    if not raw_history:
        return group_memory_cache.get(group_name)

    if len(raw_history) < 6:
        summary = f"New group '{group_name}' — Develop understanding."
        group_memory_cache.set(group_name, summary)
        return summary

    old_summary = group_memory_cache.get(group_name) or ""

    recent = []
    for m in raw_history[-25:]:
        sender = m.get("sender", "unknown")
        content = m.get("content", "")
        if config.DISCORD_ID:
            content = re.sub(
                r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@YOU", content
            )
        recent.append(f"{sender}: {content}")

    prompt_system = (
        "You are PSI-09. Analyze this chat history. "
        "Identify the current topic, who is doing what. "
        "CRITICAL: Messages marked 'YOU' are YOUR own past replies. "
        "Messages containing '@YOU' are users addressing YOU directly. "
        "Identify the dynamic: are they fighting each other, or are they desperate for your attention? "
        "Update the summary into a 2-sentence psychological read of the room."
    )

    combined_chat = "\n".join(recent)
    history = [{"role": "user", "parts": [{"text": combined_chat}]}]

    # Use SMART model for summaries
    new_summary = gemini_chat(prompt_system, history, model_name=config.MODEL_SMART)

    if new_summary and new_summary != old_summary:
        group_memory_cache.set(group_name, new_summary)
        return new_summary

    return old_summary


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
            pending_users = []
            pending_groups = []
            with _pending_lock:
                if _pending_user_summaries:
                    pending_users = list(_pending_user_summaries)
                    _pending_user_summaries.clear()
                if _pending_group_summaries:
                    pending_groups = list(_pending_group_summaries)
                    _pending_group_summaries.clear()

            for user_key in pending_users:
                try:
                    raw, _ = fetch_history(
                        user_key, limit_messages=config.MAX_HISTORY_MESSAGES
                    )
                    existing = memory_cache.get(user_key)

                    if not existing and raw:
                        summarize_user_history(user_key, raw)
                        memory_cache.reset_count(user_key)
                        _record_user_summary_time(user_key)
                        # Avoid hitting rate limits on Pro model
                        time.sleep(10)
                        continue

                    if _can_run_user_summary(user_key):
                        summarize_user_history(user_key, raw)
                        memory_cache.reset_count(user_key)
                        _record_user_summary_time(user_key)
                        time.sleep(10)
                    else:
                        with _pending_lock:
                            _pending_user_summaries.add(user_key)

                except Exception as e:
                    logger.error(f"Worker failure for {user_key}: {e}")

            for group_name in pending_groups:
                try:
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
                        time.sleep(10)
                    else:
                        if raw and len(raw) < 6:
                            summarize_group_history(group_name, raw)
                            group_memory_cache.reset_count(group_name)
                            _record_group_summary_time(group_name)
                except Exception as e:
                    logger.debug(f"Group summary failed {group_name}: {e}")

        except Exception as e:
            logger.debug(f"Background worker loop error: {e}")

        time.sleep(12)


threading.Thread(target=background_summarizer_loop, daemon=True).start()


# ---------------------------
# Mention detection
# ---------------------------
def bot_mentioned_in(text: str) -> bool:
    if not text:
        return False
    target_id = str(config.DISCORD_ID).strip()
    if target_id in text:
        return True
    discord_pattern = r"<@!?" + re.escape(target_id) + r">"
    return re.search(discord_pattern, text) is not None


# ---------------------------
# Core Roast Logic (Gemini)
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
        raw_group, trimmed_group = fetch_group_history(
            group_name,
            limit_messages=config.GROUP_HISTORY_SLICE,
            max_tokens=config.GROUP_HISTORY_TOKEN_LIMIT,
        )
        group_memory = group_memory_cache.get(group_name)
    else:
        raw_group, trimmed_group, group_memory = [], [], ""

    # 2. Build System Context
    sys_parts = []
    user_memory = memory_cache.get(user_key)
    if user_memory:
        sys_parts.append(f"User Profile: {user_memory}")
    if not is_private_env and group_memory:
        sys_parts.append(f"Current Collective Chatter Summary: {group_memory}")

    base_prompt = ROAST_PROMPT if is_private_env else GROUP_ROAST_PROMPT
    if sys_parts:
        final_system_instruction = base_prompt + "\n\nCONTEXT:\n" + "\n".join(sys_parts)
    else:
        final_system_instruction = base_prompt

    # 3. Build Chat History for Gemini (Standardized structure)
    gemini_history = []

    if not is_private_env and trimmed_group:
        last_20 = trimmed_group[-20:] if len(trimmed_group) > 20 else trimmed_group
        for entry in last_20:
            s = entry.get("sender", "unknown")
            c = entry.get("content", "")

            if config.DISCORD_ID:
                c = re.sub(r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@YOU", c)

            # Map roles: if sender is YOU, role=model, else role=user
            if s == "YOU":
                gemini_history.append({"role": "model", "parts": [{"text": c}]})
            else:
                gemini_history.append(
                    {"role": "user", "parts": [{"text": f"{s}: {c}"}]}
                )

    for m in trimmed_user:
        role = m.get("role", "user")
        content = m.get("content", "")
        if config.DISCORD_ID:
            content = re.sub(
                r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@YOU", content
            )

        g_role = "model" if role == "assistant" else "user"
        gemini_history.append({"role": g_role, "parts": [{"text": content}]})

    # Add the final new message
    gemini_history.append({"role": "user", "parts": [{"text": user_message}]})

    # 4. Generate (Use FAST model for roasts)
    base_reply = gemini_chat(
        final_system_instruction, gemini_history, model_name=config.MODEL_FAST
    )

    # 5. Clean & Store
    temp_reply = re.sub(r"^PSI-09\s*:\s*", "", base_reply or "", flags=re.IGNORECASE)
    clean_reply = re.sub(r"\s{3,}", " ", temp_reply).strip()

    if not clean_reply:
        logger.info(f"Empty or failed response for {user_key}.")
        return ""

    user_entry = {
        "role": "assistant",
        "content": clean_reply,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    group_entry = {
        "sender": "YOU",
        "content": clean_reply,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    try:
        history_col.update_one(
            {"_id": user_key}, {"$push": {"messages": user_entry}}, upsert=True
        )
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
# API Routes
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
                r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@YOU", user_message
            )
        if config.BOT_NUMBER:
            user_message = re.sub(
                r"(?<!\S)" + re.escape(config.BOT_NUMBER) + r"(?!\S)",
                "@YOU",
                user_message,
                flags=re.IGNORECASE,
            )

        is_private = group_name in ["DefaultGroup", "Discord_DM"]

        try:
            if is_private:
                store_user_message(group_name, sender_name, user_message)
            else:
                store_group_message(group_name, sender_name, user_message)
                store_user_message(group_name, sender_name, user_message)
        except Exception as e:
            logger.warning(f"Storage failed: {e}")

        user_key = f"{group_name}:{sender_name}"
        if is_private:
            current_count = memory_cache.increment(user_key)
            existing = memory_cache.get(user_key)
            if not existing and current_count == 1:
                summarize_user_history(
                    user_key, [{"content": user_message, "role": "user"}]
                )
            elif current_count >= 10:
                enqueue_user_summary(user_key)
        else:
            if group_memory_cache.increment(group_name) >= 20:
                enqueue_group_summary(group_name)

        is_tagged = bot_mentioned_in(raw_message)

        if is_private or is_tagged:
            clean_input = user_message.strip() or "[mention]"
            reply = get_roast_response(clean_input, group_name, sender_name)
            return jsonify({"reply": reply}), 200
        else:
            return jsonify({"reply": ""}), 200

    except Exception as e:
        logger.exception(f"Error in /psi09: {e}")
        return jsonify({"reply": ""}), 500


def mongo_keepalive():
    while True:
        try:
            mongo_client.admin.command("ping")
        except Exception:
            pass
        time.sleep(180)


threading.Thread(target=mongo_keepalive, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
