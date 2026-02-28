# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import os
import tiktoken
import json
import re
import threading
import time
import logging
import sys
import requests
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from dataclasses import dataclass
import certifi
from prompts import (
    ROAST_PROMPT, 
    GROUP_ROAST_PROMPT, 
    FIRST_CONTACT_PROMPT, 
    EVOLUTION_PROMPT, 
    GROUP_SUMMARY_PROMPT,
    STATUS_ROAST_PROMPT
)

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
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
    DISCORD_ID: str = os.getenv("DISCORD_ID")
    
    # Keeping your existing memory settings
    MAX_HISTORY_TOKENS: int = 1500
    MAX_HISTORY_MESSAGES: int = 30
    MEMORY_TTL: int = 300
    EVOLVE_EVERY_N_MESSAGES: int = 20
    GROUP_SUMMARY_EVERY_N: int = 25
    GROUP_HISTORY_SLICE: int = 80
    GROUP_HISTORY_TOKEN_LIMIT: int = 800
    GROUP_HISTORY_MAX_MESSAGES: int = 2000

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
# Brain Connector (OpenRouter FREE API)
# ---------------------------
def query_private_brain(messages, temperature, max_output_tokens):
    """
    Sends the complex chat history to OpenRouter's free enterprise models.
    """
    if not config.OPENROUTER_API_KEY:
        logger.error("Missing OPENROUTER_API_KEY! Add it to your Space Secrets.")
        return None

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://psi-09.engine", 
        "X-Title": "PSI-09"
    }
    
    payload = {
        # The ultimate free smart model. (Swap to "gryphe/mythomax-l2-13b:free" if you want 0% filters)
        "model": "meta-llama/llama-3.1-8b-instruct:free", 
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "top_p": 0.9
    }

    logger.info("Sending payload to OpenRouter...")

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", 
            json=payload, 
            headers=headers, 
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        reply_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        logger.info(f"OpenRouter Output: {reply_text}")
        return reply_text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"EXTERNAL BRAIN CONNECTION ERROR: {e}")
        return None

app = Flask(__name__)
CORS(app)

# ---------------------------
# Token fallback (tiktoken)
# ---------------------------
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
            summary = None

        with self.lock:
            self.cache[key] = summary
            self.expiry[key] = now + self.ttl
        return summary

    def set(self, key, value):
        now = datetime.now(UTC)
        try:
            memory_col.update_one({"_id": key}, {"$set": {"summary": value}}, upsert=True)
        except PyMongoError:
            pass

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
        except PyMongoError:
            summary = None

        with self.lock:
            self.cache[key] = summary
            self.expiry[key] = now + self.ttl
        return summary

    def set(self, key, value):
        now = datetime.now(UTC)
        try:
            group_memory_col.update_one({"_id": key}, {"$set": {"summary": value}}, upsert=True)
        except PyMongoError:
            pass

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

_pending_user_summaries = set()
_pending_group_summaries = set()
_pending_lock = threading.Lock()
_last_user_summary_time = {}
_last_group_summary_time = {}
SUMMARY_COOLDOWN_SECONDS = 60  

# ---------------------------
# Utilities
# ---------------------------
def tokens_of(text: str) -> int:
    if not text: return 0
    try: return len(ENCODING.encode(text))
    except Exception: return len(text.split())

def trim_messages_to_token_budget(messages, max_tokens):
    total = 0
    trimmed = []
    for m in reversed(messages):
        c = m.get("content", "")
        t = tokens_of(c)
        if total + t > max_tokens: break
        trimmed.insert(0, m)
        total += t
    return trimmed

def fetch_history(user_key, limit_messages=None, max_input_tokens=None):
    limit_messages = limit_messages or config.MAX_HISTORY_MESSAGES
    try:
        doc = history_col.find_one({"_id": user_key}, {"messages": {"$slice": -limit_messages}})
    except PyMongoError:
        return [], []

    if not doc or "messages" not in doc: return [], []
    raw = doc["messages"]

    if max_input_tokens:
        trimmed = trim_messages_to_token_budget(raw, max_input_tokens)
        return raw, trimmed
    return raw, raw

def fetch_group_history(group_name, limit_messages=None, max_input_tokens=None):
    limit_messages = limit_messages or config.GROUP_HISTORY_SLICE
    try:
        doc = group_history_col.find_one({"_id": group_name}, {"messages": {"$slice": -limit_messages}})
    except PyMongoError:
        return [], []

    if not doc or "messages" not in doc: return [], []
    raw = doc["messages"]
    
    if max_input_tokens:
        trimmed = []
        total = 0
        for m in reversed(raw):
            txt = f"{m.get('sender','')}: {m.get('content','')}"
            t = tokens_of(txt)
            if total + t > max_input_tokens: break
            trimmed.insert(0, m)
            total += t
        return raw, trimmed
    return raw, raw

def fetch_tagged_profiles(group_name, tagged_users, max_targets=3):
    profiles = []
    for u in tagged_users[:max_targets]:
        uid = u.get("id")
        username = u.get("username") or "Unknown"
        if not uid: continue  

        memory_key = f"{group_name}:{uid}"
        summary = memory_cache.get(memory_key)

        # XML Formatting for clean data feeds
        if summary:
            profiles.append(f'<profile username="{username}" discord_id="{uid}">\n{summary}\n</profile>')

    return profiles

def store_user_message(group_name, sender_id, username, display_name, message):
    user_key = f"{group_name}:{sender_id}"
    entry = {
        "role": "user",
        "user_id": sender_id,
        "username": username,
        "display_name": display_name,
        "content": message,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try: history_col.update_one({"_id": user_key}, {"$push": {"messages": entry}}, upsert=True)
    except PyMongoError: pass

def store_group_message(group_name, sender_id, username, display_name, message):
    entry = {
        "sender_id": sender_id,
        "username": username,
        "display_name": display_name,
        "content": message,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        group_history_col.update_one(
            {"_id": group_name},
            {"$push": {"messages": {"$each": [entry], "$slice": -config.GROUP_HISTORY_MAX_MESSAGES}}},
            upsert=True,
        )
    except PyMongoError: pass

# ---------------------------
# Summarization functions
# ---------------------------
def summarize_user_history(user_key, evolve=False):
    raw_history, _ = fetch_history(user_key, limit_messages=config.MAX_HISTORY_MESSAGES)
    if not raw_history: return None
    old_summary = memory_cache.get(user_key)

    if old_summary is None:
        try:
            prompt_messages = [
                {"role": "system", "content": FIRST_CONTACT_PROMPT},
                {"role": "user", "content": raw_history[-1]["content"]},
            ]
            summary = query_private_brain(prompt_messages, temperature=0.9, max_output_tokens=200)
            if summary:
                memory_cache.set(user_key, summary)
                return summary
            return None
        except Exception: return None

    if not evolve: return old_summary

    recent_user_msgs = [m["content"] for m in raw_history if m.get("role") == "user"][-15:]
    if not recent_user_msgs: return old_summary
    
    evolution_prompt = EVOLUTION_PROMPT.format(old_summary=old_summary)
    messages = [{"role": "system", "content": evolution_prompt}]
    for msg in recent_user_msgs:
        messages.append({"role": "user", "content": msg})

    try:
        evolved = query_private_brain(messages, temperature=0.8, max_output_tokens=200)
        if evolved:
            memory_cache.set(user_key, evolved)
            return evolved
        return old_summary
    except Exception: return old_summary

def summarize_group_history(group_name, raw_history):
    if not raw_history: return group_memory_cache.get(group_name)

    if len(raw_history) < 6:
        summary = f"New group '{group_name}' — Understand group dynamic and log observations."
        group_memory_cache.set(group_name, summary)
        return summary

    old_summary = group_memory_cache.get(group_name) or ""
    recent = []
    
    for m in raw_history[-25:]:
        sender = m.get("username") or "unknown"
        content = m.get("content", "")
        if config.DISCORD_ID:
            content = re.sub(r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@PSI-09", content)
        recent.append(f"(Sent by {sender}): {content}")

    prompt = [{"role": "system", "content": GROUP_SUMMARY_PROMPT}] + [
        {"role": "user", "content": t} for t in recent
    ]

    try:
        new_summary = query_private_brain(prompt, temperature=0.8, max_output_tokens=200)
    except Exception: new_summary = old_summary

    if new_summary and new_summary != old_summary:
        try: group_memory_cache.set(group_name, new_summary)
        except Exception: pass

    return new_summary or old_summary

# ---------------------------
# Background summarizer worker
# ---------------------------
def enqueue_user_summary(user_key):
    with _pending_lock: _pending_user_summaries.add(user_key)

def enqueue_group_summary(group_name):
    with _pending_lock: _pending_group_summaries.add(group_name)

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

def background_group_summarizer_loop():
    while True:
        try:
            to_requeue = set()
            with _pending_lock:
                groups = list(_pending_group_summaries)
                _pending_group_summaries.clear()

            for group_name in groups:
                if not _can_run_group_summary(group_name):
                    to_requeue.add(group_name)
                    continue

                raw, _ = fetch_group_history(group_name, limit_messages=config.GROUP_HISTORY_SLICE, max_input_tokens=config.GROUP_HISTORY_TOKEN_LIMIT)
                if not raw: continue

                try:
                    summary = summarize_group_history(group_name, raw)
                    if summary:
                        group_memory_cache.reset_count(group_name)
                        _record_group_summary_time(group_name)
                    else: to_requeue.add(group_name)
                except Exception: to_requeue.add(group_name)

            if to_requeue:
                with _pending_lock: _pending_group_summaries.update(to_requeue)
        except Exception: pass
        time.sleep(12)

def background_user_summarizer_loop():
    while True:
        try:
            to_requeue = set()
            with _pending_lock:
                users = list(_pending_user_summaries)
                _pending_user_summaries.clear()

            for user_key in users:
                if not _can_run_user_summary(user_key):
                    to_requeue.add(user_key)
                    continue

                count = memory_cache.msg_count.get(user_key, 0)
                if count < config.EVOLVE_EVERY_N_MESSAGES:
                    to_requeue.add(user_key)
                    continue

                try:
                    summary = summarize_user_history(user_key, evolve=True)
                    if summary:
                        memory_cache.reset_count(user_key)
                        _record_user_summary_time(user_key)
                    else: to_requeue.add(user_key)
                except Exception: to_requeue.add(user_key)

            if to_requeue:
                with _pending_lock: _pending_user_summaries.update(to_requeue)
        except Exception: pass
        time.sleep(12)

threading.Thread(target=background_group_summarizer_loop, daemon=True).start()
threading.Thread(target=background_user_summarizer_loop, daemon=True).start()

def bot_mentioned_in(text: str) -> bool:
    if not text or not config.DISCORD_ID: return False
    discord_pattern = r"<@!?" + re.escape(str(config.DISCORD_ID)) + r">"
    return re.search(discord_pattern, text) is not None

# ---------------------------
# Core roast generation - XML Formatted Feed
# ---------------------------
def get_roast_response(user_message, group_name, sender_id, username, tagged_users=None):
    tagged_users = tagged_users or []
    user_key = f"{group_name}:{sender_id}"
    is_private_env = group_name in ["DefaultGroup", "Discord_DM"]

    raw_user, trimmed_user = fetch_history(user_key, limit_messages=config.MAX_HISTORY_MESSAGES, max_input_tokens=config.MAX_HISTORY_TOKENS)

    if not is_private_env:
        raw_group, trimmed_group = fetch_group_history(group_name, limit_messages=config.GROUP_HISTORY_SLICE, max_input_tokens=config.GROUP_HISTORY_TOKEN_LIMIT)
        group_memory = group_memory_cache.get(group_name)
    else:
        raw_group, trimmed_group, group_memory = [], [], ""

    # ==========================================
    # 1. Build Strict XML Context Database
    # ==========================================
    sys_parts = []
    user_memory = memory_cache.get(user_key)

    if user_memory:
        sys_parts.append(f"<target_profile username=\"{username}\">\n{user_memory}\n</target_profile>")

    if not is_private_env and group_memory:
        sys_parts.append(f"<group_dynamic>\n{group_memory}\n</group_dynamic>")

    tagged_profiles = fetch_tagged_profiles(group_name, tagged_users)
    if tagged_profiles:
        sys_parts.append("<bystander_profiles>\n" + "\n".join(tagged_profiles) + "\n</bystander_profiles>")

    system_memory_text = "\n".join(sys_parts) if sys_parts else ""
    system_prompt = ROAST_PROMPT if is_private_env else GROUP_ROAST_PROMPT
    
    final_system_content = system_prompt
    if system_memory_text:
        # XML Injection guarantees the AI won't confuse context with instructions
        final_system_content += "\n\n<database_inject>\n" + system_memory_text + "\n</database_inject>\n\nCRITICAL RULE: DO NOT acknowledge or quote the database tags. Simply use the information against the user."

    messages = [{"role": "system", "content": final_system_content}]

    # ==========================================
    # 2. Build the Native ChatML History Feed
    # ==========================================
    if is_private_env:
        if trimmed_user:
            for m in trimmed_user:
                role = m.get("role", "user")
                content = m.get("content", "").strip()
                if config.DISCORD_ID:
                    content = re.sub(r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@PSI-09", content)
                if content:
                    # Clean User framing
                    if role == "user":
                        messages.append({"role": "user", "content": f"(Sent by {username}): {content}"})
                    else:
                        messages.append({"role": "assistant", "content": content})
    else:
        if trimmed_group:
            last_20 = trimmed_group[-20:] if len(trimmed_group) > 20 else trimmed_group
            for entry in last_20:
                s = entry.get("sender") or entry.get("username") or entry.get("display_name") or "unknown"
                c = entry.get("content", "").strip()
                if config.DISCORD_ID:
                    c = re.sub(r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@PSI-09", c)
                if not c: continue

                if s == "PSI-09":
                    messages.append({"role": "assistant", "content": c})
                else:
                    messages.append({"role": "user", "content": f"(Sent by {s}): {c}"})

    # ==========================================
    # 3. Call the Brain
    # ==========================================
    try:
        base_reply = query_private_brain(messages=messages, temperature=0.9, max_output_tokens=200)
    except Exception as e:
        logger.error(f"AI Error: {e}")
        base_reply = ""

    # Hard strip any hallucinated script prefixes
    temp_reply = re.sub(r"^(?:PSI-09|<.*?>|\[.*?\]|\(.*?\)|\*.*?\*)\s*:\s*", "", base_reply or "", flags=re.IGNORECASE)
    clean_reply = re.sub(r"\s{2,}", " ", temp_reply).strip()

    if not clean_reply: return ""

    # ==========================================
    # 4. Store the Reply
    # ==========================================
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
        history_col.update_one({"_id": user_key}, {"$push": {"messages": user_entry}}, upsert=True)
        if not is_private_env:
            group_history_col.update_one(
                {"_id": group_name},
                {"$push": {"messages": {"$each": [group_entry], "$slice": -config.GROUP_HISTORY_MAX_MESSAGES}}},
                upsert=True,
            )
    except Exception: pass

    return clean_reply

# ---------------------------
# Flask routes
# ---------------------------
@app.route("/", methods=["GET"])
def health(): return jsonify({"status": "ok"}), 200

@app.route("/psi09", methods=["POST"])
def psi09():
    try:
        data = request.get_json(force=True)
        raw_message = data.get("message", "")
        sender_id = data.get("sender_id")
        username = data.get("username")

        if username and "WEBHOOK" in username.upper():
            return jsonify({"reply": ""}), 200
        
        if username == "PSI09_STATUS" and raw_message == "status":
            status_messages = [
                {"role": "system", "content": STATUS_ROAST_PROMPT},
                {"role": "user", "content": "Generate a new cynical status update for the humans."}
            ]
            reply = query_private_brain(messages=status_messages, temperature=1.0, max_output_tokens=150)
            clean_reply = re.sub(r"^PSI-09\s*:\s*", "", reply or "", flags=re.IGNORECASE).strip()
            return jsonify({"reply": clean_reply}), 200

        display_name = data.get("display_name") or username
        group_name = data.get("group_name") or "DefaultGroup"
        tagged_users = data.get("tagged_users", [])

        if not username or not sender_id or not raw_message:
            return jsonify({"reply": ""}), 200

        user_message = raw_message
        if config.DISCORD_ID:
            user_message = re.sub(r"<@!?" + re.escape(str(config.DISCORD_ID)) + r">", "@PSI-09", user_message)

        is_private = group_name in ["DefaultGroup", "Discord_DM"]

        if is_private: store_user_message(group_name, sender_id, username, display_name, user_message)
        else:
            store_group_message(group_name, sender_id, username, display_name, user_message)
            store_user_message(group_name, sender_id, username, display_name, user_message)

        user_key = f"{group_name}:{sender_id}"
        enqueue_user_summary(user_key)
        memory_cache.increment(user_key)

        if memory_cache.get(user_key) is None:
            try:
                summary = summarize_user_history(user_key)
                if summary:
                    memory_cache.reset_count(user_key)
                    _record_user_summary_time(user_key)
            except Exception: pass

        if not is_private:
            if group_memory_cache.increment(group_name) >= config.GROUP_SUMMARY_EVERY_N:
                enqueue_group_summary(group_name)

        if is_private or bot_mentioned_in(raw_message):
            reply = get_roast_response(user_message.strip() or "[mention]", group_name, sender_id, username, tagged_users)
            return jsonify({"reply": reply}), 200

        return jsonify({"reply": ""}), 200
    except Exception: return jsonify({"reply": ""}), 500

def mongo_keepalive():
    while True:
        try: mongo_client.admin.command("ping")
        except Exception: pass
        time.sleep(180)

threading.Thread(target=mongo_keepalive, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860)) 
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)