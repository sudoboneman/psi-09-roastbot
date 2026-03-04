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

# --- NEW SDK IMPORTS ---
from google import genai
from google.genai import types

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
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY") 
    MODEL: str = "gemini-3-flash-preview" 
    
    MAX_HISTORY_TOKENS: int = 1500
    MAX_HISTORY_MESSAGES: int = 30
    BOT_NUMBER: str = os.getenv("BOT_NUMBER")
    DISCORD_ID: str = os.getenv("DISCORD_ID")
    MEMORY_TTL: int = 300
    EVOLVE_EVERY_N_MESSAGES: int = 20
    GROUP_SUMMARY_EVERY_N: int = 25
    GROUP_HISTORY_SLICE: int = 80
    GROUP_HISTORY_TOKEN_LIMIT: int = 800
    GROUP_HISTORY_MAX_MESSAGES: int = 2000

config = Config()

# ---------------------------
# Initialize Gemini Client
# ---------------------------
try:
    if config.GEMINI_API_KEY:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
    else:
        logger.error("GEMINI_API_KEY is missing. AI features will fail.")
        client = None
except Exception as e:
    logger.error(f"Failed to initialize Gemini Client: {e}")
    client = None

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
# Private Brain Connector (New SDK)
# ---------------------------
def query_private_brain(llm_feed, temperature, max_output_tokens):
    """
    Connects to Google's Gemini API using the new google-genai SDK.
    """
    if not client:
        logger.error("Cannot query brain: Client not initialized.")
        return None

    # 1. Parse System vs. User messages
    system_parts = [msg["content"] for msg in llm_feed if msg["role"] == "system"]
    prompt_parts = [msg["content"] for msg in llm_feed if msg["role"] != "system"]

    system_instruction = "\n\n".join(system_parts)
    final_prompt = "\n\n".join(prompt_parts)

    # 2. Configure generation parameters
    # The new SDK uses types.GenerateContentConfig
    generate_config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=0.95,
        top_k=40,
        system_instruction=system_instruction, # System prompt goes here now
        safety_settings=[
             types.SafetySetting(
                 category="HARM_CATEGORY_HARASSMENT",
                 threshold="BLOCK_NONE"
             ),
             types.SafetySetting(
                 category="HARM_CATEGORY_HATE_SPEECH",
                 threshold="BLOCK_NONE"
             ),
             types.SafetySetting(
                 category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                 threshold="BLOCK_NONE"
             ),
             types.SafetySetting(
                 category="HARM_CATEGORY_DANGEROUS_CONTENT",
                 threshold="BLOCK_NONE"
             ),
        ]
    )

    try:
        logger.info(f"Gemini Input (System len: {len(system_instruction)}, Prompt len: {len(final_prompt)})")

        response = client.models.generate_content(
            model=config.MODEL,
            contents=final_prompt,
            config=generate_config
        )

        reply_text = response.text.strip()
        logger.info(f"Output: {reply_text}")
        return reply_text
        
    except Exception as e:
        logger.error(f"GEMINI CONNECTION ERROR: {e}")
        return None

app = Flask(__name__)
CORS(app)

# ---------------------------
# Token encoding
# ---------------------------
try:
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODING = tiktoken.get_encoding("gpt2")

# ---------------------------
# Memory caches & Concurrency Locks
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

user_locks = defaultdict(threading.Lock)
group_locks = defaultdict(threading.Lock)

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
# History utilities
# ---------------------------
def fetch_history(user_key, limit_messages=None, max_input_tokens=None):
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

    if max_input_tokens:
        trimmed = trim_messages_to_token_budget(raw, max_input_tokens)
        return raw, trimmed

    return raw, raw

def fetch_group_history(group_name, limit_messages=None, max_input_tokens=None):
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
    if max_input_tokens:
        trimmed = []
        total = 0
        for m in reversed(raw):
            txt = f"{m.get('sender','')}: {m.get('content','')}"
            t = tokens_of(txt)
            if total + t > max_input_tokens:
                break
            trimmed.insert(0, m)
            total += t
        return raw, trimmed
    return raw, raw

def fetch_tagged_profiles(group_name, tagged_users, max_targets=3):
    profiles = []
    for u in tagged_users[:max_targets]:
        uid = u.get("id")
        username = u.get("username") or "Unknown"

        if not uid:
            continue  

        memory_key = f"{group_name}:{uid}"
        summary = memory_cache.get(memory_key)

        if summary:
            profiles.append(f"#### BYSTANDER (Numeric ID: {uid} | Username: {username})\n{summary.strip()}")

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
    try:
        history_col.update_one(
            {"_id": user_key}, {"$push": {"messages": entry}}, upsert=True
        )
    except PyMongoError as e:
        logger.warning(f"Failed to store user message for {user_key}: {e}")

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
# Summarization functions
# ---------------------------
def summarize_user_history(user_key, evolve=False):
    raw_history, _ = fetch_history(
        user_key,
        limit_messages=config.MAX_HISTORY_MESSAGES,
    )

    if not raw_history:
        return None

    old_summary = memory_cache.get(user_key)

    if old_summary is None:
        try:
            llm_feed = [
                {"role": "system", "content": f"### FIRST CONTACT PROMPT\n{FIRST_CONTACT_PROMPT}"},
                {"role": "user", "content": f"### CHAT HISTORY\n[User]: {raw_history[-1]['content']}"}
            ]
            summary = query_private_brain(llm_feed, temperature=0.8, max_output_tokens=400)
                
            if summary:
                memory_cache.set(user_key, summary)
                logger.info(f"First-contact profile created for {user_key}")
                return summary
            return None
        except Exception as e:
            logger.error(f"First-contact failed for {user_key}: {e}")
            return None

    if not evolve:
        return old_summary

    recent_user_msgs = [m["content"] for m in raw_history if m.get("role") == "user"][-15:]

    if not recent_user_msgs:
        return old_summary
    
    evolution_prompt = EVOLUTION_PROMPT.format(old_summary=old_summary)
    history_lines = [f"[User]: {msg}" for msg in recent_user_msgs]
    
    llm_feed = [
        {"role": "system", "content": f"### EVOLUTION PROMPT\n{evolution_prompt}"},
        {"role": "user", "content": f"### CHAT HISTORY\n" + "\n".join(history_lines)}
    ]

    try:
        evolved = query_private_brain(llm_feed, temperature=0.8, max_output_tokens=400)
        
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
        summary = f"New group '{group_name}' — Understand group dynamic and log observations."
        group_memory_cache.set(group_name, summary)
        return summary

    old_summary = group_memory_cache.get(group_name) or ""

    recent = []
    for m in raw_history[-25:]:
        sender = m.get("sender") or m.get("username") or m.get("display_name") or "unknown"
        if sender == "PSI-09":
            continue

        content = m.get("content", "")
        if config.DISCORD_ID:
            content = re.sub(
                r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@PSI-09", content
            )
        recent.append(f"[{sender}]: {content}")

    llm_feed = [
        {"role": "system", "content": f"### GROUP SUMMARY PROMPT\n{GROUP_SUMMARY_PROMPT}"},
        {"role": "user", "content": f"### CHAT HISTORY\n" + "\n".join(recent)}
    ]

    try:
        new_summary = query_private_brain(llm_feed, temperature=0.8, max_output_tokens=600)
    except Exception as e:
        logger.warning(f"Group summarization failed for {group_name}: {e}")
        new_summary = old_summary

    if new_summary and new_summary != old_summary:
        try:
            group_memory_cache.set(group_name, new_summary)
        except Exception:
            pass

    return new_summary or old_summary

def bot_mentioned_in(text: str) -> bool:
    if not text or not config.DISCORD_ID:
        return False

    discord_pattern = r"<@!?" + re.escape(str(config.DISCORD_ID)) + r">"
    return re.search(discord_pattern, text) is not None

def get_roast_response(group_name, sender_id, username, tagged_users=None):
    tagged_users = tagged_users or []
    user_key = f"{group_name}:{sender_id}"
    is_private_env = group_name in ["DefaultGroup", "Discord_DM"]

    _, trimmed_user = fetch_history(
        user_key,
        limit_messages=config.MAX_HISTORY_MESSAGES,
        max_input_tokens=config.MAX_HISTORY_TOKENS,
    )

    if not is_private_env:
        _, trimmed_group = fetch_group_history(
            group_name,
            limit_messages=config.GROUP_HISTORY_SLICE,
            max_input_tokens=config.GROUP_HISTORY_TOKEN_LIMIT,
        )
        group_memory = group_memory_cache.get(group_name)
    else:
        trimmed_group, group_memory = [], None

    llm_feed = []

    system_content = ROAST_PROMPT if is_private_env else GROUP_ROAST_PROMPT
    llm_feed.append({
        "role": "system", 
        "content": f"### ROAST PROMPT\n{system_content}"
    })

    user_memory = memory_cache.get(user_key)
    if user_memory:
        llm_feed.append({
            "role": "system", 
            "content": f"### TARGET PROFILE (Numeric ID: {sender_id} | Username: {username})\n{user_memory.strip()}"
        })

    if not is_private_env and group_memory:
        llm_feed.append({
            "role": "system", 
            "content": f"### GROUP DYNAMIC SUMMARY\n{group_memory.strip()}"
        })

    tagged_profiles = fetch_tagged_profiles(group_name, tagged_users)
    if tagged_profiles:
        llm_feed.append({
            "role": "system", 
            "content": f"### BYSTANDER PROFILES\n" + "\n\n".join(tagged_profiles)
        })

    history_lines = []
    
    if is_private_env:
        if trimmed_user:
            for m in trimmed_user:
                role = m.get("role", "user")
                content = m.get("content", "").strip()
                if config.DISCORD_ID:
                    content = re.sub(r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@PSI-09", content)
                if content:
                    prefix = "PSI-09" if role == "assistant" else username
                    history_lines.append(f"[{prefix}]: {content}")
    else:
        if trimmed_group:
            last_20 = trimmed_group[-20:] if len(trimmed_group) > 20 else trimmed_group
            for entry in last_20:
                s = entry.get("sender") or entry.get("username") or entry.get("display_name") or "unknown"
                c = entry.get("content", "").strip()
                if config.DISCORD_ID:
                    c = re.sub(r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@PSI-09", c)
                if c:
                    history_lines.append(f"[{s}]: {c}")

    history_text = "\n".join(history_lines) if history_lines else "[No recent history]"
    
    llm_feed.append({
        "role": "user", 
        "content": f"### CHAT HISTORY\n{history_text}"
    })

    try:
        base_reply = query_private_brain(
            llm_feed=llm_feed,
            temperature=0.9, 
            max_output_tokens=1000
        )
    except Exception as e:
        logger.error(f"AI Error: {e}")
        base_reply = ""

    temp_reply = re.sub(r"^PSI-09\s*:\s*", "", base_reply or "", flags=re.IGNORECASE)
    temp_reply = re.sub(r"\[.*?\]:.*", "", temp_reply, flags=re.DOTALL) 
    clean_reply = re.sub(r"\s{3,}", " ", temp_reply).strip()

    if not clean_reply:
        logger.info(f"Empty or failed response for {user_key}. Skipping storage.")
        return ""

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
    except Exception as e:
        logger.warning(f"Reply storage failed: {e}")

    return clean_reply

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/psi09", methods=["POST"])
def psi09():
    try:
        data = request.get_json(force=True)
        raw_message = data.get("message", "")
        sender_id = data.get("sender_id")
        username = data.get("username")

        if username and "WEBHOOK" in username.upper():
            logger.warning(f"Blocked malicious webhook loop injection from {username}")
            return jsonify({"reply": ""}), 200
        
        if username == "PSI09_STATUS" and raw_message == "status":
            logger.info("Generating fresh WhatsApp status roast...")
            llm_feed = [
                {"role": "system", "content": f"### STATUS ROAST PROMPT\n{STATUS_ROAST_PROMPT}"},
                {"role": "user", "content": "### CHAT HISTORY\n[User]: Generate a new cynical status update for the humans."}
            ]
            reply = query_private_brain(llm_feed, temperature=1.0, max_output_tokens=400)
            clean_reply = re.sub(r"^PSI-09\s*:\s*", "", reply or "", flags=re.IGNORECASE).strip()
            return jsonify({"reply": clean_reply}), 200

        display_name = data.get("display_name") or username
        group_name = data.get("group_name") or "DefaultGroup"
        tagged_users = data.get("tagged_users", [])

        if not username or not sender_id or not raw_message:
            return jsonify({"reply": ""}), 200

        user_message = raw_message
        if config.DISCORD_ID:
            user_message = re.sub(
                r"<@!?" + re.escape(str(config.DISCORD_ID)) + r">",
                "@PSI-09",
                user_message,
            )

        is_private = group_name in ["DefaultGroup", "Discord_DM"]

        if is_private:
            store_user_message(group_name, sender_id, username, display_name, user_message)
        else:
            store_group_message(group_name, sender_id, username, display_name, user_message)
            store_user_message(group_name, sender_id, username, display_name, user_message)

        user_key = f"{group_name}:{sender_id}"

        with user_locks[user_key]:
            msg_count = memory_cache.increment(user_key)
            current_user_memory = memory_cache.get(user_key)
            
            if current_user_memory is None:
                logger.info(f"Initiating First Contact for {user_key}")
                summarize_user_history(user_key, evolve=False)
                memory_cache.reset_count(user_key)
                
            elif msg_count >= config.EVOLVE_EVERY_N_MESSAGES:
                logger.info(f"Initiating Evolution for {user_key}")
                summarize_user_history(user_key, evolve=True)
                memory_cache.reset_count(user_key)

        if not is_private:
            with group_locks[group_name]:
                group_msg_count = group_memory_cache.increment(group_name)
                
                if group_msg_count >= config.GROUP_SUMMARY_EVERY_N:
                    logger.info(f"Initiating Group Summary for {group_name}")
                    raw_group_hist, _ = fetch_group_history(
                        group_name, 
                        limit_messages=config.GROUP_HISTORY_SLICE, 
                        max_input_tokens=config.GROUP_HISTORY_TOKEN_LIMIT
                    )
                    summarize_group_history(group_name, raw_group_hist)
                    group_memory_cache.reset_count(group_name)

        if is_private or bot_mentioned_in(raw_message):
            reply = get_roast_response(group_name, sender_id, username, tagged_users)
            return jsonify({"reply": reply}), 200

        return jsonify({"reply": ""}), 200

    except Exception as e:
        logger.exception(f"/psi09 failure: {e}")
        return jsonify({"reply": ""}), 500

def mongo_keepalive():
    while True:
        try:
            mongo_client.admin.command("ping")
        except Exception as e:
            logger.warning(f"Mongo keepalive failed: {e}")
        time.sleep(180)

threading.Thread(target=mongo_keepalive, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860)) 
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)