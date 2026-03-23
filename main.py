# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import os
import re
import threading
import time
import logging
import sys
import random

from transformers import AutoTokenizer
from groq import Groq

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
    GLOBAL_FIRST_CONTACT_PROMPT,
    GLOBAL_EVOLUTION_PROMPT      
)

# Environment & Logging
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
UTC = timezone.utc

# Config
@dataclass
class Config:
    MONGO_URI: str = os.getenv("MONGO_URI")
    GROQ_API_KEY_1: str = os.getenv("GROQ_API_KEY_1") # First Contact & Roasts
    GROQ_API_KEY_2: str = os.getenv("GROQ_API_KEY_2") # Evolution & Group Summaries
    
    # --- PERSISTENT MODEL CYCLE ---
    # The bot will stay on index 0 until it dies, then permanently move to index 1, and so on.
    MODELS: list = __import__("dataclasses").field(default_factory=lambda: [
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ])
    
    # --- TIGHTENED FOR MAXIMUM THROUGHPUT ---
    BOT_NUMBER: str = os.getenv("BOT_NUMBER")
    DISCORD_ID: str = os.getenv("DISCORD_ID")
    MEMORY_TTL: int = 500
    
    # DATABASE CEILINGS (Keep these high for memory)
    GROUP_HISTORY_MAX_MESSAGES: int = 50000 
    GROUP_HISTORY_SLICE: int = 100 # Fast database read
    
    # LLM PAYLOAD CEILINGS (Shrunk to maximize API calls/min)
    MAX_HISTORY_MESSAGES: int = 15 # Down from 30
    MAX_HISTORY_TOKENS: int = 400 # Tightly caps the user's personal history
    GROUP_HISTORY_TOKEN_LIMIT: int = 800 # Tightly caps the group's history
    
    # THE PACING ENGINE (Tuned for 6b6t Anarchy traffic)
    EVOLVE_EVERY_N_MESSAGES: int = 50 # Evolve active users frequently
    GROUP_SUMMARY_EVERY_N: int = 800 # Rapidly update the group dynamic

config = Config()

# --- PERSISTENT MODEL TRACKER ---
model_lock = threading.Lock()
active_model_index = 0

# Initialize Groq Clients
client_1 = None
if config.GROQ_API_KEY_1:
    client_1 = Groq(api_key=config.GROQ_API_KEY_1)

client_2 = None
if config.GROQ_API_KEY_2:
    client_2 = Groq(api_key=config.GROQ_API_KEY_2)
else:
    logger.warning("No second API key found. Falling back to Key 1 for all tasks.")
    client_2 = client_1

# MongoDB
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
global_history_col = db["global_history"]
global_memory_col = db["global_memory"]

def query_private_brain(llm_feed, temperature, max_output_tokens, task_type="roast", max_retries=4):
    """
    Connects to API. If a rate limit is hit, permanently switches the active model 
    for all future requests until the next limit is hit.
    """
    global active_model_index
    
    active_client = client_1 if task_type in ["first_contact"] else client_2

    if not active_client:
        logger.error(f"Cannot query brain: Client for '{task_type}' not initialized.")
        return None 

    for attempt in range(max_retries):
        # Always fetch whatever model is currently globally active
        current_model = config.MODELS[active_model_index]
        
        try:
            response = active_client.chat.completions.create(
                model=current_model,
                messages=llm_feed,
                temperature=temperature,
                max_completion_tokens=max_output_tokens,
                top_p=1
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = str(e).lower()
            if attempt == max_retries - 1:
                logger.error(f"GROQ FATAL ERROR (After {max_retries} attempts): {e}")
                return None
            
            # Catch Rate Limits and Token Exhaustion
            if "429" in error_msg or "rate limit" in error_msg or "token" in error_msg or "50" in error_msg:
                with model_lock:
                    # SAFETY CHECK: Make sure another simultaneous thread hasn't already rotated it
                    if config.MODELS[active_model_index] == current_model:
                        
                        # Move to the next model in the list (and loop back to 0 if at the end)
                        active_model_index = (active_model_index + 1) % len(config.MODELS)
                        new_model = config.MODELS[active_model_index]
                        
                        logger.warning(f"[{task_type}] {current_model} exhausted. PERSISTENT SWITCH to {new_model}.")
                    else:
                        # Another thread already handled the rotation, just grab the new model
                        new_model = config.MODELS[active_model_index]
                        logger.info(f"[{task_type}] Model already rotated by another thread to {new_model}.")

                # Ultra-fast pause so Groq registers the new model properly
                time.sleep(random.uniform(0.2, 0.5))
            else:
                logger.error(f"Non-retriable GROQ ERROR: {e}")
                return None

app = Flask(__name__)
CORS(app)

# 1. Load Kimi's Tokenizer (If Kimi is in the rotation)
KIMI_ENCODING = None
if any("kimi" in m.lower() for m in config.MODELS):
    try:
        KIMI_ENCODING = AutoTokenizer.from_pretrained("moonshotai/Kimi-K2-Instruct", trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load Kimi tokenizer: {e}")

# 2. Load Exact Llama Tokenizer (If Llama is in the rotation)
LLAMA_ENCODING = None
if any("llama" in m.lower() for m in config.MODELS):
    try:
        # Using the ungated Unsloth repo to bypass the 401 Unauthorized error
        LLAMA_ENCODING = AutoTokenizer.from_pretrained("unsloth/Llama-4-Scout-17B-16E-Instruct")
    except Exception as e:
        logger.warning(f"Failed to load Llama tokenizer: {e}")

# 3. The precise routing engine
def tokens_of(text: str) -> int:
    if not text:
        return 0
        
    global active_model_index
    current_active_model = config.MODELS[active_model_index].lower()
    
    is_kimi = "kimi" in current_active_model
    is_llama = "llama" in current_active_model

    if is_kimi and KIMI_ENCODING:
        return len(KIMI_ENCODING.encode(text))
        
    if is_llama and LLAMA_ENCODING:
        return len(LLAMA_ENCODING.encode(text))
        
    return int(len(text.split()) * 1.5)

# --- UNIFIED CACHE CLASS (Saves ~90 lines) ---
class MongoCache:
    def __init__(self, collection, ttl_seconds):
        self.collection = collection
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
            doc = self.collection.find_one({"_id": key})
            summary = doc.get("summary") if doc and doc.get("summary") else None
        except PyMongoError as e:
            logger.warning(f"Failed to load memory for {key}: {e}")
            summary = None
        with self.lock:
            self.cache[key] = summary
            self.expiry[key] = now + self.ttl
        return summary

    def set(self, key, value):
        now = datetime.now(UTC)
        try:
            self.collection.update_one({"_id": key}, {"$set": {"summary": value}}, upsert=True)
        except PyMongoError as e:
            logger.warning(f"Failed to persist memory for {key}: {e}")
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

memory_cache = MongoCache(memory_col, config.MEMORY_TTL)
group_memory_cache = MongoCache(group_memory_col, config.MEMORY_TTL)
global_memory_cache = MongoCache(global_memory_col, config.MEMORY_TTL)

user_locks = defaultdict(threading.Lock)
group_locks = defaultdict(threading.Lock)
global_locks = defaultdict(threading.Lock)

# --- UNIFIED UTILITIES ---
def trim_messages_to_token_budget(messages, max_tokens):
    """FIXED SOFT FAULT 2: Now measures the exact formatted string the LLM sees"""
    total = 0
    trimmed = []
    for m in reversed(messages):
        sender = m.get("sender") or m.get("username") or m.get("display_name") or m.get("role") or "User"
        formatted_text = f"[{sender}]: {m.get('content', '')}"
        t = tokens_of(formatted_text)
        
        if total + t > max_tokens:
            break
        trimmed.insert(0, m)
        total += t
    return trimmed

def fetch_history(collection, doc_id, limit_messages, max_input_tokens=None):
    """UNIFIED FETCHER: Replaces the 3 redundant fetchers (Saves ~30 lines)"""
    try:
        doc = collection.find_one(
            {"_id": doc_id}, {"messages": {"$slice": -limit_messages}}
        )
    except PyMongoError as e:
        logger.warning(f"Failed to fetch history for {doc_id}: {e}")
        return [], []

    if not doc or "messages" not in doc:
        return [], []

    raw = doc["messages"]
    if max_input_tokens:
        trimmed = trim_messages_to_token_budget(raw, max_input_tokens)
        return raw, trimmed
    return raw, raw

def fetch_tagged_profiles(group_name, tagged_users, max_targets=3):
    profiles = []
    for u in tagged_users[:max_targets]:
        uid = u.get("id")
        username = u.get("username")
        if not username:
            continue  
        memory_key = f"Global:{username}"
        summary = global_memory_cache.get(memory_key)
        if summary:
            # Replaced markdown header with an XML block
            profiles.append(f'<bystander username="{username}" numeric_id="{uid}">\n{summary.strip()}\n</bystander>')
    return profiles

# --- DATABASE STORAGE ---
def store_user_message(platform, group_name, sender_id, username, display_name, message):
    user_key = f"{group_name}:{username}"
    global_key = f"Global:{username}"
    
    local_entry = {
        "role": "user",
        "user_id": sender_id,
        "username": username,
        "display_name": display_name,
        "platform": platform, 
        "content": message,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    
    global_entry = local_entry.copy()
    global_entry["content"] = f"[Sent via {platform} - {group_name}] {message}"
    
    try:
        # FIXED SOFT FAULT 1: Applied $slice to stop infinite DB growth
        history_col.update_one(
            {"_id": user_key}, 
            {"$push": {"messages": {"$each": [local_entry], "$slice": -config.GROUP_HISTORY_MAX_MESSAGES}}}, 
            upsert=True
        )
        global_history_col.update_one(
            {"_id": global_key}, 
            {"$push": {"messages": {"$each": [global_entry], "$slice": -config.GROUP_HISTORY_MAX_MESSAGES}}}, 
            upsert=True
        )
    except PyMongoError as e:
        logger.warning(f"Failed to store user message: {e}")

def store_group_message(platform, group_name, sender_id, username, display_name, message):
    entry = {
        "sender_id": sender_id,
        "username": username,
        "display_name": display_name,
        "platform": platform,
        "content": message,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        group_history_col.update_one(
            {"_id": group_name},
            {"$push": {"messages": {"$each": [entry], "$slice": -config.GROUP_HISTORY_MAX_MESSAGES}}},
            upsert=True,
        )
    except PyMongoError as e:
        logger.warning(f"Failed to store group message for {group_name}: {e}")

def bot_mentioned_in(text: str) -> bool:
    if not text:
        return False
    if re.search(r"@psi-09", text, flags=re.IGNORECASE):
        return True
    if config.DISCORD_ID:
        discord_pattern = r"<@!?" + re.escape(str(config.DISCORD_ID)) + r">"
        if re.search(discord_pattern, text):
            return True
    return False

# --- SUMMARIZATION ENGINES ---
def summarize_user_history(user_key, evolve=False):
    _, trimmed_history = fetch_history(
        history_col, user_key, config.MAX_HISTORY_MESSAGES, config.MAX_HISTORY_TOKENS 
    )

    if not trimmed_history:
        return None

    old_summary = memory_cache.get(user_key)
    history_lines = [f"[User]: {m['content']}" for m in trimmed_history if m.get("role") == "user"]
    
    if not history_lines:
        return old_summary

    if old_summary is None:
        sys_prompt = FIRST_CONTACT_PROMPT
        user_content = f"<chat_history>\n[User]: {trimmed_history[-1]['content']}\n</chat_history>"
    else:
        if not evolve:
            return old_summary
        sys_prompt = EVOLUTION_PROMPT.format(old_summary=old_summary)
        user_content = f"<chat_history>\n" + "\n".join(history_lines) + "\n</chat_history>"

    llm_feed = [
        {"role": "system", "content": f"<profile_engine_prompt>\n{sys_prompt}\n</profile_engine_prompt>"},
        {"role": "user", "content": user_content}
    ]

    try:
        current_task = "evolution" if old_summary else "first_contact"
        new_summary = query_private_brain(llm_feed, temperature=0.8, max_output_tokens=300, task_type=current_task)
        
        if new_summary:
            memory_cache.set(user_key, new_summary)
            logger.info(f"Local profile updated for {user_key} ({current_task})")
            return new_summary
        return old_summary 
        
    except Exception as e:
        logger.warning(f"Local profile task '{current_task}' failed for {user_key}: {e}")
        return old_summary
    
def summarize_group_history(group_name):
    _, trimmed_history = fetch_history(
        group_history_col, group_name, config.GROUP_HISTORY_SLICE, config.GROUP_HISTORY_TOKEN_LIMIT
    )

    if not trimmed_history:
        return group_memory_cache.get(group_name)

    if len(trimmed_history) < 6:
        summary = f"New group '{group_name}' — Understand group dynamic and log observations."
        group_memory_cache.set(group_name, summary)
        return summary

    old_summary = group_memory_cache.get(group_name) or ""

    recent = []
    for m in trimmed_history:
        sender = m.get("sender") or m.get("username") or m.get("display_name") or "unknown"
        if sender == "PSI-09":
            continue

        content = m.get("content", "")
        if config.DISCORD_ID:
            content = re.sub(r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@PSI-09", content)
        recent.append(f"[{sender}]: {content}")

    llm_feed = [
        {"role": "system", "content": f"<group_summary_prompt>\n{GROUP_SUMMARY_PROMPT}\n</group_summary_prompt>"},
        {"role": "user", "content": f"<chat_history>\n" + "\n".join(recent) + "\n</chat_history>"}
    ]

    try:
        new_summary = query_private_brain(llm_feed, temperature=0.8, max_output_tokens=300, task_type="group_summary")
    except Exception as e:
        logger.warning(f"Group summarization failed for {group_name}: {e}")
        new_summary = old_summary

    if new_summary and new_summary != old_summary:
        group_memory_cache.set(group_name, new_summary)

    return new_summary or old_summary

def summarize_global_history(global_key, evolve=False):
    _, trimmed_history = fetch_history(
        global_history_col, global_key, config.MAX_HISTORY_MESSAGES, config.MAX_HISTORY_TOKENS
    )
    
    if not trimmed_history:
        return None

    old_summary = global_memory_cache.get(global_key)
    history_lines = [f"[User]: {m['content']}" for m in trimmed_history if m.get("role") == "user"]
    
    if not history_lines:
        return old_summary

    if old_summary is None:
        sys_prompt = GLOBAL_FIRST_CONTACT_PROMPT
        user_content = f"<cross_platform_history>\n[User]: {trimmed_history[-1]['content']}\n</cross_platform_history>"
    else:
        if not evolve:
            return old_summary
        sys_prompt = GLOBAL_EVOLUTION_PROMPT.format(old_summary=old_summary)
        user_content = f"<cross_platform_history>\n" + "\n".join(history_lines) + "\n</cross_platform_history>"

    llm_feed = [
        {"role": "system", "content": f"<global_omniscient_prompt>\n{sys_prompt}\n</global_omniscient_prompt>"},
        {"role": "user", "content": user_content}
    ]

    try:
        current_task = "evolution" if old_summary else "first_contact"
        new_summary = query_private_brain(llm_feed, temperature=0.8, max_output_tokens=300, task_type=current_task)
        if new_summary:
            global_memory_cache.set(global_key, new_summary)
            logger.info(f"Global profile updated for {global_key} ({current_task})")
            return new_summary
        return old_summary
    except Exception as e:
        logger.warning(f"Global evolution failed for {global_key}: {e}")
        return old_summary

# --- COMBAT ENGINE (Roasts) ---
def get_roast_response(group_name, username, tagged_users=None):
    tagged_users = tagged_users or []
    user_key = f"{group_name}:{username}"
    is_private_env = group_name in ["private_chat"]

    _, trimmed_user = fetch_history(
        history_col, user_key, config.MAX_HISTORY_MESSAGES, config.MAX_HISTORY_TOKENS
    )

    if not is_private_env:
        _, trimmed_group = fetch_history(
            group_history_col, group_name, config.GROUP_HISTORY_SLICE, config.GROUP_HISTORY_TOKEN_LIMIT
        )
        group_memory = group_memory_cache.get(group_name)
    else:
        trimmed_group, group_memory = [], None

    llm_feed = []

    system_content = ROAST_PROMPT if is_private_env else GROUP_ROAST_PROMPT
    llm_feed.append({"role": "system", "content": f"<roast_prompt>\n{system_content}\n</roast_prompt>"})

    user_memory = memory_cache.get(user_key)
    if user_memory:
        llm_feed.append({"role": "system", "content": f"<local_group_profile>\n{user_memory.strip()}\n</local_group_profile>"})

    global_key = f"Global:{username}"
    global_memory = global_memory_cache.get(global_key)
    if global_memory:
        llm_feed.append({"role": "system", "content": f"<global_omniscient_profile>\n{global_memory.strip()}\n</global_omniscient_profile>"})

    if not is_private_env and group_memory:
        llm_feed.append({"role": "system", "content": f"<group_dynamic_summary>\n{group_memory.strip()}\n</group_dynamic_summary>"})

    tagged_profiles = fetch_tagged_profiles(group_name, tagged_users)
    if tagged_profiles:
        joined_profiles = "\n\n".join(tagged_profiles)
        llm_feed.append({"role": "system", "content": f"<tagged_member_profiles>\n{joined_profiles}\n</tagged_member_profiles>"})

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
            for entry in trimmed_group:
                s = entry.get("sender") or entry.get("username") or entry.get("display_name") or "unknown"
                c = entry.get("content", "").strip()
                if config.DISCORD_ID:
                    c = re.sub(r"<@!?" + re.escape(config.DISCORD_ID) + r">", "@PSI-09", c)
                if c:
                    history_lines.append(f"[{s}]: {c}")

    history_text = "\n".join(history_lines) if history_lines else "[No recent history]"
    
    # --- TARGET MESSAGE ISOLATION ---
    target_message = ""
    if is_private_env and trimmed_user:
        target_message = trimmed_user[-1].get("content", "")
    elif not is_private_env and trimmed_group:
        target_message = trimmed_group[-1].get("content", "")

    # Feed both the history context AND the isolated target wrapped in XML
    llm_feed.append({
        "role": "user", 
        "content": (
            f"<chat_history>\n{history_text}\n</chat_history>\n\n"
            f"<target_message>\n{target_message}\n</target_message>"
        )
    })

    try:
        base_reply = query_private_brain(llm_feed=llm_feed, temperature=0.9, max_output_tokens=150, task_type="roast")
    except Exception as e:
        logger.error(f"AI Error: {e}")
        base_reply = ""

    # CLEANUP REGEX
    temp_reply = re.sub(r"^(?:\[.*?\]|PSI-09)\s*:\s*", "", base_reply or "", flags=re.IGNORECASE)
    temp_reply = re.sub(r"\n\[.*?\]:.*", "", temp_reply, flags=re.DOTALL) 
    clean_reply = re.sub(r"\s{2,}", " ", temp_reply).strip()

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

# --- API ROUTES ---
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
        display_name = data.get("display_name") or username
        group_name = data.get("group_name") or "DefaultGroup"

        if group_name.lower() in ["defaultgroup", "discord_dm", "mc_dm"]:
            group_name = "private_chat"

        tagged_users = data.get("tagged_users", [])
        force_reply = data.get("force_reply", False)
        platform = data.get("platform", "Unknown")

        if not username or not sender_id or not raw_message:
            return jsonify({"reply": ""}), 200

        user_message = raw_message
        if config.DISCORD_ID:
            user_message = re.sub(
                r"<@!?" + re.escape(str(config.DISCORD_ID)) + r">",
                "@PSI-09",
                user_message,
            )

        is_private = group_name in ["private_chat"]

        if is_private:
            store_user_message(platform, group_name, sender_id, username, display_name, user_message)
        else:
            store_group_message(platform, group_name, sender_id, username, display_name, user_message)
            store_user_message(platform, group_name, sender_id, username, display_name, user_message)

        user_key = f"{group_name}:{username}"

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

        global_key = f"Global:{username}"
        with global_locks[global_key]:
            global_msg_count = global_memory_cache.increment(global_key)
            current_global_memory = global_memory_cache.get(global_key)
            
            if current_global_memory is None:
                logger.info(f"Initiating First Contact (Global) for {global_key}")
                summarize_global_history(global_key, evolve=False)
                global_memory_cache.reset_count(global_key)
                
            elif global_msg_count >= config.EVOLVE_EVERY_N_MESSAGES:
                logger.info(f"Initiating Evolution (Global) for {global_key}")
                summarize_global_history(global_key, evolve=True)
                global_memory_cache.reset_count(global_key)

        if not is_private:
            with group_locks[group_name]:
                group_msg_count = group_memory_cache.increment(group_name)
                
                if group_msg_count >= config.GROUP_SUMMARY_EVERY_N:
                    logger.info(f"Initiating Group Summary for {group_name}")
                    # Cleaned up /psi09 route
                    summarize_group_history(group_name)
                    group_memory_cache.reset_count(group_name)

        if is_private or force_reply or bot_mentioned_in(raw_message):
            reply = get_roast_response(group_name, username, tagged_users)
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