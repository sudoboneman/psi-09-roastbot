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
import tiktoken
import requests

from huggingface_hub import login
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

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
UTC = timezone.utc

# Inside your Config dataclass:
@dataclass
class Config:
    MONGO_URI: str = os.getenv("MONGO_URI")
    GROQ_API_KEY_1: str = os.getenv("GROQ_API_KEY_1") 
    GROQ_API_KEY_2: str = os.getenv("GROQ_API_KEY_2") 
    
    # Dual Nvidia Keys
    NVIDIA_API_KEY_1: str = os.getenv("NVIDIA_API_KEY_1")
    NVIDIA_API_KEY_2: str = os.getenv("NVIDIA_API_KEY_2")
    
    @property
    def NVIDIA_KEYS(self):
        # Dynamically returns a list of only the keys that exist
        return [k for k in [self.NVIDIA_API_KEY_1, self.NVIDIA_API_KEY_2] if k]

    # --- DUAL PERSISTENT MODEL CYCLES ---
    ROAST_MODELS: list = __import__("dataclasses").field(default_factory=lambda: [
        "google/gemma-4-31b-it"
    ])
    
    BACKGROUND_MODELS: list = __import__("dataclasses").field(default_factory=lambda: [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-120b"
    ])
    
    # --- TIGHTENED FOR MAXIMUM THROUGHPUT ---
    BOT_NUMBER: str = os.getenv("BOT_NUMBER")
    DISCORD_ID: str = os.getenv("DISCORD_ID")
    DISCORD_ID_2: str = os.getenv("DISCORD_ID_2")
    MEMORY_TTL: int = 500
    
    # DATABASE CEILINGS (Keep these high for memory)
    GROUP_HISTORY_MAX_MESSAGES: int = 50000  #
    GROUP_HISTORY_SLICE: int = 80 # Fast database read
    
    # LLM PAYLOAD CEILINGS (Shrunk to maximize API calls/min)
    MAX_HISTORY_MESSAGES: int = 16 # Down from 30
    MAX_HISTORY_TOKENS: int = 400 # Tightly caps the user's personal history
    GROUP_HISTORY_TOKEN_LIMIT: int = 2000 # Tightly caps the group's history
    
    # THE PACING ENGINE (Tuned for 6b6t Anarchy traffic)
    EVOLVE_EVERY_N_MESSAGES: int = 50 # Evolve active users frequently
    GROUP_SUMMARY_EVERY_N: int = 300 # Rapidly update the group dynamic

config = Config()


# --- EXISTING STATE TRACKERS ---
roast_model_lock = threading.Lock()
active_roast_index = 0

bg_model_lock = threading.Lock()
active_bg_index = 0

    # --- NEW NVIDIA KEY TRACKER ---
nvidia_key_lock = threading.Lock()
active_nvidia_key_index = 0

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
    global active_roast_index, active_bg_index, active_nvidia_key_index
    
    is_roast = (task_type == "roast")
    active_list = config.ROAST_MODELS if is_roast else config.BACKGROUND_MODELS
    active_lock = roast_model_lock if is_roast else bg_model_lock

    base_delay = 1.0 

    for attempt in range(max_retries):
        current_index = active_roast_index if is_roast else active_bg_index
        current_model = active_list[current_index]
        current_nv_key_index = None # Track which Nvidia key we use in this loop
        
        try:
            # --- NVIDIA NIM ROUTING (For Kimi) ---
            if is_roast and ("kimi" in current_model.lower() or "gemma" in current_model.lower()) and config.NVIDIA_KEYS:
                current_nv_key_index = active_nvidia_key_index
                current_key = config.NVIDIA_KEYS[current_nv_key_index]
                
                invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {current_key}",
                    "Accept": "application/json"
                }


                payload = {
                    "model": current_model,
                    "messages": llm_feed,
                    "max_tokens": max_output_tokens,
                    "temperature": temperature,
                    "top_p": 1.00,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False} 
                }
                
                logger.info(f"[ROAST] Firing request to Nvidia ({current_model})...")
                start_time = time.time()

                # Make the raw HTTP request to Nvidia
                response = requests.post(invoke_url, headers=headers, json=payload, timeout=180)
                
                # --- NEW: NVIDIA ERROR X-RAY ---
                if response.status_code != 200:
                    logger.error(f"NVIDIA API REJECTED REQUEST. Status: {response.status_code}")
                    logger.error(f"NVIDIA RAW ERROR BODY: {response.text}")
                
                response.raise_for_status() # Trigger except block on 4xx/5xx errors
                
                data = response.json()
                content = data["choices"][0]["message"].get("content") or ""
                    
                logger.info(f"[ROAST] Nvidia responded in {time.time() - start_time:.2f} seconds.")
                return content.strip()
                
            # --- GROQ ROUTING (For Background & Fallbacks) ---
            else:
                active_client = client_1 if is_roast else client_2
                if not active_client:
                    logger.error("Cannot query brain: Groq Client not initialized.")
                    return None 

                response = active_client.chat.completions.create(
                    model=current_model,
                    messages=llm_feed,
                    temperature=temperature,
                    max_completion_tokens=max_output_tokens,
                    top_p=1
                )
                content = response.choices[0].message.content or ""
                return content.strip()
            
        except Exception as e:
            error_msg = str(e).lower()
            if attempt == max_retries - 1:
                logger.error(f"FATAL API ERROR (After {max_retries} attempts): {e}")
                return None
            
            # PATH A: Token Exhaustion or Rate Limit
            if "429" in error_msg or "rate limit" in error_msg or "token" in error_msg or "credit" in error_msg:
                
                # If Nvidia failed, rotate the Nvidia KEY
                if current_nv_key_index is not None:
                    with nvidia_key_lock:
                        if active_nvidia_key_index == current_nv_key_index:
                            active_nvidia_key_index = (active_nvidia_key_index + 1) % len(config.NVIDIA_KEYS)
                            logger.warning(f"[{task_type}] Nvidia Key {current_nv_key_index + 1} exhausted. SWITCHING to Key {active_nvidia_key_index + 1}.")
                
                # If Groq failed, rotate the Groq MODEL
                else:
                    with active_lock:
                        check_index = active_roast_index if is_roast else active_bg_index
                        if active_list[check_index] == current_model:
                            new_index = (check_index + 1) % len(active_list)
                            if is_roast: active_roast_index = new_index
                            else: active_bg_index = new_index
                            logger.warning(f"[{task_type}] {current_model} exhausted. SWITCHING to {active_list[new_index]}.")
                            
                time.sleep(random.uniform(0.2, 0.5))

            # PATH B: Server Overload 
            elif "500" in error_msg or "502" in error_msg or "503" in error_msg or "504" in error_msg or "connection" in error_msg:
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.1, 1.0)
                logger.warning(f"[{task_type}] Server Overload ({e}). Backing off for {sleep_time:.2f}s...")
                time.sleep(sleep_time)

            # PATH C: Unknown Fatal Error
            else:
                logger.error(f"Non-retriable API ERROR: {e}")
                return None

app = Flask(__name__)
CORS(app)

# --- LAZY LOAD TOKENIZERS ---
all_models = config.ROAST_MODELS + config.BACKGROUND_MODELS

KIMI_ENCODING = None
LLAMA_ENCODING = None
GPT_ENCODING = None
QWEN_ENCODING = None
GEMMA_ENCODING = None

def background_tokenizer_load():
    global KIMI_ENCODING, LLAMA_ENCODING, GPT_ENCODING, QWEN_ENCODING, GEMMA_ENCODING
    logger.info("Starting background download of tokenizers...")
    
    if any("kimi" in m.lower() or "moonshot" in m.lower() for m in all_models):
        try:
            KIMI_ENCODING = AutoTokenizer.from_pretrained("moonshotai/Kimi-K2-Instruct", trust_remote_code=True)
            logger.info("Kimi tokenizer loaded.")
        except Exception: pass

    if any("llama" in m.lower() for m in all_models):
        try:
            LLAMA_ENCODING = AutoTokenizer.from_pretrained("unsloth/Llama-4-Scout-17B-16E-Instruct", trust_remote_code=True)
            logger.info("Llama tokenizer loaded.")
        except Exception: pass

    if any("gpt" in m.lower() for m in all_models):
        try:
            GPT_ENCODING = tiktoken.get_encoding("cl100k_base")
            logger.info("GPT tokenizer loaded.")
        except Exception: pass

    if any("qwen" in m.lower() for m in all_models):
        try:
            QWEN_ENCODING = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
            logger.info("Qwen tokenizer loaded.")
        except Exception: pass

    if any("gemma" in m.lower() for m in all_models):
        try:
            GEMMA_ENCODING = AutoTokenizer.from_pretrained("google/gemma-4-31b-it")
            logger.info("Gemma tokenizer loaded.")
        except Exception as e: pass

threading.Thread(target=background_tokenizer_load, daemon=True).start()

def tokens_of(text: str, task_type: str = "roast") -> int:
    if not text: return 0
    global active_roast_index, active_bg_index
    
    current_active_model = config.ROAST_MODELS[active_roast_index].lower() if task_type == "roast" else config.BACKGROUND_MODELS[active_bg_index].lower()
    
    if ("kimi" in current_active_model or "moonshot" in current_active_model) and KIMI_ENCODING:
        return len(KIMI_ENCODING.encode(text))
    if "llama" in current_active_model and LLAMA_ENCODING:
        return len(LLAMA_ENCODING.encode(text))
    if "gpt" in current_active_model and GPT_ENCODING:
        return len(GPT_ENCODING.encode(text))
    if "qwen" in current_active_model and QWEN_ENCODING:
        return len(QWEN_ENCODING.encode(text))
    if "gemma" in current_active_model and GEMMA_ENCODING:
        return len(GEMMA_ENCODING.encode(text))
        
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
def trim_messages_to_token_budget(messages, max_tokens, task_type="roast"):
    total = 0
    trimmed = []
    for m in reversed(messages):
        sender = m.get("sender") or m.get("username") or m.get("display_name") or m.get("role") or "User"
        formatted_text = f"[{sender}]: {m.get('content', '')}"
        t = tokens_of(formatted_text, task_type) 
        
        if total + t > max_tokens:
            break
        trimmed.insert(0, m)
        total += t
    return trimmed

def fetch_history(collection, doc_id, limit_messages, max_input_tokens=None, task_type="roast"):
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
        trimmed = trim_messages_to_token_budget(raw, max_input_tokens, task_type)
        return raw, trimmed
    return raw, raw

def fetch_tagged_profiles(tagged_users, max_targets=3):
    profiles = []
    for u in tagged_users[:max_targets]:
        uid = u.get("id")
        username = u.get("username")
        if not username:
            continue  
        memory_key = f"Global:{username}"
        summary = global_memory_cache.get(memory_key)
        if summary:
            profiles.append(f'<bystander username="{username}" numeric_id="{uid}">\n{summary.strip()}\n</bystander>')
    return profiles

# --- DATABASE STORAGE ---
def store_user_message(platform, group_name, channel_name, sender_id, username, display_name, message):
    user_key = f"{group_name}:{username}"
    global_key = f"Global:{username}"
    
    local_entry = {
        "role": "user",
        "user_id": sender_id,
        "username": username,
        "display_name": display_name,
        "platform": platform, 
        "channel": channel_name, # <-- ADD METADATA
        "content": message,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    
    global_entry = local_entry.copy()
    global_entry["content"] = f"[Sent via {platform} - {group_name} #{channel_name}] {message}"
    
    try:
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

def store_group_message(platform, group_name, channel_name, sender_id, username, display_name, message):
    entry = {
        "sender_id": sender_id,
        "username": username,
        "display_name": display_name,
        "platform": platform,
        "channel": channel_name, # <-- ADD METADATA
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
    
    # Loop through all configured Discord IDs
    for d_id in [config.DISCORD_ID, config.DISCORD_ID_2]:
        if d_id:
            discord_pattern = r"<@!?" + re.escape(str(d_id)) + r">"
            if re.search(discord_pattern, text):
                return True
    return False

# --- SUMMARIZATION ENGINES ---
def summarize_user_history(user_key, evolve=False):
    old_summary = memory_cache.get(user_key)
    current_task = "evolution" if old_summary and evolve else "first_contact"

    _, trimmed_history = fetch_history(
        history_col, user_key, config.MAX_HISTORY_MESSAGES, config.MAX_HISTORY_TOKENS, task_type=current_task
    )

    if not trimmed_history:
        return None

    history_lines = [f"[User]: {m['content']}" for m in trimmed_history if m.get("role") == "user"]
    
    if not history_lines:
        return old_summary

    if old_summary is None:
        sys_prompt = FIRST_CONTACT_PROMPT
        user_content = f"<chat_history>\n{history_lines[-1]}\n</chat_history>"
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
        new_summary = query_private_brain(llm_feed, temperature=0.8, max_output_tokens=1024, task_type=current_task)
        
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
        group_history_col, group_name, config.GROUP_HISTORY_SLICE, config.GROUP_HISTORY_TOKEN_LIMIT, task_type="group_summary"
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
        for d_id in [config.DISCORD_ID, config.DISCORD_ID_2]:
            if d_id:
                content = re.sub(r"<@!?" + re.escape(str(d_id)) + r">", "@PSI-09", content)
        chan = m.get("channel", "unknown")
        recent.append(f"[#{chan}] [{sender}]: {content}")

    llm_feed = [
        {"role": "system", "content": f"<group_summary_prompt>\n{GROUP_SUMMARY_PROMPT}\n</group_summary_prompt>"},
        {"role": "user", "content": f"<chat_history>\n" + "\n".join(recent) + "\n</chat_history>"}
    ]

    try:
        new_summary = query_private_brain(llm_feed, temperature=0.8, max_output_tokens=1024, task_type="group_summary")
    except Exception as e:
        logger.warning(f"Group summarization failed for {group_name}: {e}")
        new_summary = old_summary

    if new_summary and new_summary != old_summary:
        group_memory_cache.set(group_name, new_summary)

    return new_summary or old_summary

def summarize_global_history(global_key, evolve=False):
    old_summary = global_memory_cache.get(global_key)
    current_task = "evolution" if old_summary and evolve else "first_contact"

    _, trimmed_history = fetch_history(
        global_history_col, global_key, config.MAX_HISTORY_MESSAGES, config.MAX_HISTORY_TOKENS, task_type=current_task
    )
    
    if not trimmed_history:
        return None

    history_lines = [f"[User]: {m['content']}" for m in trimmed_history if m.get("role") == "user"]
    
    if not history_lines:
        return old_summary

    if old_summary is None:
        sys_prompt = GLOBAL_FIRST_CONTACT_PROMPT
        user_content = f"<cross_platform_history>\n{history_lines[-1]}\n</cross_platform_history>"
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
        new_summary = query_private_brain(llm_feed, temperature=0.8, max_output_tokens=1024, task_type=current_task)
        if new_summary:
            global_memory_cache.set(global_key, new_summary)
            logger.info(f"Global profile updated for {global_key} ({current_task})")
            return new_summary
        return old_summary
    except Exception as e:
        logger.warning(f"Global evolution failed for {global_key}: {e}")
        return old_summary

# --- COMBAT ENGINE (Roasts) ---
def get_roast_response(group_name, username, active_message, tagged_users=None):
    tagged_users = tagged_users or []
    user_key = f"{group_name}:{username}"
    is_private_env = group_name in ["private_chat"]

    # 1. Fetch the background history for context (but NOT for the target message)
    _, trimmed_user = fetch_history(
        history_col, user_key, config.MAX_HISTORY_MESSAGES, config.MAX_HISTORY_TOKENS, task_type="roast"
    )

    if not is_private_env:
        _, trimmed_group = fetch_history(
            group_history_col, group_name, config.GROUP_HISTORY_SLICE, config.GROUP_HISTORY_TOKEN_LIMIT, task_type="roast"
        )
        group_memory = group_memory_cache.get(group_name)
    else:
        trimmed_group, group_memory = [], None

    llm_feed = []

    # 2. Inject Prompts and Memory Profiles
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

    tagged_profiles = fetch_tagged_profiles(tagged_users)
    if tagged_profiles:
        joined_profiles = "\n\n".join(tagged_profiles)
        llm_feed.append({"role": "system", "content": f"<tagged_member_profiles>\n{joined_profiles}\n</tagged_member_profiles>"})

    # 3. Format the recent chat history
    history_lines = []
    
    if is_private_env:
        if trimmed_user:
            for m in trimmed_user:
                role = m.get("role", "user")
                content = m.get("content", "").strip()
                for d_id in [config.DISCORD_ID, config.DISCORD_ID_2]:
                    if d_id:
                        content = re.sub(r"<@!?" + re.escape(str(d_id)) + r">", "@PSI-09", content)
                if content:
                    prefix = "PSI-09" if role == "assistant" else username
                    chan = m.get("channel", "unknown")
                    history_lines.append(f"[#{chan}] [{prefix}]: {content}")
    else:
        if trimmed_group:
            for entry in trimmed_group:
                s = entry.get("sender") or entry.get("username") or entry.get("display_name") or "unknown"
                c = entry.get("content", "").strip()
                for d_id in [config.DISCORD_ID, config.DISCORD_ID_2]:
                    if d_id:
                        c = re.sub(r"<@!?" + re.escape(str(d_id)) + r">", "@PSI-09", c)
                if c:
                    chan = entry.get("channel", "unknown")
                    history_lines.append(f"[#{chan}] [{s}]: {c}")

    history_text = "\n".join(history_lines) if history_lines else "[No recent history]"
    
    # 4. EXPLICIT TARGET LOCK (Replaces the old database extraction)
    # We pass the history for context, but force the AI to focus on the live active_message
    llm_feed.append({
        "role": "user", 
        "content": (
            f"<chat_history>\n{history_text}\n</chat_history>\n\n"
            f"<active_target>\nTARGET USER: [{username}]\nMESSAGE: {active_message}\n</active_target>"
        )
    })

    # 5. Fire the Engine
    try:
        base_reply = query_private_brain(llm_feed=llm_feed, temperature=0.75, max_output_tokens=1024, task_type="roast")
    except Exception as e:
        base_reply = ""

    if not base_reply:
        return ""

    # --- RAW OUTPUT X-RAY ---
    logger.info(f"RAW OUTPUT FROM NVIDIA:\n{'='*40}\n{base_reply}\n{'='*40}")

    # 6. Clean up hallucinations or prefix tags
    clean_reply = re.sub(r"<\|?think\|?>.*?</\|?think\|?>\s*", "", clean_reply, flags=re.DOTALL | re.IGNORECASE)
    clean_reply = re.sub(r"<\|?think\|?>.*", "", clean_reply, flags=re.DOTALL | re.IGNORECASE)
    clean_reply = re.sub(r"<\|channel>thought.*?</channel\|>\s*", "", clean_reply, flags=re.DOTALL | re.IGNORECASE)
    clean_reply = re.sub(r"<think>.*?</think>\s*", "", base_reply, flags=re.DOTALL | re.IGNORECASE)
    clean_reply = re.sub(r"<think>.*", "", clean_reply, flags=re.DOTALL | re.IGNORECASE)
    clean_reply = clean_reply.strip()
    
    if not clean_reply:
        return ""

    logger.info(f"FINAL PAYLOAD DELIVERED: {clean_reply}")
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
        channel_name = data.get("channel") or "unknown"

        if group_name.lower() in ["defaultgroup", "discord_dm"]:
            group_name = "private_chat"

        tagged_users = data.get("tagged_users", [])
        force_reply = data.get("force_reply", False)
        platform = data.get("platform", "Unknown")

        if not username or not sender_id or not raw_message:
            return jsonify({"reply": ""}), 200

        user_message = raw_message
        for d_id in [config.DISCORD_ID, config.DISCORD_ID_2]:
            if d_id:
                user_message = re.sub(
                    r"<@!?" + re.escape(str(d_id)) + r">",
                    "@PSI-09",
                    user_message,
                )

        is_private = group_name in ["private_chat"]
        user_key = f"{group_name}:{username}"
        global_key = f"Global:{username}"

        # 1. GENERATE ROAST FIRST (Before database storage)
        will_reply = is_private or force_reply or bot_mentioned_in(raw_message)
        reply = ""
        if will_reply:
            reply = get_roast_response(group_name, username, user_message, tagged_users)

        # 2. STORE USER MESSAGE
        if is_private:
            store_user_message(platform, group_name, channel_name, sender_id, username, display_name, user_message)
        else:
            store_group_message(platform, group_name, channel_name, sender_id, username, display_name, user_message)
            store_user_message(platform, group_name, channel_name, sender_id, username, display_name, user_message)

        # 3. STORE BOT REPLY (Chronologically after the user message)
        if reply:
            user_entry = {
                "role": "assistant",
                "content": reply,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            try:
                history_col.update_one({"_id": user_key}, {"$push": {"messages": user_entry}}, upsert=True)
                if not is_private:
                    group_entry = {
                        "sender": "PSI-09",
                        "content": reply,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    group_history_col.update_one(
                        {"_id": group_name},
                        {"$push": {"messages": {"$each": [group_entry], "$slice": -config.GROUP_HISTORY_MAX_MESSAGES}}},
                        upsert=True,
                    )
            except Exception as e:
                logger.warning(f"Reply storage failed: {e}")

        # 4. RUN EVOLUTION CHECKS LAST
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
                    summarize_group_history(group_name)
                    group_memory_cache.reset_count(group_name)

        return jsonify({"reply": reply}), 200

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