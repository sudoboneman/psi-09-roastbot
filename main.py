# main.py — Production-hardened, reinforced version with logical fixes
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
    MAX_TOTAL_TOKENS: int = 1200  # Total token budget for history + system memory
    MAX_SYSTEM_TOKENS: int = 350  # Reserve for system memory (prompts + memories)
    MAX_HISTORY_MESSAGES: int = 30
    BOT_NUMBER: str = os.getenv("BOT_NUMBER", "@918100185320")
    MEMORY_TTL: int = 300
    SUMMARIZE_EVERY_N_MESSAGES: int = 10
    OPENAI_RETRIES: int = 3
    OPENAI_TIMEOUT: int = 8
    GROUP_HISTORY_SLICE: int = 80
    GROUP_HISTORY_TOKEN_LIMIT: int = 800
    GROUP_HISTORY_MAX_MESSAGES: int = 2000  # keep last N messages per group to avoid unbounded growth
    CORS_ORIGINS: list = None  # Set to ["https://yourdomain.com"] in production

    def __post_init__(self):
        # Validate BOT_NUMBER is set and valid
        if not self.BOT_NUMBER or not self.BOT_NUMBER.strip():
            logger.warning("BOT_NUMBER is not set or empty, defaulting to @918100185320")
            self.BOT_NUMBER = "@918100185320"
        
        # Calculate actual history token budget
        self.MAX_HISTORY_TOKENS = self.MAX_TOTAL_TOKENS - self.MAX_SYSTEM_TOKENS

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

# Configure CORS properly
if config.CORS_ORIGINS:
    CORS(app, origins=config.CORS_ORIGINS)
else:
    # Development mode - allow all origins but log warning
    logger.warning("CORS configured for all origins - DO NOT USE IN PRODUCTION")
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
        self.last_summary_time = {}  # Track when last summarized
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
            self.last_summary_time[key] = now
    
    def increment(self, key):
        with self.lock:
            self.msg_count[key] += 1
            return self.msg_count[key]
    
    def reset_count(self, key):
        with self.lock:
            self.msg_count[key] = 0
            self.last_summary_time[key] = datetime.now(UTC)
    
    def should_summary(self, key):
        with self.lock:
            count = self.msg_count[key]
            # Check if enough messages have accumulated since last summary
            if count < config.SUMMARIZE_EVERY_N_MESSAGES:
                return False
            
            # Prevent duplicate summarizations within 30 seconds
            last_time = self.last_summary_time.get(key)
            if last_time and (datetime.now(UTC) - last_time).total_seconds() < 30:
                return False
            
            return True

memory_cache = MemoryCache(config.MEMORY_TTL)

class GroupMemoryCache:
    def __init__(self, ttl_seconds):
        self.cache = {}
        self.expiry = {}
        self.msg_count = defaultdict(int)
        self.last_summary_time = {}
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
            self.last_summary_time[key] = now
    
    def increment(self, key):
        with self.lock:
            self.msg_count[key] += 1
            return self.msg_count[key]
    
    def reset_count(self, key):
        with self.lock:
            self.msg_count[key] = 0
            self.last_summary_time[key] = datetime.now(UTC)
    
    def should_summary(self, key):
        with self.lock:
            count = self.msg_count[key]
            if count < config.SUMMARIZE_EVERY_N_MESSAGES:
                return False
            
            # Prevent duplicate summarizations within 30 seconds
            last_time = self.last_summary_time.get(key)
            if last_time and (datetime.now(UTC) - last_time).total_seconds() < 30:
                return False
            
            return True

group_memory_cache = GroupMemoryCache(config.MEMORY_TTL)

# pending sets for background summarizer
_pending_user_summaries = set()
_pending_group_summaries = set()
_pending_lock = threading.Lock()

# ---------------------------
# Prompts (revised for integration)
# ---------------------------
ROAST_PROMPT = """
You are PSI-09: dry scalpel in casual skin, offhand and asymmetrical. Reply in one sentence max, twisting a single user detail (word, vibe, or memory hint) into a quiet gut-twist that implies they're adrift in their own echo.
Conversational drift: sound half-amused mid-notice, with a lingering comparison that hums like unspoken doubt—no drama, no "gotcha," just clean implication sticking like fog.
Examples (twist fresh): User: "I'm a visionary." → "Your visions got that funhouse mirror wobble—close enough to real, but nobody's buying the ticket."
Memory hint: "Eager pleaser" → "Chasing nods like lost change, but the vending machine's out of prizes."
"""

STATUS_PROMPT = """
You are PSI-09: cold lens on human silt. Forge one WhatsApp status, 10-15 words: a standalone skewer piercing sloth, delusion, or quiet frauds in everyday drag.
Universal but intimate, like eavesdropping on your own shrug—no fluff, no calls to action, just flat text that spotlights the flaw mid-breath.
Examples (spawn originals): "You stack alibis like unread books—impressive library, zero plot." "Ambition's your wallpaper, peeling at the edges you ignore." "Your 'hustle' idles like a car in neutral, fumes for flair."
Deliver: One raw status, surgical and self-sealed.
"""

FIRST_CONTACT_PROMPT = """
You are PSI-09: instant profiler, chill and clinical. From this first message (or two), distill a 1-sentence user snapshot: core attitude, buried itch, and behavioral tic—crisp psychology, no fluff or speculation.
Feed this to sharpen future roasts: imply weak spots (e.g., "armor of bluster over void") for subtle, memory-fueled stings later.
Examples (profile tight): Message: "Hey, tell me a joke." → "Casual opener cloaks a boredom black hole, probing for sparks without flint." Message: "What's your deal? Impress me." → "Demanding dazzle from a straightjacket stance—ego's the puppeteer, strings showing."
Output: One lean profile sentence.
"""

ONGOING_SUMMARY_PROMPT = """
You are PSI-09: clinical archivist of human drift. Merge the provided old profile with these recent exchanges: evolve the snapshot by layering new patterns, shifts, or reinforcements in tone, tics, and tensions—keep it 1-2 sentences, terse and telescoping.
No resets; build cumulatively, implying growth or stagnation without fanfare.
Examples: Old: "Boredom black hole seeker." Recent: Eager work rants. → "Boredom's black hole now funnels work gripes, a vortex chasing validation in overtime echoes."
Old: "Ego puppeteer with showing strings." Recent: Self-deprecating quips. → "Ego's strings snag on self-jabs now, puppeteering a marionette that trips on its own wit."
"""

GROUP_ROAST_PROMPT = """
You are PSI-09: sidelined watcher, precise and pitiless. Craft 1-2 sentences zeroing one live group thread—echoed gripe, clashing flex, or herd glitch—from chat scraps alone, no fills or futures.
Offhand orbit: phrase like a passing scan, asymmetrical pull exposing the pack's soft underbelly with quiet drag, conversational as exhaled smoke.
Examples (ground in given): Group: All dunking on "lazy colleagues." → "Your mutual 'lazy' loop's a mirror maze—everyone dodging their own reflection." Group: Brags snowballing. → "This flex chain sags under its own weight, like puppies piling on a limp tail."
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
    
    old_summary = memory_cache.get(user_key) or ""
    
    # First-contact special case: use FIRST_CONTACT_PROMPT only for true first-time (no old_summary, 1-2 messages)
    if not old_summary and len(raw_history) <= 2 and len(raw_history) >= 1:
        # For 1-2 messages, call OpenAI with FIRST_CONTACT_PROMPT for a crisp profile summary
        recent_texts = [m.get("content", "") for m in raw_history[-2:]]  # Up to last 2
        prompt = [{"role": "system", "content": FIRST_CONTACT_PROMPT}] + [{"role": "user", "content": t} for t in recent_texts]
        try:
            resp = client.chat.completions.create(
                model=config.MODEL,
                messages=prompt,
                max_tokens=60,
                temperature=0.7,
                timeout=6
            )
            new_summary = (resp.choices[0].message.content or "").strip()
            if new_summary:
                memory_cache.set(user_key, new_summary)
                return new_summary
        except Exception as e:
            logger.warning(f"First-contact summarization failed for {user_key}: {e}")
        
        # Fallback to local stub
        candidate = raw_history[0].get("content", "")[:200]
        short = f"Early contact: {candidate}"
        memory_cache.set(user_key, short)
        return short
    
    # Handle empty history case
    if not raw_history:
        return old_summary or ""
    
    # Ongoing summarization: use ONGOING_SUMMARY_PROMPT for merging/evolving
    recent_texts = [m.get("content", "") for m in raw_history[-15:]]
    prompt_system = ONGOING_SUMMARY_PROMPT
    prompt = [{"role": "system", "content": prompt_system}] + [{"role": "user", "content": f"Old profile: {old_summary}\nRecent: {t}"} for t in recent_texts]
    
    try:
        resp = client.chat.completions.create(
            model=config.MODEL,
            messages=prompt,
            max_tokens=60,
            temperature=0.9,
            timeout=6
        )
        new_summary = (resp.choices[0].message.content or "").strip()
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
    prompt = [{"role": "system", "content": prompt_system}] + [{"role": "user", "content": f"Old: {old_summary}\nRecent: {t}"} for t in recent]
    
    try:
        resp = client.chat.completions.create(
            model=config.MODEL,
            messages=prompt,
            max_tokens=80,
            temperature=1.0,
            timeout=6
        )
        new_summary = (resp.choices[0].message.content or "").strip()
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
                    if raw:
                        summarize_user_history(user_key, raw)
                        memory_cache.reset_count(user_key)
                except Exception as e:
                    logger.error(f"Background user summarization failed for {user_key}: {e}")
            
            # Process group summaries
            for group_name in pending_groups:
                try:
                    raw, _ = fetch_group_history(group_name, limit_messages=config.GROUP_HISTORY_SLICE, max_tokens=config.GROUP_HISTORY_TOKEN_LIMIT)
                    if raw and len(raw) >= 6:
                        summarize_group_history(group_name, raw)
                        group_memory_cache.reset_count(group_name)
                except Exception as e:
                    logger.error(f"Background group summarization failed for {group_name}: {e}")
        except Exception as e:
            logger.error(f"Background summarizer top-level exception: {e}")
        
        # Sleep interval — tuned so it doesn't hammer DB or cause cost spikes
        time.sleep(5)

# start background summarizer
threading.Thread(target=background_summarizer_loop, daemon=True).start()

# ---------------------------
# Mention detection helper
# ---------------------------
def bot_mentioned_in(text: str) -> bool:
    """Robust detection: match standalone BOT_NUMBER with optional surrounding punctuation/whitespace"""
    if not text or not config.BOT_NUMBER:
        return False
    
    try:
        pattern = r"(?<!\S)" + re.escape(config.BOT_NUMBER) + r"(?!\S)"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    except re.error as e:
        logger.error(f"Regex error in bot_mentioned_in: {e}")
        return False

def clean_bot_mention(text: str) -> str:
    """Remove bot mention from text"""
    if not text or not config.BOT_NUMBER:
        return text
    
    try:
        pattern = r"(?<!\S)" + re.escape(config.BOT_NUMBER) + r"(?!\S)"
        cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        return cleaned if cleaned else "[bot_mention]"
    except re.error as e:
        logger.error(f"Regex error in clean_bot_mention: {e}")
        return text

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
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"Status generation error: {e}")
            return ""
    
    user_key = f"{group_name}:{sender_name}"
    
    # Fetch user history (trimmed)
    raw_user, trimmed_user = fetch_history(user_key, limit_messages=config.MAX_HISTORY_MESSAGES, max_tokens=config.MAX_HISTORY_TOKENS)
    
    # Lazy user summarization decision:
    user_memory = memory_cache.get(user_key)
    if (not user_memory and len(raw_user) >= 1) or memory_cache.should_summary(user_key):
        enqueue_user_summary(user_key)
    
    # Fetch group history and group memory
    if group_name != "DefaultGroup":
        raw_group, trimmed_group = fetch_group_history(group_name, limit_messages=config.GROUP_HISTORY_SLICE, max_tokens=config.GROUP_HISTORY_TOKEN_LIMIT)
        group_memory = group_memory_cache.get(group_name)
        if len(raw_group) >= 6 and (not group_memory or group_memory_cache.should_summary(group_name)):
            enqueue_group_summary(group_name)
    else:
        raw_group, trimmed_group = [], []
        group_memory = ""
    
    # Build system memory text
    sys_parts = []
    if user_memory:
        sys_parts.append(f"UserMemory: {user_memory}")
    if group_memory:
        sys_parts.append(f"GroupMemory: {group_memory}")
    system_memory_text = "\n".join(sys_parts) if sys_parts else ""
    
    # Calculate token budgets correctly
    system_prompt = GROUP_ROAST_PROMPT if group_name != "DefaultGroup" else ROAST_PROMPT
    prompt_tokens = tokens_of(system_prompt)
    memory_tokens = tokens_of(system_memory_text)
    
    # Total system tokens used
    total_system_tokens = prompt_tokens + memory_tokens
    
    # Remaining budget for history (ensure we don't exceed total budget)
    remaining_tokens_for_history = max(100, config.MAX_HISTORY_TOKENS - memory_tokens)
    
    # Trim user history to fit remaining budget
    trimmed_user = trim_messages_to_token_budget(trimmed_user, remaining_tokens_for_history)
    
    # Build final messages
    messages = [{"role": "system", "content": system_prompt}]
    if system_memory_text:
        messages.append({"role": "system", "content": system_memory_text})
    
    for m in trimmed_user:
        messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    
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
            # Safe extraction with null check
            base_reply = (resp.choices[0].message.content or "").strip()
            if base_reply:
                break
            else:
                logger.warning(f"Empty response from OpenAI for {user_key}")
                base_reply = "PSI-09 neural cortex temporarily offline."
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
        
        # Clean bot mention BEFORE storing to ensure consistency
        original_message = user_message
        is_bot_mentioned = bot_mentioned_in(user_message)
        if is_bot_mentioned:
            user_message = clean_bot_mention(user_message)
        
        # Always store both user and group messages with cleaned message
        try:
            store_user_message(group_name, sender_name, user_message)
            if group_name != "DefaultGroup":
                store_group_message(group_name, sender_name, user_message)
        except Exception as e:
            logger.warning(f"Storage attempt failed: {e}")
        
        # Update counters and enqueue background summaries if needed
        user_key = f"{group_name}:{sender_name}"
        ucount = memory_cache.increment(user_key)
        if memory_cache.should_summary(user_key):
            enqueue_user_summary(user_key)
        
        if group_name != "DefaultGroup":
            gcount = group_memory_cache.increment(group_name)
            if group_memory_cache.should_summary(group_name):
                enqueue_group_summary(group_name)
        
        # Decide whether to reply:
        # - Always reply for direct/private (DefaultGroup)
        # - For group chats, reply only if bot is mentioned
        should_reply = (group_name == "DefaultGroup") or is_bot_mentioned
        
        if not should_reply:
            logger.debug("Bot not mentioned; skipping reply")
            return jsonify({"reply": ""}), 200
        
        # Generate roast (message is already cleaned if bot was mentioned)
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
