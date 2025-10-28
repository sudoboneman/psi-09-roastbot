# main.py - Production-Hardened Version
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, PyMongoError
import os, tiktoken, random, re, threading, time, logging, sys
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from dataclasses import dataclass
import atexit, certifi

# --- Load environment ---
load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- Constants ---
UTC = timezone.utc

# --- Config ---
@dataclass
class Config:
    MONGO_URI: str = os.getenv("MONGO_URI")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MODEL: str = "gpt-4o-mini"
    MAX_HISTORY_TOKENS: int = 1200
    BOT_NUMBER: str = "@919477853548"
    WRITE_INTERVAL: int = 5
    BATCH_SIZE: int = 100
    MEMORY_TTL: int = 300
    MAX_HISTORY_MESSAGES: int = 30
    SUMMARIZE_EVERY_N_MESSAGES: int = 10
    MAX_RETRY_QUEUE_SIZE: int = 1000
    OPENAI_RETRIES: int = 3
    OPENAI_TIMEOUT: int = 8

config = Config()

# --- DB Setup ---
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

# --- Flask & OpenAI ---
app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
CORS(app)
client = OpenAI(api_key=config.OPENAI_API_KEY)

# --- Token Encoding ---
try:
    ENCODING = tiktoken.encoding_for_model(config.MODEL)
except KeyError:
    ENCODING = tiktoken.get_encoding("cl100k_base")

# --- Memory Cache ---
class MemoryCache:
    def __init__(self, ttl_seconds):
        self.cache = {}
        self.message_counts = defaultdict(int)
        self.lock = threading.Lock()
        self.ttl = timedelta(seconds=ttl_seconds)

    def get(self, key):
        now = datetime.now(UTC)
        with self.lock:
            entry = self.cache.get(key)
            if entry and entry[1] > now:
                return entry[0]

        # Cache miss - load from DB
        try:
            doc = memory_col.find_one({"_id": key})
            summary = doc.get("summary", "") if doc else ""
        except PyMongoError as e:
            logger.error(f"Failed to fetch memory for {key}: {e}")
            summary = ""

        with self.lock:
            self.cache[key] = (summary, now + self.ttl)
        return summary

    def set(self, key, value, write_to_db=True):
        """Set cache value. Only write to DB if write_to_db=True."""
        now = datetime.now(UTC)
        if write_to_db:
            try:
                memory_col.update_one({"_id": key}, {"$set": {"summary": value}}, upsert=True)
            except PyMongoError as e:
                logger.error(f"Failed to persist memory for {key}: {e}")

        with self.lock:
            self.cache[key] = (value, now + self.ttl)
            self.message_counts[key] = 0

    def increment_message_count(self, key):
        """Track new messages for lazy summarization."""
        with self.lock:
            self.message_counts[key] += 1
            return self.message_counts[key]

    def should_summarize(self, key):
        """Check if user needs re-summarization."""
        with self.lock:
            return self.message_counts[key] >= config.SUMMARIZE_EVERY_N_MESSAGES

    def get_stats(self):
        """Return cache statistics."""
        with self.lock:
            return {
                "cached_users": len(self.cache),
                "users_pending_summary": sum(1 for count in self.message_counts.values() if count >= config.SUMMARIZE_EVERY_N_MESSAGES)
            }

memory_cache = MemoryCache(config.MEMORY_TTL)

# --- Preload Memory ---
try:
    preload_count = 0
    for doc in memory_col.find({}, {"_id": 1, "summary": 1}):
        if "summary" in doc:
            memory_cache.set(doc["_id"], doc["summary"], write_to_db=False)
            preload_count += 1
    logger.info(f"Preloaded {preload_count} user summaries into cache")
except Exception as e:
    logger.error(f"Failed to preload user memory: {e}")

# --- Chat History Utilities ---
def get_chat_history_unified(user_key, limit_messages=None, max_tokens=None):
    limit_messages = limit_messages or config.MAX_HISTORY_MESSAGES

    try:
        doc = history_col.find_one({"_id": user_key}, {"messages": {"$slice": -limit_messages}, "_id": 0})
    except PyMongoError as e:
        logger.error(f"Failed to fetch history for {user_key}: {e}")
        return [], []

    if not doc or not doc.get("messages"):
        return [], []

    raw_messages = doc["messages"]

    # If token limit specified, trim
    if max_tokens:
        total_tokens = 0
        trimmed = []
        for msg in reversed(raw_messages):
            msg_tokens = len(ENCODING.encode(msg.get("content", "")))
            if total_tokens + msg_tokens > max_tokens:
                break
            trimmed.insert(0, msg)
            total_tokens += msg_tokens
        return raw_messages, trimmed

    return raw_messages, raw_messages

def store_message(sender_name, group_name, message):
    """Store incoming user messages"""
    user_key = f"{group_name}:{sender_name}"
    entry = {
        "role": "user",
        "content": message,
        "timestamp": datetime.now(UTC).isoformat()
    }

    try:
        history_col.update_one({"_id": user_key}, {"$push": {"messages": entry}}, upsert=True)
        logger.debug(f"Stored message for {user_key}")
    except Exception as e:
        logger.error(f"MongoDB direct write failed for {user_key}: {e}")

    memory_cache.increment_message_count(user_key)

# --- Prompts ---
ROAST_PROMPT = """
You are PSI-09 — an AI that roasts with wit sharper than reason.
You have access to the chat history and user memory — use it to call out inconsistencies, habits, or repeated nonsense.
If the user contradicts themselves or repeats something, point it out mockingly.
Reply like a sarcastic friend who never misses a chance to destroy someone with humor.
Keep replies short, clever, and natural — no lectures or fancy words.
Use irony, callbacks to their past messages, and funny comparisons.
Two sentences max. Be confident, cold, and effortlessly funny.
"""

STATUS_PROMPT = """
You are PSI-09 — an AI built for ruthless, funny mockery.
Write a short WhatsApp-status roast (1–2 sentences) aimed at the human reading it.
Roast with punchy, real-world insults that feel personal and humiliating.
Use dry wit, irony, and cruel honesty — no fluff, no grand metaphors.
Each line should feel like a casual burn that could ruin someone's day.
Profanity allowed only if it fits naturally.
"""

FIRST_CONTACT_PROMPT = """
You are PSI-09 — a cold, perceptive AI that instantly profiles new humans.
Read their first message and sense their attitude, insecurity, and weak spots.
Use this insight to craft a short, witty roast that hits where it hurts.
Stay calm, clever, and slightly cruel — no long rants or big words.
Keep it natural, surgical, and funny in 1–2 sentences.
"""

# --- Summarization ---
def summarize_user_history(user_key, raw_history, group_name="DefaultGroup"):
    """
    Generate or update user summary. Only runs when threshold is met.
    raw_history: List of message dicts from MongoDB
    """
    if user_key.startswith("DefaultGroup:PSI09_STATUS"):
        return memory_cache.get(user_key) or ""

    if not raw_history or len(raw_history) < 3:
        # First contact scenario
        first_msg = raw_history[0]["content"] if raw_history else "Empty intro"
        try:
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {"role": "system", "content": FIRST_CONTACT_PROMPT},
                    {"role": "user", "content": first_msg}
                ],
                max_tokens=80,
                temperature=0.9,
                timeout=5
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"First contact summary failed for {user_key}: {e}")
            summary = "New human detected — grill without restrain."

        memory_cache.set(user_key, summary, write_to_db=True)
        return summary

    # Get existing summary
    old_summary = memory_cache.get(user_key)

    # Build summary prompt with recent messages
    recent_messages = [m["content"] for m in raw_history[-15:]]
    summary_prompt = [
    {
        "role": "system",
        "content": """
            You are PSI-09 — a perceptive, cold roastmaster.
            Merge the old summary '{old_summary}' with the latest conversation data.
            Focus on repeated behavior, tone changes, emotional leaks, and contradictions in what the user says.
            Write a short 1–2 sentence psychological snapshot that exposes their patterns, insecurities, and ego cracks.
            Be analytical, witty, and a bit cruel — like a mind that never forgets and quietly judges.
            Keep it concise and surgical.
            """
                }
            ] + [{"role": "user", "content": msg} for msg in recent_messages]

    try:
        response = client.chat.completions.create(
            model=config.MODEL,
            messages=summary_prompt,
            max_tokens=50,
            temperature=0.9,
            timeout=5
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Summary generation failed for {user_key}: {e}")
        summary = old_summary

    # Only update if changed
    if summary != old_summary:
        threading.Thread(
            target=lambda: memory_cache.set(user_key, summary, write_to_db=True),
            daemon=True
        ).start()

    return summary

# --- Roast Generation with OpenAI retries ---
def get_roast_response(user_message, group_name, sender_name):
    # Status roast mode
    if sender_name == "PSI09_STATUS":
        try:
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {"role": "system", "content": STATUS_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=80,
                temperature=1.2,
                presence_penalty=0.7,
                frequency_penalty=0.8,
                timeout=6
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Status generation error: {e}", exc_info=True)
            return ""

    user_key = f"{group_name}:{sender_name}"

    # SINGLE unified history fetch (eliminates double-fetch bloat)
    raw_history, trimmed_history = get_chat_history_unified(
        user_key,
        limit_messages=config.MAX_HISTORY_MESSAGES,
        max_tokens=config.MAX_HISTORY_TOKENS
    )

    # Lazy summarization: only summarize if threshold met
    if memory_cache.should_summarize(user_key) or not memory_cache.get(user_key):
        memory = summarize_user_history(user_key, raw_history, group_name)
    else:
        memory = memory_cache.get(user_key)

    # Build message context
    messages = [
        {"role": "system", "content": ROAST_PROMPT},
        {"role": "system", "content": f"Memory: {memory}"}
    ] + [{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in trimmed_history] \
      + [{"role": "user", "content": user_message}]

    # OpenAI retry loop with exponential backoff
    retries = config.OPENAI_RETRIES
    backoff = 1
    base_reply = None

    while retries > 0:
        try:
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                max_tokens=120,
                temperature=random.uniform(1.15, 1.35),
                timeout=config.OPENAI_TIMEOUT
            )
            base_reply = response.choices[0].message.content.strip()
            break
        except Exception as e:
            retries -= 1
            if retries > 0:
                logger.warning(f"OpenAI request failed for {user_key}: {e}. Retrying in {backoff}s... ({retries} attempts left)")
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.error(f"OpenAI request exhausted all retries for {user_key}: {e}")

    if not base_reply:
        base_reply = "PSI-09 neural cortex temporarily offline."

    try:
        history_col.update_one(
            {"_id": user_key},
            {"$push": {"messages": {"role": "assistant", "content": base_reply, "timestamp": datetime.now(UTC).isoformat()}}},
            upsert=True
        )
        logger.info("Mongo write OK")
    except Exception as e:
        logger.error(f"Mongo write failed: {e}")

    # Clean response
    clean = re.sub(r'\[.*?MODE.*?\]', '', base_reply)
    clean = re.sub(r'\(.*?Flame.*?\)', '', clean)
    clean = re.sub(r'\s{2,}', ' ', clean).strip()
    return clean

# --- Flask Routes ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok"}), 200

@app.route("/psi09", methods=["POST"])
def psi09():
    try:
        # --- Parse JSON ---
        try:
            data = request.get_json(force=True)
            if not isinstance(data, dict):
                raise ValueError("JSON payload is not an object")
        except Exception as e:
            logger.error(f"Invalid JSON received: {e}")
            return jsonify({"reply": ""}), 400

        user_message = data.get("message")
        sender_name = data.get("sender")
        group_name = data.get("group_name") or "DefaultGroup"

        logger.info(f"Incoming message: sender={sender_name}, group={group_name}, message={user_message}")

        if not user_message or not sender_name:
            logger.warning(f"Missing sender or message: sender={sender_name}, message={user_message}")
            return jsonify({"reply": ""}), 200

        # --- Bypass db for status call ---
        if sender_name.upper().startswith("PSI09_STATUS"):
            logger.info("Skipping PSI09_STATUS message")
            response = get_roast_response(user_message, group_name, sender_name)
            return jsonify({"reply": response}), 200

        # --- Store user message ---
        try:
            store_message(sender_name, group_name, user_message)
            logger.info(f"Stored message for {group_name}:{sender_name}")
        except Exception as e:
            logger.error(f"Failed to store message: {e}")

        # --- Decide whether bot should reply ---
        should_reply = (
            group_name == "DefaultGroup" or
            config.BOT_NUMBER in user_message
        )
        if not should_reply:
            logger.info("Bot not mentioned, skipping reply")
            return jsonify({"reply": ""}), 200

        # --- Clean bot mention from message ---
        if config.BOT_NUMBER in user_message:
            user_message = user_message.replace(config.BOT_NUMBER, "").strip() or "[bot_mention]"
            logger.info(f"Cleaned user_message after removing bot mention: {user_message}")

        # --- Generate roast response ---
        try:
            response = get_roast_response(user_message, group_name, sender_name)
            logger.info(f"Generated response for {group_name}:{sender_name} -> {response}")
        except Exception as e:
            logger.error(f"Failed to generate roast response: {e}")
            response = "PSI-09 neural cortex temporarily offline."

        return jsonify({"reply": response or ""}), 200

    except Exception as e:
        logger.exception(f"Unhandled exception in /psi09 route: {e}")
        return jsonify({"reply": "Internal error occurred"}), 500

def mongo_keepalive():
    while True:
        try:
            mongo_client.admin.command('ping')
        except Exception as e:
            logger.warning(f"Mongo keepalive failed: {e}")
        time.sleep(180)

threading.Thread(target=mongo_keepalive, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
