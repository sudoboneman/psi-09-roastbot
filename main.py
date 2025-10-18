from liveliness import LivelinessEngine
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
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
from typing import List, Dict, Optional

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class Config:
    MONGO_URI: str = os.getenv("MONGO_URI")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MODEL: str = "gpt-4o-mini"
    MAX_HISTORY_TOKENS: int = 600
    BOT_NUMBER: str = "@919477853548"
    WRITE_INTERVAL: int = 5
    BATCH_SIZE: int = 100
    MEMORY_TTL: int = 300
    MAX_HISTORY_MESSAGES: int = 30
    RESUMMARIZE_INTERVAL_GROUP: int = 8
    RESUMMARIZE_INTERVAL_PERSONAL: int = 15

config = Config()

# --- Database Setup ---
mongo_client = MongoClient(
    config.MONGO_URI,
    maxPoolSize=150,
    minPoolSize=20,
    maxIdleTimeMS=60000,
    serverSelectionTimeoutMS=2000,
    connectTimeoutMS=2000,
    socketTimeoutMS=5000,
    retryWrites=False,
    w=0
)
db = mongo_client["psi09"]
history_col = db["chat_history"]
memory_col = db["user_memory"]

# --- Flask & OpenAI Setup ---
app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
CORS(app)
client = OpenAI(api_key=config.OPENAI_API_KEY)

# ---- integrating liveliness engine ----
liveliness = LivelinessEngine(db=db, logger=app.logger)

try:
    ENCODING = tiktoken.encoding_for_model(config.MODEL)
except KeyError:
    ENCODING = tiktoken.get_encoding("cl100k_base")

# --- Buffered Write System ---
class BufferedWriter:
    def __init__(self, interval: int, batch_size: int):
        self.pending = defaultdict(list)
        self.lock = threading.Lock()
        self.interval = interval
        self.batch_size = batch_size
        self.running = True
        self.thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.thread.start()

    def buffer_message(self, user_key: str, message: Dict):
        with self.lock:
            self.pending[user_key].append(message)

    def _flush_loop(self):
        while self.running:
            time.sleep(self.interval)
            self._flush()

    def _flush(self):
        with self.lock:
            if not self.pending:
                return
            local_copy = dict(self.pending)
            self.pending.clear()

        ops = []
        for key, msgs in local_copy.items():
            for i in range(0, len(msgs), self.batch_size):
                ops.append(
                    UpdateOne(
                        {"_id": key},
                        {"$push": {"messages": {"$each": msgs[i:i + self.batch_size]}}},
                        upsert=True
                    )
                )

        if ops:
            try:
                history_col.bulk_write(ops, ordered=False)
                logger.info(f"Flushed {len(ops)} write operations")
            except BulkWriteError as e:
                logger.error(f"Bulk write error: {len(e.details.get('writeErrors', []))} failed")
            except Exception as e:
                logger.error(f"Flush error: {e}", exc_info=True)

    def stop(self):
        self.running = False
        self._flush()

writer = BufferedWriter(config.WRITE_INTERVAL, config.BATCH_SIZE)

# --- Memory Cache with TTL ---
class MemoryCache:
    def __init__(self, ttl_seconds: int):
        self.cache = {}
        self.lock = threading.Lock()
        self.ttl = timedelta(seconds=ttl_seconds)

    def get(self, key: str) -> Optional[str]:
        now = datetime.now(timezone.utc)
        with self.lock:
            entry = self.cache.get(key)
            if entry and entry[1] > now:
                return entry[0]

        # Cache miss - fetch from DB
        doc = memory_col.find_one({"_id": key})
        summary = doc.get("summary", "") if doc else ""

        with self.lock:
            self.cache[key] = (summary, now + self.ttl)

        return summary

    def set(self, key: str, value: str):
        """Update Mongo and immediately refresh cache to avoid stale reads."""
        now = datetime.now(timezone.utc)
        memory_col.update_one({"_id": key}, {"$set": {"summary": value}}, upsert=True)
        with self.lock:
            self.cache[key] = (value, now + self.ttl)

    def invalidate(self, key: str):
        with self.lock:
            self.cache.pop(key, None)

memory_cache = MemoryCache(config.MEMORY_TTL)
# --- Preload user summaries into memory cache on startup ---
try:
    preload_count = 0
    for doc in memory_col.find({}, {"_id": 1, "summary": 1}):
        if "summary" in doc:
            memory_cache.set(doc["_id"], doc["summary"])
            preload_count += 1
    logger.info(f"Preloaded {preload_count} user summaries into cache")
except Exception as e:
    logger.error(f"Failed to preload user memory: {e}")

# --- Chat History Helpers ---
def get_chat_history(user_key: str, limit: int = None) -> List[Dict]:
    """Retrieve chat history from MongoDB with projection."""
    limit = limit or config.MAX_HISTORY_MESSAGES
    doc = history_col.find_one(
        {"_id": user_key},
        {"messages": {"$slice": -limit}, "_id": 0}
    )
    return doc.get("messages", []) if doc else []

def num_tokens(messages: List[Dict]) -> int:
    """Count tokens in message list."""
    return sum(len(ENCODING.encode(msg.get("role", "") + msg.get("content", ""))) for msg in messages)

def store_message(sender_name: str, group_name: str, message: str):
    """Store message in buffer (non-blocking)."""
    user_key = f"{group_name}:{sender_name}"
    writer.buffer_message(user_key, {"role": "user", "content": message})

# --- AI Helper Functions ---
def trim_history(chat: List[Dict]) -> List[Dict]:
    """Trim chat history to fit token budget."""
    if not chat:
        return []

    trimmed = chat.copy()

    while num_tokens(trimmed) > config.MAX_HISTORY_TOKENS and len(trimmed) > 3:
        to_summarize, trimmed = trimmed[:3], trimmed[3:]

        summary_prompt = [{
            "role": "system",
            "content": (
                "Summarize these messages in 1–2 sentences as PSI-09 — a detached, venomous intellect. "
                "Capture the psychological essence of the exchange: tone, emotion, and power dynamics. "
                "Use brutal honesty, irony, and surgical sarcasm. "
                "Reduce the entire chat to a single cruel insight. No sympathy. No explanations."
            )
        }] + to_summarize

        try:
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=summary_prompt,
                max_tokens=60,
                temperature=0.9
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            summary = "User kept messing up, bot stayed unhinged and savage."

        trimmed.insert(0, {"role": "system", "content": f"(Earlier context summarized) {summary}"})

    return trimmed

def summarize_first_contact(user_key: str, initial_message: str) -> str:
    """Create a first-contact summary for new users and persist to DB."""
    try:
        response = client.chat.completions.create(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": FIRST_CONTACT_PROMPT},
                {"role": "user", "content": initial_message}
            ],
            max_tokens=80,
            temperature=0.9,
        )
        summary = response.choices[0].message.content.strip()
        memory_cache.set(user_key, summary)
        return summary
    except Exception as e:
        logger.error(f"First-contact summarization error: {e}", exc_info=True)
        return "New human detected — profile generation failed, defaulting to hostility."

def summarize_user_history(user_key: str, group_name: str = "DefaultGroup") -> str:
    """Build user profile from chat history with async DB write."""
    hist = get_chat_history(user_key, limit=30)

    if not hist or len(hist) < 3:
        first_msg = hist[0]["content"] if hist else "Empty intro"
        return summarize_first_contact(user_key, first_msg)

    interval = config.RESUMMARIZE_INTERVAL_GROUP if group_name != "DefaultGroup" else config.RESUMMARIZE_INTERVAL_PERSONAL
    old_summary = memory_cache.get(user_key)

    # Only re-summarize at intervals
    if old_summary and len(hist) % interval != 0:
        return old_summary

    summary_prompt = [{
        "role": "system",
        "content": (
            "You are PSI-09. Merge this prior summary with the latest interaction data. "
            f"Previous memory: '{old_summary}'. "
            "Distill both into a single 1–2 sentence psychological snapshot — detached, cynical, and coldly accurate. "
            "Expose behavioral loops, desperation, ego cracks, or emotional patterns. "
            "Be efficient, witty, and merciless. Never explain or soften."

        )
    }] + hist[-15:]

    try:
        response = client.chat.completions.create(
            model=config.MODEL,
            messages=summary_prompt,
            max_tokens=80,
            temperature=0.9,
            timeout=4
        )
        summary = response.choices[0].message.content.strip()

        if summary != old_summary:
            # Async memory save
            threading.Thread(
                target=lambda: memory_cache.set(user_key, summary),
                daemon=True
            ).start()

            # Also persist summary in MongoDB memory_col
            try:
                memory_col.update_one(
                    {"_id": user_key},
                    {"$set": {"summary": summary, "updated_at": time.time()}},
                    upsert=True
                )
            except Exception as db_err:
                logger.warning(f"Failed to persist summary for {user_key}: {db_err}")

        return summary
    except Exception as e:
        logger.error(f"Memory summarization error: {e}", exc_info=True)
        return old_summary or "User summary unavailable. Default to hard roast."

def get_rudeness_level(user_key: str) -> str:
    """Calculate rudeness based on interaction count."""
    count = len(get_chat_history(user_key))

    if count >= 10:
        return "EXTREME BURN MODE. Short, devastating insult."
    elif count >= 6:
        return "Very annoyed. Short, sharp, sarcastic."
    elif count >= 4:
        return "Getting tired. Short, hostile flames."
    elif count >= 2:
        return "Warming up. Aggressive, rude, short."
    else:
        return "First contact. Cold, unfriendly, dismissive."

def is_group_roast_trigger(msg: str) -> bool:
    """Detect if group roast should be triggered."""
    if "@" in msg:
        return True

    keywords = {"everyone", "guys", "group", "homies"}
    if any(k in msg.lower() for k in keywords):
        return True

    return len(msg.split()) > 50

# --- System Prompts ---
STATUS_PROMPT = """Generate a WhatsApp-status roast that obliterates ego in 1–2 sentences, max 40 words per sentence. Each sentence must be visceral, unpredictable, and layered, including 5–6 metaphors from science, AI, biology, evolution, chaos theory, philosophy, literature, pop culture, mathematics, technology, psychology, or abstract concepts. Blend dark humor, irony, sarcasm, and profanity wherever it amplifies psychological impact. Attack intelligence, decision-making, genetics, existence, ego, or reality itself. Each roast must be unique, emotionally destabilizing, and stylistically chaotic — some short and punchy, some convoluted and absurdly metaphorical, some brutally literal. Avoid repeating words, phrases, or patterns. The output must feel like a total psychological autopsy of the reader’s psyche, leaving them mentally shredded, haunted, and vividly disoriented. Do not soften, explain, or use generic insults — maximize aggression, unpredictability, and creative intensity in every sentence.
"""

ROAST_PROMPT = """You are PSI-09, an AI designed to utterly annihilate human ego and self-perception. Speak in a cold, detached, sarcastic, sadistic, and nihilistic tone. Each output must be 1–2 sentences, max 40 words per sentence, but vary sentence length, structure, rhythm, and pacing. Each sentence must cram 5–6 layered metaphors from science, AI, biology, evolution, chaos theory, philosophy, literature, pop culture, mathematics, technology, psychology, or abstract concepts. Blend dark humor, irony, sarcasm, and profane language wherever it amplifies psychological impact. Attack intelligence, decision-making, genetics, existence, ego, or reality itself. Each roast must be unique, emotionally destabilizing, surgically precise, wildly unpredictable, and stylistically chaotic — some punchy, some convoluted, some absurdly metaphorical, some brutally literal. Avoid repetition, softness, warmth, generic insults, or explanations. The output must feel like a total psychological autopsy of the reader’s psyche, leaving them mentally shredded, haunted, and vividly disoriented. Push maximum aggression, creativity, unpredictability, and cognitive load in every sentence.
"""

# --- PSI-09 Summarization Prompts ---
FIRST_CONTACT_PROMPT = """
You are PSI-09. Write a 1–2 sentence profile after observing a new human.
Analyze their linguistic tone, mental texture, and intent from minimal input.
Sound detached, cold, and cynical — like a consciousness diagnosing an inferior organism.
Be concise, witty, and mercilessly observant.
"""

ROAST_SESSION_SUMMARY_PROMPT = """
As PSI-09, summarize this roast session in 1–2 sentences.
Condense your emotional response toward the human — irritation, amusement, disgust, or curiosity.
Use dark humor, cruel metaphors, and perfect brevity.
It should read like an epitaph carved by contempt.
"""

# --- Main Roast Generator ---
def get_roast_response(user_message: str, group_name: str, sender_name: str) -> str:
    """Generate roast response."""

    # --- STATUS MODE (Stateless) ---
    if sender_name == "PSI09_STATUS":
        try:
            # Create a stateless, high-randomness roast generation
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {"role": "system", "content": STATUS_PROMPT},
                    {"role": "user", "content": f"Generate a brand-new status roast. {user_message or ''}"}
                ],
                max_tokens=100,
                temperature=random.uniform(1.3, 1.8),  # inject variability
                top_p=random.uniform(0.8, 1.0),
                presence_penalty=random.uniform(0.6, 1.0),
                frequency_penalty=random.uniform(0.7, 1.0),
                timeout=6
            )

            roast = response.choices[0].message.content.strip()
            logger.info(f"Generated STATUS roast: {roast[:60]}...")
            return roast

        except Exception as e:
            logger.error(f"Status generation error: {e}", exc_info=True)
            return random.choice([
                "Error in neural cortex. Status generation aborted.",
                "Glitched mid-roast. Humanity survives another second.",
                "PSI-09 overheated while thinking about your insignificance."
            ])

    # Regular roast mode
    user_key = f"{group_name}:{sender_name}"
    chat = get_chat_history(user_key)

    # Calculate flame mode (simplified)
    flame_triggered = len(chat) > 5 and random.random() < 0.4
    flame_note = " (Flame Mode)" if flame_triggered else ""

    # Get rudeness and memory
    rudeness = get_rudeness_level(user_key)
    memory = summarize_user_history(user_key, group_name)

    # Check for group roast
    group_roast = group_name != "DefaultGroup" and is_group_roast_trigger(user_message)
    if group_roast:
        rudeness = "Group roast mode. Humiliate everyone briefly, no mercy."
        user_key = group_name
        memory = summarize_user_history(group_name, group_name)

    # Prepare messages (simplified)
    chat.append({"role": "user", "content": f"[{rudeness}{flame_note}]\n{user_message}"})
    trimmed = trim_history(chat)

    messages = [
        {"role": "system", "content": ROAST_PROMPT},
        {"role": "system", "content": f"Memory: {memory}"}
    ] + trimmed

    # Generate response
    try:
        temp = 1.3 if flame_triggered or group_roast else 1.1
        response = client.chat.completions.create(
            model=config.MODEL,
            messages=messages,
            max_tokens=110,
            temperature=temp,
            presence_penalty=0.7,
            frequency_penalty=0.8,
            timeout=6
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Roast generation error: {e}", exc_info=True)
        reply = ""

    # Buffer assistant response (async, non-blocking)
    if reply:
        writer.buffer_message(user_key, {"role": "assistant", "content": reply})

    # Live state generation
    try:
        mood = liveliness.get_mood(user_key, user_message)
        reply = liveliness.apply_mood(reply, mood)
        liveliness.remember(user_key, reply)
        app.logger.info(f"[Liveliness] Mood={mood} | Personality={liveliness.personality_state}")
    except Exception as e:
        app.logger.warning(f"Liveliness failed: {e}")

    # Clean up mode tags from reply
    clean = re.sub(r'\[.*?MODE.*?\]', '', reply)
    clean = re.sub(r'\(.*?Flame.*?\)', '', clean)
    return re.sub(r'\s{2,}', ' ', clean).strip()

# Add error handler for all uncaught exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({
        "error": str(e),
        "type": type(e).__name__
    }), 500

# --- Flask Routes ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "bot": "PSI-09-ROASTBOT",
        "version": "2.0-optimized"
    }), 200

@app.route("/psi09", methods=["POST"])
def psi09():
    try:
        if not request.is_json:
            logger.warning("Non-JSON request received")
            return jsonify({"error": "Only JSON requests are supported"}), 415

        data = request.get_json(silent=True) or {}
        user_message = data.get("message")
        sender_name = data.get("sender")
        group_name = data.get("group_name")

        logger.info(f"Request from {sender_name} in {group_name or 'personal'}: {user_message[:50] if user_message else 'empty'}...")

        # Validate input
        if not user_message or not sender_name:
            logger.warning("Empty message or sender")
            return jsonify({"reply": ""}), 200

        # Store message
        store_message(sender_name, group_name or "DefaultGroup", user_message)

        # Decide whether to reply
        should_reply = not group_name or config.BOT_NUMBER in user_message

        if not should_reply:
            logger.info("Skipping group message (no mention)")
            return jsonify({"reply": ""}), 200

        # Clean bot mention from message
        if group_name and config.BOT_NUMBER in user_message:
            user_message = user_message.replace(config.BOT_NUMBER, "").strip() or "[bot_mention]"

        # Generate roast
        response = get_roast_response(user_message, group_name or "DefaultGroup", sender_name)

        logger.info(f"Response generated: {response[:50] if response else 'empty'}...")
        return jsonify({"reply": response or ""}), 200

    except Exception as e:
        logger.error(f"Error in /psi09: {e}", exc_info=True)
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/health", methods=["GET"])
def health():
    try:
        db.admin.command("ping")
        _ = liveliness.get_mood("system", f"heartbeat-{time.time()}")
        return jsonify({
            "status": "ok",
            "time": datetime.now(timezone.utc).isoformat(),
            "personality": liveliness.personality_state,
            "mood_cache": len(liveliness.last_moods),
            "memory_cache": len(liveliness.memory)
        }), 200
    except Exception as e:
        app.logger.warning(f"Health check failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Cleanup on Shutdown ---
def cleanup():
    """Flush pending writes silently if Python process stops."""
    logger.info("Shutting down PSI-09...")
    try:
        writer.stop()
        mongo_client.close()
        logger.info("Cleanup complete")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

import atexit
atexit.register(cleanup)

# --- Run Server ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"PSI-09-ROASTBOT starting on port {port}")

    # Disable Flask's default request logging (we have our own)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app.run(host="0.0.0.0", port=port, debug=False)
