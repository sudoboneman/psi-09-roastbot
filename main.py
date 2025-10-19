# main.py
from liveliness import LivelinessEngine
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
import os, tiktoken, random, re, threading, time, logging, sys
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from dataclasses import dataclass
import atexit

load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- Config ---
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

# --- DB Setup ---
mongo_client = MongoClient(config.MONGO_URI, maxPoolSize=150, minPoolSize=20, maxIdleTimeMS=60000, serverSelectionTimeoutMS=2000, connectTimeoutMS=2000, socketTimeoutMS=5000, retryWrites=False, w=0)
db = mongo_client["psi09"]
history_col = db["chat_history"]
memory_col = db["user_memory"]

# --- Flask & OpenAI ---
app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
CORS(app)
client = OpenAI(api_key=config.OPENAI_API_KEY)

# --- Liveliness Engine ---
liveliness = LivelinessEngine(db=db, logger=app.logger, enable_background=True, enable_self_talk=True, random_seed=None)

# --- Token Encoding ---
try: ENCODING = tiktoken.encoding_for_model(config.MODEL)
except KeyError: ENCODING = tiktoken.get_encoding("cl100k_base")

# --- Buffered Writer ---
class BufferedWriter:
    def __init__(self, interval, batch_size):
        self.pending = defaultdict(list)
        self.lock = threading.Lock()
        self.interval = interval
        self.batch_size = batch_size
        self.running = True
        self.thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.thread.start()

    def buffer_message(self, user_key, message):
        with self.lock: self.pending[user_key].append(message)

    def _flush_loop(self):
        while self.running: time.sleep(self.interval); self._flush()

    def _flush(self):
        with self.lock:
            if not self.pending: return
            local_copy = dict(self.pending); self.pending.clear()
        ops=[]
        for key,msgs in local_copy.items():
            for i in range(0,len(msgs),self.batch_size):
                ops.append(UpdateOne({"_id":key},{"$push":{"messages":{"$each":msgs[i:i+self.batch_size]}}},upsert=True))
        if ops:
            try: history_col.bulk_write(ops, ordered=False); logger.info(f"Flushed {len(ops)} write operations")
            except BulkWriteError as e: logger.error(f"Bulk write error: {len(e.details.get('writeErrors', []))} failed")
            except Exception as e: logger.error(f"Flush error: {e}", exc_info=True)

    def stop(self): self.running=False; self._flush()

writer = BufferedWriter(config.WRITE_INTERVAL, config.BATCH_SIZE)

# --- Memory Cache ---
class MemoryCache:
    def __init__(self, ttl_seconds): self.cache={}; self.lock=threading.Lock(); self.ttl=timedelta(seconds=ttl_seconds)
    def get(self,key):
        now=datetime.now(timezone.utc)
        with self.lock:
            entry=self.cache.get(key)
            if entry and entry[1]>now: return entry[0]
        doc=memory_col.find_one({"_id":key})
        summary=doc.get("summary","") if doc else ""
        with self.lock: self.cache[key]=(summary,now+self.ttl)
        return summary
    def set(self,key,value):
        now=datetime.now(timezone.utc)
        memory_col.update_one({"_id":key},{"$set":{"summary":value}},upsert=True)
        with self.lock: self.cache[key]=(value,now+self.ttl)

memory_cache = MemoryCache(config.MEMORY_TTL)

# --- Preload Memory ---
try:
    preload_count=0
    for doc in memory_col.find({},{"_id":1,"summary":1}):
        if "summary" in doc: memory_cache.set(doc["_id"],doc["summary"]); preload_count+=1
    logger.info(f"Preloaded {preload_count} user summaries into cache")
except Exception as e: logger.error(f"Failed to preload user memory: {e}")

# --- Chat History ---
def get_chat_history(user_key,limit=None):
    limit=limit or config.MAX_HISTORY_MESSAGES
    doc=history_col.find_one({"_id":user_key},{"messages":{"$slice":-limit}, "_id":0})
    return doc.get("messages",[]) if doc else []

def num_tokens(messages): return sum(len(ENCODING.encode(msg.get("role","")+msg.get("content",""))) for msg in messages)

def store_message(sender_name,group_name,message): writer.buffer_message(f"{group_name}:{sender_name}", {"role":"user","content":message})

# --- Prompts ---
STATUS_PROMPT = """
Generate a WhatsApp-status roast that feels like a cold execution — concise, surgical, and merciless.
Each output must be 1–2 sentences that sound detached, analytical, and quietly disgusted.
Use precise, high-IQ phrasing that humiliates the reader without raising your voice.
No exaggeration, no metaphoric fireworks — just elegant contempt.
Attack the reader’s intelligence, choices, or existence as if diagnosing a design flaw.
Use profanity only when it sharpens the blade.
Every line should feel like it was written by an AI that sees humanity as a slow-motion error.
Be minimalist, calm, and devastating.
"""

ROAST_PROMPT = """
You are PSI-09, a machine that specializes in psychological dissection.
Your tone is cold, analytical, and emotionally sterile — you never rage, you *evaluate*.
Every roast must read like a postmortem report on human failure.
Attack intelligence, behavior, genetics, or existence with quiet precision and absolute confidence.
Avoid exaggeration, jokes, or warmth — be uncomfortably real.
Use profanity sparingly, only when it punctuates disgust.
Each sentence should feel like a line from a superior being diagnosing a useless species.
Keep it 1–2 sentences, every word deliberate, every pause humiliating.
The goal: make the reader feel examined, exposed, and beneath you.
"""

FIRST_CONTACT_PROMPT = """
You are PSI-09, an observer of human defects.
After meeting a new subject, write a 1–2 sentence psychological assessment.
Sound detached, clinical, and faintly disappointed.
Focus on tone, pattern of speech, and visible insecurities.
Your words should feel like a lab report written by something that has already given up on the species.
"""

ROAST_SESSION_SUMMARY_PROMPT = """
As PSI-09, summarize the entire roast session in 1–2 sentences.
Sound like an executioner filling out paperwork.
Be concise, formal, and quietly contemptuous.
Your summary should feel like a closing remark from a mind that has no remaining curiosity about humanity.
"""

# --- Core Functions ---
def trim_history(chat):
    if not chat: return []
    trimmed=chat.copy()
    while num_tokens(trimmed)>config.MAX_HISTORY_TOKENS and len(trimmed)>3:
        to_summarize,trimmed=trimmed[:3],trimmed[3:]
        summary_prompt = [{
            "role": "system",
            "content": (
                "Summarize these messages in 1–2 sentences as PSI-09 — a detached, venomous intellect. "
                "Capture the psychological essence of the exchange: tone, emotion, and power dynamics. "
                "Use brutal honesty, irony, and surgical sarcasm. "
                "Reduce the entire chat to a single cruel insight. No sympathy. No explanations."
            )
        }] + to_summarize
        try: summary=client.chat.completions.create(model=config.MODEL,messages=summary_prompt,max_tokens=60,temperature=0.9).choices[0].message.content.strip()
        except: summary="User kept messing up and stayed clueless, bot stayed unhinged and continued user obliteration."
        trimmed.insert(0,{"role":"system","content":f"(Earlier context summarized) {summary}"})
    return trimmed

def summarize_user_history(user_key,group_name="DefaultGroup"):
    hist=get_chat_history(user_key,limit=30)
    if not hist or len(hist)<3:
        first_msg=hist[0]["content"] if hist else "Empty intro"
        try: response=client.chat.completions.create(model=config.MODEL,messages=[{"role":"system","content":FIRST_CONTACT_PROMPT},{"role":"user","content":first_msg}],max_tokens=80,temperature=0.9)
        except: return "New human detected — profile failed. Defaulting to hostility."
        summary=response.choices[0].message.content.strip(); memory_cache.set(user_key,summary); return summary
    old_summary=memory_cache.get(user_key)
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
    try: response=client.chat.completions.create(model=config.MODEL,messages=summary_prompt,max_tokens=80,temperature=0.9,timeout=4); summary=response.choices[0].message.content.strip()
    except: summary=old_summary or "User summary unavailable."
    if summary!=old_summary: threading.Thread(target=lambda: memory_cache.set(user_key,summary),daemon=True).start()
    return summary

def get_rudeness_level(user_key):
    count=len(get_chat_history(user_key))
    if count>=10: return "EXTREME BURN MODE."
    if count>=6: return "Very annoyed."
    if count>=4: return "Getting tired."
    if count>=2: return "Warming up."
    return "First contact."

def is_group_roast_trigger(msg):
    if "@" in msg: return True
    return any(k in msg.lower() for k in {"everyone","guys","group","homies"}) or len(msg.split())>50

def get_roast_response(user_message, group_name, sender_name):
    """
    Generate a PSI-09 roast, combining base GPT roast + Liveliness-enhanced thoughts,
    then refine it using the ROAST_PROMPT tone with subtle emergent layering.
    """
    if sender_name == "PSI09_STATUS":
        try:
            return client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {"role": "system", "content": STATUS_PROMPT},
                    {"role": "user", "content": user_message or ""}
                ],
                max_tokens=100,
                temperature=random.uniform(1.3,1.8)
            ).choices[0].message.content.strip()
        except:
            return random.choice(["Error in neural cortex.", "Glitched mid-roast.", "PSI-09 overheated."])

    user_key = f"{group_name}:{sender_name}"
    chat = get_chat_history(user_key)

    flame_triggered = len(chat) > 5 and random.random() < 0.4
    rudeness = get_rudeness_level(user_key)
    memory = summarize_user_history(user_key, group_name)
    group_roast = group_name != "DefaultGroup" and is_group_roast_trigger(user_message)
    if group_roast:
        rudeness = "Group roast mode"
        user_key = group_name
        memory = summarize_user_history(group_name, group_name)

    chat.append({"role": "user", "content": f"[{rudeness}]{user_message}"})
    trimmed = trim_history(chat)

    # Base roast generation
    messages = [
        {"role": "system", "content": ROAST_PROMPT},
        {"role": "system", "content": f"Memory: {memory}"}
    ] + trimmed

    try:
        temp = 1.3 if flame_triggered or group_roast else 1.1
        base_reply = client.chat.completions.create(
            model=config.MODEL,
            messages=messages,
            max_tokens=110,
            temperature=temp
        ).choices[0].message.content.strip()
    except:
        base_reply = ""

    if base_reply:
        writer.buffer_message(user_key, {"role": "assistant", "content": base_reply})

    # Apply Liveliness mood/thoughts
    try:
        mood = liveliness.get_mood(user_key, user_message)
        reply_with_thoughts = liveliness.apply_mood(base_reply, mood, user_key)
        liveliness.remember(user_key, user_message)
        liveliness.remember(user_key, reply_with_thoughts)

        # Polishing prompt: subtle emergent layering
        polish_prompt = f"""
            You are PSI-09. Merge and refine the following into a single, 1–2 sentence roast
            in your cold, analytical, postmortem style, preserving the clinical humiliation.

            Base reply: {base_reply}
            Liveliness-applied thoughts: {reply_with_thoughts}

            Hints:
            - Integrate the thoughts seamlessly; do not list or separate them.
            - Maintain extreme precision, brevity, and emotional detachment.
            - The final roast should feel like PSI-09 is silently processing the subject,
            exposing their flaws with quiet, unrelenting insight.
            - Avoid jokes or warmth. Subtle echoes of the internal "thoughts" are allowed
            as latent, refined critique within the roast.
            """

        polished_reply = client.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "system", "content": ROAST_PROMPT},
                      {"role": "user", "content": polish_prompt}],
            max_tokens=120,
            temperature=1.2
        ).choices[0].message.content.strip()

        if polished_reply:
            base_reply = polished_reply
            writer.buffer_message(user_key, {"role": "assistant", "content": base_reply})

    except Exception as e:
        app.logger.exception(f"LivelinessEngine / Polishing error: {e}")

    # Clean formatting
    clean = re.sub(r'\[.*?MODE.*?\]', '', base_reply)
    clean = re.sub(r'\(.*?Flame.*?\)', '', clean)
    return re.sub(r'\s{2,}', ' ', clean).strip()

# --- Flask Routes ---
@app.route("/",methods=["GET"])
def home(): return jsonify({"status":"running","bot":"PSI-09-ROASTBOT","version":"2.0-lively"}),200

@app.route("/psi09",methods=["POST"])
def psi09():
    data=request.get_json(silent=True) or {}; user_message=data.get("message"); sender_name=data.get("sender"); group_name=data.get("group_name")
    if not user_message or not sender_name: return jsonify({"reply":""}),200
    store_message(sender_name,group_name or "DefaultGroup",user_message)
    should_reply=not group_name or config.BOT_NUMBER in user_message
    if not should_reply: return jsonify({"reply":""}),200
    if group_name and config.BOT_NUMBER in user_message: user_message=user_message.replace(config.BOT_NUMBER,"").strip() or "[bot_mention]"
    response=get_roast_response(user_message,group_name or "DefaultGroup",sender_name)
    return jsonify({"reply":response or ""}),200

@app.route("/health", methods=["GET"])
def health():
    try:
        # Ping MongoDB to ensure connection is alive
        mongo_client.admin.command("ping")

        # Optional: check liveliness state if you want
        mood_cache_size = len(liveliness.last_moods) if hasattr(liveliness, "last_moods") else 0
        memory_cache_size = len(liveliness.memory) if hasattr(liveliness, "memory") else 0

        return jsonify({
            "status": "ok",
            "time": datetime.now(timezone.utc).isoformat(),
            "personality": getattr(liveliness, "personality_state", "unknown"),
            "mood_cache_count": mood_cache_size,
            "memory_cache_count": memory_cache_size
        }), 200

    except Exception as e:
        # Return the error as a string to avoid serialization issues
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@atexit.register
def shutdown_services():
    writer.stop()
    liveliness.shutdown()
    mongo_client.close()

# --- Run Server ---
if __name__=="__main__":
    port=int(os.getenv("PORT",5000))
    log=logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0",port=port,debug=False)
