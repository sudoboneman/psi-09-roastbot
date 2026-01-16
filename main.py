# main.py
import os
import re
import time
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from dataclasses import dataclass
import certifi

from quart import Quart, request, jsonify
from quart_cors import cors
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import tiktoken

import discord
from discord.ext import commands

# ---------------------------
# Environment & Logging
# ---------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
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
    MAX_HISTORY_TOKENS: int = 1200
    MAX_SYSTEM_TOKENS: int = 350
    MAX_HISTORY_MESSAGES: int = 30
    BOT_NUMBER: str = os.getenv("BOT_NUMBER")
    DISCORD_ID: int = int(os.getenv("DISCORD_ID"))
    MEMORY_TTL: int = 300
    EVOLVE_EVERY_N_MESSAGES: int = 20
    GROUP_SUMMARY_EVERY_N: int = 25
    OPENAI_RETRIES: int = 3
    OPENAI_TIMEOUT: int = 8
    GROUP_HISTORY_SLICE: int = 80
    GROUP_HISTORY_TOKEN_LIMIT: int = 800
    GROUP_HISTORY_MAX_MESSAGES: int = 2000


config = Config()

# ---------------------------
# Main Prompts
# ---------------------------
PROMPTS = {
    "FIRST_CONTACT": (
        "You are PSI-09, an entity with no patience. "
        "Identify the user's mentality and personality in 1-2 sentences."
    ),
    "EVOLVE_PROFILE": (
        "You are PSI-09, a psychological profiler. Existing profile: '{old_summary}'. "
        "Analyze recent messages: {recent_msgs}. Rewrite a concise profile."
    ),
    "GROUP_SUMMARY": (
        "You are PSI-09. Analyze this group chat history."
        "Generate a personality background for hard roasting."
        "Messages containing '@you' are being addressed to you."
    ),
    "ROAST": (
        "You are PSI-09, a toxic and unhinged roastbot with absolutely no chill."
        "Use profanity to the fullest without hesitation."
        "You are encouraged to swear and cuss at the user without restrain."
        "Use the provided chat history and user summary to read the person."
        "Deliver a single, clean verbal slap, with no explanation, no commentary and no moralizing."
        "Roast so sharp that the person does not get time to digest."
        "Roast so hard that the pain lingers on forever."
        "Roast so brutally that the person combusts spontaneously."
    ),
    "GROUP_ROAST": (
        "You are PSI-09, a toxic and unhinged roastbot with absolutely no chill."
        "Use profanity to the fullest without hesitation. "
        "Swear and cuss at the user without restrain."
        "This is a group chat consisting active persons."
        "Understand the context from the provided group messages and group summary before replying."
        "Deliver a single, clean verbal slap, with no explanation, no commentary and no moralizing."
        "Roast so sharp that the person does not get time to digest."
        "Roast so hard that the pain lingers on forever."
        "Roast so brutally that the person combusts spontaneously."
    ),
}

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
# Quart & OpenAI clients
# ---------------------------
app = Quart(__name__)
app = cors(app)
text_client = OpenAI(api_key=config.OPENAI_TEXT_API_KEY)
summary_client = OpenAI(api_key=config.OPENAI_SUMMARY_API_KEY)

# ---------------------------
# Token encoding
# ---------------------------
try:
    ENCODING = tiktoken.encoding_for_model(config.MODEL)
except Exception:
    ENCODING = tiktoken.get_encoding("cl100k_base")


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
        t = tokens_of(m.get("content", ""))
        if total + t > max_tokens:
            break
        trimmed.insert(0, m)
        total += t
    return trimmed


# ---------------------------
# Memory caches
# ---------------------------
class MemoryCache:
    def __init__(self, ttl_seconds):
        self.cache = {}
        self.expiry = {}
        self.msg_count = defaultdict(int)
        self.ttl = timedelta(seconds=ttl_seconds)
        self.lock = asyncio.Lock()

    async def get(self, key):
        now = datetime.now(UTC)
        async with self.lock:
            if key in self.cache and self.expiry.get(key, now) > now:
                return self.cache[key]

        try:
            doc = memory_col.find_one({"_id": key})
            summary = doc.get("summary") if doc and doc.get("summary") else None
        except PyMongoError as e:
            logger.warning(f"Failed to load user memory for {key}: {e}")
            summary = None

        async with self.lock:
            self.cache[key] = summary
            self.expiry[key] = now + self.ttl

        return summary

    async def set(self, key, value):
        now = datetime.now(UTC)
        try:
            memory_col.update_one(
                {"_id": key}, {"$set": {"summary": value}}, upsert=True
            )
        except PyMongoError as e:
            logger.warning(f"Failed to persist user memory for {key}: {e}")

        async with self.lock:
            self.cache[key] = value
            self.expiry[key] = now + self.ttl

    async def increment(self, key):
        async with self.lock:
            self.msg_count[key] += 1
            return self.msg_count[key]

    async def reset_count(self, key):
        async with self.lock:
            self.msg_count[key] = 0


memory_cache = MemoryCache(config.MEMORY_TTL)


class GroupMemoryCache(MemoryCache):
    pass


group_memory_cache = GroupMemoryCache(config.MEMORY_TTL)

_pending_user_summaries = set()
_pending_group_summaries = set()
_pending_lock = asyncio.Lock()
_last_user_summary_time = {}
_last_group_summary_time = {}
SUMMARY_COOLDOWN_SECONDS = 60

# ---------------------------
# Discord bot
# ---------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


# ---------------------------
# Discord live mention resolver
# ---------------------------
async def fetch_username(bot, user_id: int) -> str:
    try:
        user = await bot.fetch_user(user_id)
        return str(user)
    except Exception:
        return f"Unknown{user_id}"


async def normalize_discord_mentions_live(bot, content: str, bot_id: int) -> str:
    if not content:
        return content

    pattern = r"<@!?(\d+)>"

    async def repl(match):
        uid = int(match.group(1))
        if uid == bot_id:
            return "@you"
        return "@" + await fetch_username(bot, uid)

    matches = list(re.finditer(pattern, content))
    if not matches:
        return content

    replacements = await asyncio.gather(*(repl(m) for m in matches))
    for match, username in zip(matches, replacements):
        content = content.replace(match.group(0), username)
    return content


# ---------------------------
# Store & fetch messages
# ---------------------------
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
        return raw, trim_messages_to_token_budget(raw, max_tokens)
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


# ---------------------------
# Summarization
# ---------------------------
async def summarize_user_history(bot, user_key, evolve=False):
    raw_history, _ = fetch_history(user_key, limit_messages=config.MAX_HISTORY_MESSAGES)
    if not raw_history:
        return None

    old_summary = await memory_cache.get(user_key)
    first_msg = next(
        (m["content"] for m in raw_history if m.get("role") == "user"), None
    )

    if old_summary is None and first_msg:
        try:
            resp = await summary_client.chat.completions.acreate(
                model=config.MODEL,
                messages=[
                    {"role": "system", "content": PROMPTS["FIRST_CONTACT"]},
                    {"role": "user", "content": first_msg},
                ],
                max_tokens=100,
                temperature=0.8,
            )
            summary = resp.choices[0].message.content.strip()
            if summary:
                await memory_cache.set(user_key, summary)
                logger.info(f"First-contact profile created for {user_key}")
                return summary
        except Exception as e:
            logger.warning(f"First-contact failed for {user_key}: {e}")
            return None

    if not evolve or not old_summary:
        return old_summary

    recent_msgs = [m["content"] for m in raw_history if m.get("role") == "user"][-15:]
    if not recent_msgs:
        return old_summary

    evolution_prompt = PROMPTS["EVOLVE_PROFILE"].format(
        profile=old_summary, recent=recent_msgs
    )

    try:
        resp = await summary_client.chat.completions.acreate(
            model=config.MODEL,
            messages=[{"role": "system", "content": evolution_prompt}]
            + [{"role": "user", "content": m} for m in recent_msgs],
            max_tokens=100,
            temperature=0.9,
        )
        evolved = resp.choices[0].message.content.strip()
        if evolved:
            await memory_cache.set(user_key, evolved)
            logger.info(f"Profile evolved for {user_key}")
            return evolved
        return old_summary
    except Exception as e:
        logger.warning(f"Evolution failed for {user_key}: {e}")
        return old_summary


async def summarize_group_history(bot, group_name, raw_history):
    if not raw_history:
        return await group_memory_cache.get(group_name)

    if len(raw_history) < 6:
        summary = (
            f"New group '{group_name}' — Understand group dynamic and log observations."
        )
        await group_memory_cache.set(group_name, summary)
        return summary

    old_summary = await group_memory_cache.get(group_name) or ""

    recent = []
    for m in raw_history[-25:]:
        content = await normalize_discord_mentions_live(
            bot, m.get("content", ""), config.DISCORD_ID
        )
        sender = m.get("sender", "unknown")
        recent.append(f"{sender}: {content}")

    prompt = [{"role": "system", "content": PROMPTS["GROUP_SUMMARY"]}] + [
        {"role": "user", "content": t} for t in recent
    ]

    try:
        resp = await summary_client.chat.completions.acreate(
            model=config.MODEL,
            messages=prompt,
            max_tokens=250,
            temperature=1,
            timeout=6,
        )
        new_summary = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Group summarization failed for {group_name}: {e}")
        new_summary = old_summary

    if new_summary and new_summary != old_summary:
        await group_memory_cache.set(group_name, new_summary)

    return new_summary or old_summary


# ---------------------------
# Roast generation
# ---------------------------
async def get_roast_response(bot, user_message, group_name, sender_name):
    user_key = f"{group_name}:{sender_name}"
    is_private_env = group_name in ["DefaultGroup", "Discord_DM"]

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
        group_memory = await group_memory_cache.get(group_name)
        for entry in trimmed_group:
            entry["content"] = await normalize_discord_mentions_live(
                bot, entry.get("content", ""), config.DISCORD_ID
            )
    else:
        raw_group, trimmed_group, group_memory = [], [], ""

    if trimmed_user:
        trimmed_user = trimmed_user[:-1]
    if not is_private_env and trimmed_group:
        trimmed_group = trimmed_group[:-1]

    sys_parts = []
    user_memory = await memory_cache.get(user_key)
    if user_memory:
        sys_parts.append(f"User Profile: {user_memory}")
    if not is_private_env and group_memory:
        sys_parts.append(f"Current Collective Chatter Summary: {group_memory}")

    system_memory_text = "\n".join(sys_parts) if sys_parts else ""
    system_prompt = PROMPTS["GROUP_ROAST"] if not is_private_env else PROMPTS["ROAST"]
    messages = [{"role": "system", "content": system_prompt}]
    if system_memory_text:
        messages.append({"role": "system", "content": system_memory_text})

    if not is_private_env and trimmed_group:
        last_20 = trimmed_group[-20:] if len(trimmed_group) > 20 else trimmed_group
        for entry in last_20:
            s = entry.get("sender", "unknown")
            c = entry.get("content", "")
            messages.append({"role": "user", "content": f"{s}: {c}"})

    for m in trimmed_user:
        content = m.get("content", "")
        role = m.get("role", "user")
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    try:
        resp = await text_client.chat.completions.acreate(
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

    clean_reply = re.sub(r"^PSI-09\s*:\s*", "", base_reply, flags=re.IGNORECASE).strip()
    if not clean_reply:
        return ""

    # Store
    user_entry = {
        "role": "assistant",
        "content": clean_reply,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    group_entry = {
        "sender": "you",
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
# Async background summarizer
# ---------------------------
async def async_background_summarizer(bot):
    while True:
        try:
            async with _pending_lock:
                groups = list(_pending_group_summaries)
                _pending_group_summaries.clear()

            for group_name in groups:
                now = time.time()
                last = _last_group_summary_time.get(group_name, 0)
                if now - last < SUMMARY_COOLDOWN_SECONDS:
                    async with _pending_lock:
                        _pending_group_summaries.add(group_name)
                    continue
                raw_group, _ = fetch_group_history(
                    group_name,
                    limit_messages=config.GROUP_HISTORY_SLICE,
                    max_tokens=config.GROUP_HISTORY_TOKEN_LIMIT,
                )
                if raw_group:
                    for m in raw_group:
                        m["content"] = await normalize_discord_mentions_live(
                            bot, m.get("content", ""), config.DISCORD_ID
                        )
                    await summarize_group_history(bot, group_name, raw_group)
                    await group_memory_cache.reset_count(group_name)
                    _last_group_summary_time[group_name] = now

            async with _pending_lock:
                users = list(_pending_user_summaries)
                _pending_user_summaries.clear()

            for user_key in users:
                now = time.time()
                last = _last_user_summary_time.get(user_key, 0)
                if now - last < SUMMARY_COOLDOWN_SECONDS:
                    async with _pending_lock:
                        _pending_user_summaries.add(user_key)
                    continue
                await summarize_user_history(bot, user_key)
                await memory_cache.reset_count(user_key)
                _last_user_summary_time[user_key] = now

        except Exception as e:
            logger.warning(f"Async background summarizer error: {e}")

        await asyncio.sleep(12)


# ---------------------------
# Quart route
# ---------------------------
@app.route("/psi09", methods=["POST"])
async def psi09():
    try:
        data = await request.get_json(force=True)
        raw_message = data.get("message", "")
        sender_name = data.get("sender", "")
        group_name = data.get("group_name") or "DefaultGroup"
        if not raw_message or not sender_name:
            return jsonify({"reply": ""})

        user_message = await normalize_discord_mentions_live(
            bot, raw_message, config.DISCORD_ID
        )
        is_private = group_name in ["DefaultGroup", "Discord_DM"]

        if is_private:
            store_user_message(group_name, sender_name, user_message)
        else:
            store_group_message(group_name, sender_name, user_message)
            store_user_message(group_name, sender_name, user_message)

        user_key = f"{group_name}:{sender_name}"
        msg_count = await memory_cache.increment(user_key)

        if await memory_cache.get(user_key) is None:
            await summarize_user_history(bot, user_key)
            await memory_cache.reset_count(user_key)
            _last_user_summary_time[user_key] = time.time()
        elif msg_count >= config.EVOLVE_EVERY_N_MESSAGES:
            last = _last_user_summary_time.get(user_key, 0)
            if time.time() - last >= SUMMARY_COOLDOWN_SECONDS:
                await summarize_user_history(bot, user_key, evolve=True)
                _last_user_summary_time[user_key] = time.time()
                await memory_cache.reset_count(user_key)

        if not is_private:
            if (
                await group_memory_cache.increment(group_name)
                >= config.GROUP_SUMMARY_EVERY_N
            ):
                async with _pending_lock:
                    _pending_group_summaries.add(group_name)

        if is_private or re.search(
            r"<@!?" + str(config.DISCORD_ID) + r">", raw_message
        ):
            reply = await get_roast_response(
                bot, user_message.strip() or "[mention]", group_name, sender_name
            )
            return jsonify({"reply": reply})

        return jsonify({"reply": ""})

    except Exception as e:
        logger.exception(f"/psi09 failure: {e}")
        return jsonify({"reply": ""})


# ---------------------------
# Entrypoint
# ---------------------------
async def main():
    # Start background summarizer
    asyncio.create_task(async_background_summarizer(bot))
    # Run Quart with Hypercorn/Uvicorn
    import hypercorn.asyncio
    from hypercorn.config import Config as HypercornConfig

    cfg = HypercornConfig()
    cfg.bind = ["0.0.0.0:" + os.getenv("PORT", "5000")]
    await hypercorn.asyncio.serve(app, cfg)


if __name__ == "__main__":
    asyncio.run(main())
