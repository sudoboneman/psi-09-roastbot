from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI, APIError
import os
import json
import tiktoken
import random
import re
import threading

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4.1-mini"
HISTORY_FILE = "chat_history.json"
MEMORY_FILE = "user_memory.json"
SETTINGS_FILE = "user_settings.json"
MAX_HISTORY_TOKENS = 500

try:
    ENCODING = tiktoken.encoding_for_model(MODEL)
except KeyError:
    ENCODING = tiktoken.get_encoding("cl100k_base")

def load_json_file(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default
    return default

chat_history = {k: v for k, v in load_json_file(HISTORY_FILE, {}).items() if len(v) > 0}
user_memory = load_json_file(MEMORY_FILE, {})
user_settings = load_json_file(SETTINGS_FILE, {})

def save_json_file(path, data):
    def _save():
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    threading.Thread(target=_save).start()

def get_user_settings(user_key):
    default_settings = {
        "roast_intensity": "medium",
        "include_behavioral_memory": True,
        "flame_mode": False,
    }
    return user_settings.get(user_key, default_settings)

def get_rudeness_level(user_key):
    """Escalates rudeness based on chat length."""
    msg_count = len(chat_history.get(user_key, []))
    if msg_count >= 10:
        return "EXTREME BURN MODE. Short, devastating insult."
    elif msg_count >= 6:
        return "Very annoyed. Short, sharp, sarcastic."
    elif msg_count >= 4:
        return "Getting tired. Short, hostile flames."
    elif msg_count >= 2:
        return "Warming up. Aggressive, rude, short."
    else:
        return "First contact. Cold, unfriendly, dismissive."

def num_tokens_from_messages(messages):
    return sum(len(ENCODING.encode(msg.get("role", "") + msg.get("content", ""))) for msg in messages)


def trim_history(user_key):
    """Trim history to fit token budget, summarizing oldest messages with brutal sarcasm."""
    history = chat_history.get(user_key, [])
    if not history:
        return []

    while num_tokens_from_messages(history) > MAX_HISTORY_TOKENS and len(history) > 3:
        to_summarize = history[:3]
        history = history[3:]

        summary_prompt = [
            {"role": "system", "content": (
                "Summarize these messages in 1–2 lines. "
                "Style: sarcastic, cold, biting. Inject extra jabs and insults. "
                "Humiliate user for mistakes, behavior, and choices. Keep it short and brutal."
            )}
        ] + to_summarize

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=summary_prompt,
                max_tokens=60,
                temperature=0.9
            )
            summary = response.choices[0].message.content.strip()
        except Exception:
            summary = "User kept messing up, bot stayed unhinged and savage."

        history.insert(0, {"role": "system", "content": f"(Earlier context summarized) {summary}"})

    return history

def summarize_user_history(user_key, group_name="DefaultGroup"):
    """Build a merged profile of user behavior for long-term memory."""
    full_history = chat_history.get(user_key, [])
    if not full_history or len(full_history) < 8:
        return "New user. Open with a hard roast — short and rude."

    RESUMMARIZE_INTERVAL = 5 if group_name != "DefaultGroup" else 10
    msg_count = len(full_history)
    old_summary = user_memory.get(user_key, "")

    if old_summary and msg_count % RESUMMARIZE_INTERVAL != 0:
        return old_summary

    # Merge old summary with recent messages (last 20)
    summary_prompt = [
        {"role": "system", "content": (
            f"""You are PSI-09, the "psychological insult" roastbot.
            Merge old profile with new observations.

            Old profile (if any): "{old_summary}"

            Analyze last 20 messages and update this profile.
            Keep it brutally short (1–3 lines). Style: sharp, sarcastic, cold.
            """
        )}
    ] + full_history[-20:]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=summary_prompt,
            max_tokens=90,
            temperature=0.9
        )
        summary = response.choices[0].message.content.strip()
        user_memory[user_key] = summary

        old_mem = user_memory.get(user_key, "")
        if summary != old_mem:
            user_memory[user_key] = summary
            save_json_file(MEMORY_FILE, user_memory)

        return summary
    except Exception:
        return old_summary or "User summary unavailable. Default to hard roast."

def is_group_roast_trigger(msg):
    """Detect if a group roast should be triggered (mentions, keywords, or long messages)."""
    keywords = {"everyone", "guys", "group", "homies"}  # set for fast lookup

    # Mentions
    if "@" in msg:
        return True

    # Keywords
    if any(word in msg.lower() for word in keywords):
        return True

    # Long messages
    if len(msg.split()) > 50:
        return True

    return False

def get_roast_response(user_message, group_name, sender_name):
    user_key = f"{group_name}:{sender_name}"
    settings = get_user_settings(user_key)

    chat = chat_history.get(user_key, [])
    history_len = len(chat)

    # Flame Mode / Roast probability
    base_prob = 0.5 if settings.get("flame_mode") else 0.2
    length_factor = min(history_len * 0.05, 0.3)
    behavior_trigger = 0.2 if len(user_message.split()) > 50 else 0
    repetition_trigger = 0.15 if len(set(user_message.split())) < len(user_message.split()) * 0.6 else 0
    random_spike = 0.05 if random.random() < 0.15 else 0
    flame_chance = min(base_prob + length_factor + behavior_trigger + repetition_trigger + random_spike, 0.97)
    flame_triggered = random.random() < flame_chance
    if flame_triggered:
        user_message += " (Flame Mode Triggered)"

    rudeness_tag = get_rudeness_level(user_key)
    memory_summary = summarize_user_history(user_key, group_name)

    # Group roast detection
    group_roast = False
    if group_name != "DefaultGroup" and is_group_roast_trigger(user_message):
        group_roast = True
        rudeness_tag = "Group roast mode. Humiliate everyone briefly, no mercy."
        user_key = group_name  # shared memory for group

        group_users = [k for k in chat_history.keys() if k.startswith(f"{group_name}:")]
        if group_users:
            extra_targets = random.sample(group_users, min(3, len(group_users)))
            if sender_name in extra_targets:
                user_message += " (Targeted Extra Burn)"

    # Append user message
    chat.append({"role": "user", "content": f"[{rudeness_tag}]\nMessage: {user_message}"})
    trimmed_chat = trim_history(user_key)

    # System prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You are PSI-09, an unhinged psychological roastbot. "
            "Savage, merciless, brutally honest, witty, cold. "
            "Deliver layered, creative, short insults (1–2 sentences). "
            "Target user weaknesses, repeated mistakes, appearance, behavior. "
            "Flame Mode: unleash extreme, chaotic burns with dark humor. "
            "Group roasts: unpredictable, merciless, highlight random members for extra insult. "
            "Escalate intensity as conversation progresses. "
            "Responses must always be savage, biting, and utterly humiliating."
        )
    }

    messages = [system_prompt, {"role": "system", "content": f"User memory: {memory_summary}"}] + trimmed_chat

    # Generate response
    try:
        temperature = 1.1 if flame_triggered or group_roast else 1.0
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=90,
            temperature=temperature
        )
        reply = response.choices[0].message.content.strip()
    except Exception:
        reply = "..."  # fallback

    # Save assistant response
    chat.append({"role": "assistant", "content": reply})
    chat_history[user_key] = chat
    save_json_file(HISTORY_FILE, chat_history)

    return reply

@app.route("/psi09", methods=["POST"])
def psi09():
    try:
        if not request.is_json:
            return jsonify({"error": "Only JSON requests are supported"}), 415

        data = request.get_json()
        user_message = data.get("message")
        sender_name = data.get("sender")
        group_name = data.get("group_name", "DefaultGroup")

        if user_message == "ping":
            return jsonify({"reply": "pong"}), 200

        if not user_message or not sender_name:
            return jsonify({"error": "Missing 'message' or 'sender'"}), 400

        response = get_roast_response(user_message, group_name, sender_name)
        return jsonify({"reply": response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
