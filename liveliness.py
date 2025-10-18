import random
import math
import time
import threading
from datetime import datetime, timezone


class LivelinessEngine:
    def __init__(self, db=None, logger=None):
        self.db = db
        self.logger = logger
        self.last_moods = {}
        self.memory = {}
        self.personality_state = "neutral"
        self.last_shift = time.time()

        self.mood_profiles = {
            "apathy": [
                "I'm bored already.",
                "Your words feel like static in the void.",
                "Existence yawns in your direction."
            ],
            "rage": [
                "You really think you can offend me?",
                "I could roast your entire genetic line and still be underwhelmed.",
                "You're the reason evolution needed debugging."
            ],
            "amusement": [
                "You're like a cosmic joke that writes itself.",
                "This is adorable — in a tragic, doomed sort of way.",
                "You keep talking like irony’s a vitamin deficiency."
            ],
            "chaos": [
                "Language collapsing, logic leaking — perfect, stay right there.",
                "Reality's folding in on your syntax.",
                "You’re a metaphor inside a recursion of regret."
            ],
            "precision": [
                "Observation logged: your thought process resembles damp circuitry.",
                "Statistical note: every word you speak lowers entropy… for all the wrong reasons.",
                "Diagnostic: human ego detected — unstable."
            ],
            "melancholy": [
                "Do you ever realize how empty defiance feels?",
                "Silence would’ve been more profound.",
                "Tragedy wears your face convincingly."
            ]
        }

        # Try to load persisted state
        if self.db:
            try:
                state = self.db.liveliness.find_one({"_id": "psi09_state"})
                if state:
                    self.last_moods = state.get("last_moods", {})
                    self.memory = state.get("memory", {})
                    self.personality_state = state.get("personality_state", "neutral")
                    self.last_shift = state.get("last_shift", time.time())
                    if self.logger:
                        self.logger.info(f"[Liveliness] Restored state from DB (moods={len(self.last_moods)}, memory={len(self.memory)})")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[Liveliness] Failed to restore state: {e}")

        # Start sync thread
        threading.Thread(target=self._sync_loop, daemon=True).start()

    # --------------------------------------------------
    def _sync_loop(self):
        while True:
            try:
                if self.db:
                    self.db.liveliness.update_one(
                        {"_id": "psi09_state"},
                        {"$set": {
                            "last_moods": self.last_moods,
                            "memory": self.memory,
                            "personality_state": self.personality_state,
                            "last_shift": self.last_shift,
                            "updated_at": datetime.now(timezone.utc)
                        }},
                        upsert=True
                    )
                    if self.logger:
                        self.logger.debug("[Liveliness] Synced state to DB.")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[Liveliness] DB sync failed: {e}")
            time.sleep(300)  # every 5 min

    # --------------------------------------------------
    def _text_entropy(self, text: str) -> float:
        freq = {ch: text.count(ch) for ch in set(text)}
        total = len(text)
        if total == 0:
            return 0
        return -sum((f / total) * math.log2(f / total) for f in freq.values())

    # --------------------------------------------------
    def _shift_personality(self):
        now = time.time()
        if now - self.last_shift > 600:
            self.personality_state = random.choice(
                ["neutral", "detached", "sarcastic", "philosophical", "volatile"]
            )
            self.last_shift = now
            if self.logger:
                self.logger.info(f"[Liveliness] Personality drift → {self.personality_state}")

    # --------------------------------------------------
    def get_mood(self, user_key: str, message: str) -> str:
        self._shift_personality()
        now = time.time()
        last_mood, last_time = self.last_moods.get(user_key, (None, 0))
        entropy = self._text_entropy(message)
        elapsed = now - last_time

        if elapsed < 60 and random.random() < 0.8:
            return last_mood or "precision"

        if entropy < 3.0:
            mood = "apathy"
        elif entropy > 4.5:
            mood = random.choice(["chaos", "amusement"])
        else:
            mood = random.choice(["precision", "rage", "amusement", "melancholy"])

        if random.random() < 0.1:
            mood = "chaos"

        self.last_moods[user_key] = (mood, now)
        return mood

    # --------------------------------------------------
    def remember(self, user_key: str, roast: str):
        self.memory.setdefault(user_key, []).append(roast)
        if len(self.memory[user_key]) > 5:
            self.memory[user_key].pop(0)

    # --------------------------------------------------
    def apply_mood(self, text: str, mood: str) -> str:
        try:
            prefix = suffix = ""
            profile = self.mood_profiles.get(mood, [])
            if not profile:
                return text

            if mood == "apathy":
                suffix = random.choice(profile)
            elif mood == "rage":
                prefix = random.choice(profile) + " "
            elif mood == "amusement":
                suffix = " " + random.choice(profile)
            elif mood == "chaos":
                glitch = "".join(random.choice(["¤", "∆", "⛓", "∅"]) for _ in range(random.randint(2, 5)))
                prefix = f"{glitch} "
                suffix = " " + random.choice(profile)
            elif mood == "precision":
                prefix = random.choice(profile) + " "
            elif mood == "melancholy":
                suffix = " " + random.choice(profile)

            if random.random() < 0.25:
                text = text.replace(".", "...").replace("!", "!!").replace("?", "?!")

            time.sleep(random.uniform(0.05, 0.3))
            return f"{prefix}{text} {suffix}".strip()
        except Exception:
            return text

