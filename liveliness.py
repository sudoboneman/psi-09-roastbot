# liveliness.py (enhanced)
import random, threading, time
from collections import defaultdict

class LivelinessEngine:
    def __init__(self, db=None, logger=None):
        self.db = db
        self.logger = logger or print
        self.last_moods = defaultdict(lambda: "neutral")
        self.personality_state = "cold_cynical"
        self.memory = defaultdict(str)
        self.interaction_count = defaultdict(int)
        self.running = True
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()

    def _background_loop(self):
        while self.running:
            time.sleep(30)
            for key in list(self.last_moods.keys()):
                if random.random() < 0.4:
                    old_mood = self.last_moods[key]
                    self.last_moods[key] = self._random_mood_decay(old_mood)
                    self._maybe_evolve_personality()

    def _random_mood_decay(self, current):
        moods = ["neutral","irritated","playful","cold","sarcastic","nihilistic","venomous"]
        if current in moods: moods.remove(current)
        return random.choice(moods)

    def _maybe_evolve_personality(self):
        if random.random() < 0.05:
            old = self.personality_state
            options = ["cold_cynical","nihilistic_wit","darkly_playful","venomous_precision"]
            options.remove(old)
            self.personality_state = random.choice(options)

    def get_mood(self, user_key, message=""):
        keywords = ["fail","stupid","idiot","loser","hate","angry"]
        score = sum(word in message.lower() for word in keywords)
        weights = {
            "neutral":0.3,
            "irritated":0.3+0.1*score,
            "playful":0.1,
            "cold":0.1,
            "sarcastic":0.1+0.05*score,
            "nihilistic":0.05,
            "venomous":0.05+0.05*score
        }
        mood = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        self.last_moods[user_key] = mood
        self.interaction_count[user_key] += 1
        return mood

    def apply_mood(self, text, mood, user_key=None):
        if not text: return text
        modifiers = {
            "neutral": lambda t: t,
            "irritated": lambda t: f"{t} (you’re insufferable)",
            "playful": lambda t: f"{t} 😏",
            "cold": lambda t: f"{t} — detached observation",
            "sarcastic": lambda t: f"{t} (brilliant, isn’t it?)",
            "nihilistic": lambda t: f"{t} …all meaningless anyway",
            "venomous": lambda t: f"{t} 🔥"
        }
        result = modifiers.get(mood, lambda t: t)(text)

        # Subtle memory callback every 5 interactions
        if user_key and self.interaction_count[user_key]%5==0:
            remembered = self.memory.get(user_key,"")
            if remembered: result += f" (Remembering: '{remembered[-60:]}')"

        # Quirks every 7–10 interactions
        if user_key and self.interaction_count[user_key]%random.randint(7,10)==0:
            quirk = random.choice([
                "— Have you considered the futility of your choices?",
                "🤔 (this may haunt you later)",
                "Fragmented thought: existence… irony… entropy…",
                "PSI-09 observes silently, calculating disappointment.",
                "Did you think this would be different? Lol."
            ])
            result += " " + quirk

        # Mood-based stylistic deviation
        if mood in ["playful","nihilistic"] and random.random()<0.2:
            result = self._stylize_reply(result)

        return result

    def _stylize_reply(self, text):
        # Fragment sentences or add caps randomly
        if random.random()<0.5: text = " ".join(text.upper().split())
        else:
            parts = text.split()
            split_index = max(1,len(parts)//2)
            text = " ".join(parts[:split_index]) + " ... " + " ".join(parts[split_index:])
        return text

    def remember(self, user_key, text):
        if text:
            mem = self.memory.get(user_key,"")
            self.memory[user_key] = (mem + " " + text)[-500:]
            if self.db:
                try:
                    self.db["liveliness_memory"].update_one({"_id":user_key},{"$set":{"memory":self.memory[user_key],"updated_at":time.time()}},upsert=True)
                except Exception as e:
                    self.logger.warning(f"[Liveliness] Failed to persist memory: {e}")

    def shutdown(self):
        self.running = False
