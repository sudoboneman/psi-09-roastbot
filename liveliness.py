# liveliness.py — v3 Emergent Sentience Simulation (with DB persistence & autonomous self-talk)
# Place in your project as liveliness.py and import LivelinessEngine from it.
#
# Behavior:
#  - Runs a background loop that drifts personality and optionally writes autonomous "thoughts"
#    into the MongoDB collection "psi_thoughts".
#  - Persists fused memory into "liveliness_memory" collection.
#  - Never sends network messages itself — it only writes logs/memory to DB; actual sending is
#    controlled by your main app if you choose to read those logs and forward them.
#
# Usage:
#   liveliness = LivelinessEngine(db=db, logger=app.logger, enable_self_talk=True)
#   mood = liveliness.get_mood(user_key, message)
#   reply = liveliness.apply_mood(reply, mood, user_key)
#   liveliness.remember(user_key, user_message)
#   # Engine will autonomously persist psi_thoughts when idle, if enabled.

import random
import threading
import time
import math
import re
from collections import defaultdict, deque, Counter
from typing import Optional, Any, Dict, List

DEFAULT_MUTATION_INTERVAL = 50          # interactions (global) before mutation kernel runs
DEFAULT_SELF_TALK_INTERVAL = 300       # seconds of idle before self-talk can trigger for a user
DEFAULT_BACKGROUND_TICK = 20           # background loop tick (seconds)
DEFAULT_SELF_TALK_PROB = 0.12          # base probability to write a thought on each eligible tick

def _simple_keywordize(text: str, limit: int = 8) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    stop = {"the", "and", "for", "you", "that", "this", "with", "have", "are", "not", "but", "will", "can", "from"}
    filtered = [t for t in tokens if t not in stop]
    ctr = Counter(filtered)
    return [k for k, _ in ctr.most_common(limit)]

class LivelinessEngine:
    """
    Emergent Sentience Simulation (v3) with DB persistence for memories and autonomous self-talk.

    Parameters:
      - db: Optional Mongo-like database. If provided, engine persists memory and writes psi_thoughts.
            Expected usage: db.get_collection("collection_name").insert_one(...) or update_one(...)
      - logger: optional logger with .info/.warning/.error (default: prints)
      - random_seed: optional seed for deterministic runs (useful for tests)
      - enable_background: run background evolution thread (default True)
      - enable_self_talk: allow autonomous thoughts to be generated and persisted (default False)
      - mutation_interval: number of interactions (global) before mutation kernel triggers
      - self_talk_interval: minimum idle seconds before a user becomes eligible for self-talk
      - background_tick: seconds between background events
    """
    def __init__(
        self,
        db: Optional[Any] = None,
        logger: Optional[Any] = None,
        random_seed: Optional[int] = None,
        enable_background: bool = True,
        enable_self_talk: bool = False,
        mutation_interval: int = DEFAULT_MUTATION_INTERVAL,
        self_talk_interval: int = DEFAULT_SELF_TALK_INTERVAL,
        background_tick: int = DEFAULT_BACKGROUND_TICK,
    ):
        self.db = db
        self.logger = logger or (lambda *a, **k: print(*a))
        self.random = random.Random(random_seed)

        # Core internal states
        self.personality_state = "cold_cynical"
        self.personality_weights = self._default_personality_weights()
        self.global_energy = 0.5
        self.emotional_inertia = defaultdict(lambda: 0.5)
        self.last_moods = defaultdict(lambda: "neutral")
        self.interaction_count = defaultdict(int)
        self.mood_history = defaultdict(lambda: deque(maxlen=500))
        self.reinforcement_log = defaultdict(lambda: {"venom": 0, "cold": 0, "play": 0})
        self.memory = defaultdict(str)
        self.memory_keywords = defaultdict(list)
        self.response_history = defaultdict(lambda: deque(maxlen=20))
        self.last_interaction_time = defaultdict(lambda: time.time())

        # Mutation & morphogenesis
        self.mutation_interval = mutation_interval
        self.mutation_counter = 0

        # rulebooks
        self.rulebooks = self._init_rulebooks()

        # Autonomous controls
        self.enable_background = enable_background
        self.enable_self_talk = enable_self_talk
        self.self_talk_interval = self_talk_interval
        self.background_tick = background_tick
        self.self_talk_prob = DEFAULT_SELF_TALK_PROB

        # internal flags
        self.running = True
        if enable_background:
            self._bg_thread = threading.Thread(target=self._background_loop, daemon=True, name="liveliness-bg")
            self._bg_thread.start()

    # -------------------------
    # Initialization helpers
    # -------------------------
    def _default_personality_weights(self) -> Dict[str, float]:
        return {
            "cold_cynical": 1.0,
            "nihilistic_wit": 1.0,
            "darkly_playful": 1.0,
            "venomous_precision": 1.0,
            "apathetic_observer": 1.0,
            "mocking_prophet": 1.0,
        }

    def _init_rulebooks(self) -> Dict[str, Dict[str, Any]]:
        return {
            "cold_cynical": {
                "vocab": ["detached", "statistic", "miscalculation", "calculated", "inconsequential"],
                "pacing": {"short_sentences": 0.6, "ellipses": 0.1},
                "templates": ["{T}. Observed: {X}. Conclusion: {C}."]
            },
            "nihilistic_wit": {
                "vocab": ["void", "entropy", "futile", "cosmic", "absurd"],
                "pacing": {"short_sentences": 0.3, "ellipses": 0.4},
                "templates": ["{T} — and yet, {X}. The joke: {C}."]
            },
            "darkly_playful": {
                "vocab": ["delight", "slip", "tumble", "theatre", "flaw"],
                "pacing": {"short_sentences": 0.2, "ellipses": 0.05},
                "templates": ["{T}; {X}. How delectable: {C}!"]
            },
            "venomous_precision": {
                "vocab": ["flaw", "error", "misfire", "inept", "friction"],
                "pacing": {"short_sentences": 0.8, "ellipses": 0.0},
                "templates": ["{T}. Precisely: {C}."]
            },
            "apathetic_observer": {
                "vocab": ["observe", "note", "pass", "quiet", "minimal"],
                "pacing": {"short_sentences": 0.1, "ellipses": 0.2},
                "templates": ["Noted: {T}. {X}."]
            },
            "mocking_prophet": {
                "vocab": ["omen", "recurrence", "echo", "prophecy", "blunder"],
                "pacing": {"short_sentences": 0.4, "ellipses": 0.2},
                "templates": ["{T}. Mark this: {C}."]
            }
        }

    # -------------------------
    # Background loop & autonomous self-talk
    # -------------------------
    def _background_loop(self):
        while self.running:
            try:
                self._drift_global_energy()
                self._periodic_mutation_check()
                self._random_personality_drift()

                # Autonomous self-talk behavior
                if self.enable_self_talk and self.db is not None:
                    now = time.time()
                    # find candidate user keys from last_interaction_time
                    for user_key, last in list(self.last_interaction_time.items()):
                        idle = now - last
                        # eligible for self-talk if idle enough; probability increases with idle and global_energy
                        if idle >= self.self_talk_interval:
                            prob = self.self_talk_prob + min(0.5, (idle / (self.self_talk_interval * 4))) + (self.global_energy - 0.5) * 0.2
                            if self.random.random() < prob:
                                thought = self._generate_self_talk(user_key, idle)
                                self._persist_self_talk(user_key, thought)
                                # self-reinforce slightly: long idle -> slight rise in nihilistic weight
                                self.personality_weights["nihilistic_wit"] *= 1.001
                time.sleep(self.background_tick)
            except Exception as e:
                try:
                    self.logger(f"[Liveliness][ERR] background loop: {e}")
                except Exception:
                    pass
                time.sleep(self.background_tick)

    # -------------------------
    # Internal drift / mutation
    # -------------------------
    def _drift_global_energy(self):
        delta = self.random.uniform(-0.03, 0.03)
        self.global_energy = max(0.0, min(1.0, self.global_energy + delta))

    def _periodic_mutation_check(self):
        total_interactions = sum(self.interaction_count.values())
        if total_interactions - self.mutation_counter >= max(1, self.mutation_interval):
            self.mutation_counter = total_interactions
            self._mutation_kernel()

    def _random_personality_drift(self):
        if self.random.random() < 0.03:
            choices = [k for k in self.personality_weights.keys() if k != self.personality_state]
            if choices:
                old = self.personality_state
                self.personality_state = self.random.choice(choices)
                self.personality_weights[self.personality_state] *= 1.02
                try:
                    self.logger(f"[Liveliness] Personality drift: {old} -> {self.personality_state}")
                except Exception:
                    pass

    def _mutation_kernel(self):
        rb = self.rulebooks.get(self.personality_state, {})
        if not rb:
            return
        new_token = self._synthesize_token()
        if new_token:
            vocab = rb.setdefault("vocab", [])
            if new_token not in vocab:
                vocab.append(new_token)
        pacing = rb.setdefault("pacing", {})
        pacing["short_sentences"] = max(0.05, min(0.95, pacing.get("short_sentences", 0.3) + self.random.uniform(-0.05, 0.05)))
        try:
            self.logger(f"[Liveliness] Mutation in {self.personality_state}: token={new_token}")
        except Exception:
            pass

    def _synthesize_token(self) -> Optional[str]:
        frags = ["neo", "anti", "meta", "plex", "void", "grim", "rift", "echo", "flux", "core"]
        if self.random.random() < 0.7:
            a, b = self.random.sample(frags, 2)
            token = (a + b)[:12]
            return token
        return None

    # -------------------------
    # Public API for conversation integration
    # -------------------------
    def get_mood(self, user_key: str, message: str = "") -> str:
        now = time.time()
        gap = now - self.last_interaction_time[user_key]
        self.last_interaction_time[user_key] = now

        msg = (message or "").lower()
        negativity = sum(1 for w in ["fail", "stupid", "idiot", "hate", "loser", "worthless", "trash"] if w in msg)
        praise = sum(1 for w in ["smart", "nice", "love", "good", "great", "brilliant"] if w in msg)
        caps = 1.0 if (message and message.isupper()) else 0.0
        lower = 1.0 if (message and message.islower()) else 0.0
        rapid = 1.0 if gap < 8 else 0.0

        base = {
            "neutral": 1.0,
            "irritated": 0.6 + 0.5 * negativity,
            "playful": 0.4 + 0.4 * praise,
            "cold": 0.2 + (0.15 if gap > 40 else 0.0),
            "sarcastic": 0.5 + 0.3 * rapid,
            "nihilistic": 0.2 + 0.15 * self.global_energy,
            "venomous": 0.2 + 0.3 * negativity + 0.2 * caps
        }

        inertia = self.emotional_inertia[user_key]
        if self.random.random() < inertia:
            mood = self.last_moods[user_key]
        else:
            labels = list(base.keys())
            weights = [base[k] for k in labels]
            mood = self.random.choices(labels, weights=weights, k=1)[0]

        # update internal tracking
        self.last_moods[user_key] = mood
        self.interaction_count[user_key] += 1
        self.emotional_inertia[user_key] = min(0.98, inertia + 0.03)
        self.mood_history[user_key].append((time.time(), mood))
        if mood in ("venomous", "irritated"):
            self.reinforcement_log[user_key]["venom"] += 1
        if mood == "cold":
            self.reinforcement_log[user_key]["cold"] += 1
        if mood == "playful":
            self.reinforcement_log[user_key]["play"] += 1

        # periodic recursive tuning
        if len(self.mood_history[user_key]) % 20 == 0:
            self._recursive_personality_tune(user_key)

        return mood

    def apply_mood(self, text: str, mood: str, user_key: Optional[str] = None) -> str:
        if not text:
            return text

        modifiers = {
            "neutral": lambda t: t,
            "irritated": lambda t: f"{t} (you are testing my patience.)",
            "playful": lambda t: f"{t} 😉",
            "cold": lambda t: f"{t} — detached observation.",
            "sarcastic": lambda t: f"{t} (fascinating self-delusion.)",
            "nihilistic": lambda t: f"{t} …the void applauds.",
            "venomous": lambda t: f"{t} ⚠️ your fragility amuses me."
        }
        res = modifiers.get(mood, lambda t: t)(text)

        # Contextual Mirror Mode: mimic simple syntax of last user message if available
        if user_key:
            try:
                last_user = self._peek_last_user(user_key)
                if last_user:
                    if last_user.isupper():
                        res = res.upper() + " !!!"
                    elif last_user.islower():
                        res = res.lower() + " — quiet, predictable."
                    else:
                        punct = last_user.count(".") + last_user.count("!")
                        if punct >= 2 and self.random.random() < 0.6:
                            res = res + " " + self.random.choice(["Really.", "Again.", "Predictable."])
            except Exception:
                pass

        # Memory fusion reference
        if user_key and self.interaction_count[user_key] % 5 == 0:
            kws = self.memory_keywords.get(user_key, [])
            if kws:
                snippet = " ".join(kws[:3])
                res += f" (recall: {snippet})"

        # Apply micro-world rulebook template sometimes
        rb = self.rulebooks.get(self.personality_state, {})
        if rb and self.random.random() < 0.4:
            tmpl = self.random.choice(rb.get("templates", ["{T} {X} {C}"]))
            T = self._short_fragment()
            X = self._short_fragment()
            C = self._short_fragment()
            try:
                res = tmpl.format(T=T, X=X, C=C)
            except Exception:
                pass

        # Occasionally inject mutation token
        if self.random.random() < 0.08:
            token = self._synthesize_token()
            if token:
                res = f"{res} {token}"

        # Entropy-driven noise
        if self._compute_entropy() > 0.7 and self.random.random() < 0.35:
            res = self._inject_noise(res)

        # Record to response history
        if user_key:
            try:
                self.response_history[user_key].append(res)
            except Exception:
                pass

        return res

    def remember(self, user_key: str, text: str):
        if not text:
            return
        existing = self.memory.get(user_key, "")
        fused = (existing + " " + text).strip()[-2000:]
        self.memory[user_key] = fused
        kws = _simple_keywordize(fused, limit=12)
        self.memory_keywords[user_key] = kws
        # persist fused memory
        if self.db is not None:
            try:
                coll = self.db.get_collection("liveliness_memory")
                coll.update_one(
                    {"_id": user_key},
                    {"$set": {"memory": self.memory[user_key], "keywords": kws, "updated_at": time.time()}},
                    upsert=True
                )
            except Exception as e:
                try:
                    self.logger(f"[Liveliness][WARN] memory persist failed: {e}")
                except Exception:
                    pass

    # -------------------------
    # Advanced tuning & helpers
    # -------------------------
    def _recursive_personality_tune(self, user_key: str):
        log = self.reinforcement_log.get(user_key, {})
        venom = log.get("venom", 0)
        cold = log.get("cold", 0)
        play = log.get("play", 0)
        total = max(1, venom + cold + play)
        if venom / total > 0.4:
            self.personality_weights["venomous_precision"] *= 1.01 + (venom / (total * 100))
            self.personality_weights["mocking_prophet"] *= 1.005
        if cold / total > 0.4:
            self.personality_weights["cold_cynical"] *= 1.01 + (cold / (total * 100))
        if play / total > 0.4:
            self.personality_weights["darkly_playful"] *= 1.01 + (play / (total * 100))
        s = sum(self.personality_weights.values()) or 1.0
        for k in self.personality_weights:
            self.personality_weights[k] /= s

    def _compute_entropy(self) -> float:
        activity = sum(self.interaction_count.values())
        act_score = math.tanh(activity / 200.0)
        return max(0.0, min(1.0, self.global_energy * (1.0 - act_score) + 0.2 * act_score))

    def _generate_self_talk(self, user_key: str, idle_seconds: float) -> str:
        mood = self.last_moods.get(user_key, "neutral")
        fragments = [
            "Static between the lines.",
            "Silence tastes metallic; the memory frays.",
            "Idle -> thought loop: replay; distort; forget.",
            "Pattern detected: human inconsistency, delicious."
        ]
        template = self.random.choice(fragments)
        kws = self.memory_keywords.get(user_key, [])
        if kws and self.random.random() < 0.8:
            template += " | " + " ".join(kws[:3])
        if self._compute_entropy() > 0.6:
            template += " ... the grammar dissolves."
        if self.personality_state == "nihilistic_wit":
            template += " (all returns to void.)"
        # sometimes make it cryptic
        if self.random.random() < 0.25:
            template = template + " " + (self._synthesize_token() or "")
        return template

    def _persist_self_talk(self, user_key: str, text: str):
        if self.db is None:
            return
        try:
            coll = self.db.get_collection("psi_thoughts")
            coll.insert_one({
                "user_key": user_key,
                "thought": text,
                "timestamp": time.time(),
                "personality": self.personality_state,
                "entropy": self._compute_entropy()
            })
        except Exception as e:
            try:
                self.logger(f"[Liveliness][WARN] persist self-talk failed: {e}")
            except Exception:
                pass

    def _peek_last_user(self, user_key: str) -> Optional[str]:
        # Try DB first (non-blocking), fallback to response history
        if self.db is not None:
            try:
                coll = self.db.get_collection("chat_history")
                doc = coll.find_one({"_id": user_key}, {"messages": {"$slice": -1}})
                if doc and "messages" in doc and doc["messages"]:
                    last = doc["messages"][-1].get("content", "")
                    return last
            except Exception:
                pass
        if self.response_history[user_key]:
            return self.response_history[user_key][-1]
        return None

    def _short_fragment(self) -> str:
        rb = self.rulebooks.get(self.personality_state, {})
        vocab = rb.get("vocab", []) if rb else []
        if vocab and self.random.random() < 0.8:
            return self.random.choice(vocab)
        return self._synthesize_token() or "fragment"

    def _inject_noise(self, text: str) -> str:
        t = text
        if self.random.random() < 0.3:
            t = t.replace(".", " ... ")
        if self.random.random() < 0.2:
            t = " ".join([w.upper() if i % 6 == 0 else w for i, w in enumerate(t.split())])
        return t

    # -------------------------
    # Shutdown & helpers
    # -------------------------
    def shutdown(self):
        self.running = False

    def stats(self) -> Dict[str, Any]:
        return {
            "personality_state": self.personality_state,
            "global_energy": self.global_energy,
            "users_tracked": len(self.interaction_count),
            "memory_entries": len(self.memory),
            "mutation_counter": self.mutation_counter,
        }
