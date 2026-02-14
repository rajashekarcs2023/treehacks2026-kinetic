"""
AEGIS Memory System — Persistent, structured memory for coaching intelligence.

Three memory layers:
  1. Session Memory — current coaching session state (volatile)
  2. User Profile — persistent user data: injury history, preferences, progress
  3. Observation Memory — semantic observations with recency-weighted retrieval

No external vector DB dependency — uses TF-IDF similarity for retrieval
(lightweight, fast, zero setup). Can be upgraded to embeddings later.
"""

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Observation:
    """A single observation stored in memory."""
    content: str
    category: str  # coaching, posture, pattern, safety, user_preference
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    metadata: dict = field(default_factory=dict)
    relevance_boost: float = 1.0  # manual boost for important observations

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "category": self.category,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "relevance_boost": self.relevance_boost,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class UserProfile:
    """Persistent user profile for personalized coaching."""
    user_id: str = "default"
    name: str = ""

    # Physical attributes (affects coaching advice)
    injury_history: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    fitness_level: str = "beginner"  # beginner, intermediate, advanced
    dominant_side: str = "right"

    # Preferences
    coaching_style: str = "balanced"  # gentle, balanced, intense
    voice_feedback: bool = True
    preferred_skills: list[str] = field(default_factory=list)

    # Progress tracking
    total_sessions: int = 0
    total_reps: int = 0
    skills_practiced: dict[str, int] = field(default_factory=dict)  # skill → session count
    avg_scores: dict[str, float] = field(default_factory=dict)  # skill → avg score
    last_session: float = 0.0
    streak_days: int = 0

    def update_from_session(self, skill: str, reps: int, avg_score: float):
        """Update profile after a coaching session."""
        self.total_sessions += 1
        self.total_reps += reps
        self.skills_practiced[skill] = self.skills_practiced.get(skill, 0) + 1
        # Running average
        prev = self.avg_scores.get(skill, avg_score)
        count = self.skills_practiced[skill]
        self.avg_scores[skill] = prev + (avg_score - prev) / count
        self.last_session = time.time()

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "injury_history": self.injury_history,
            "limitations": self.limitations,
            "fitness_level": self.fitness_level,
            "dominant_side": self.dominant_side,
            "coaching_style": self.coaching_style,
            "voice_feedback": self.voice_feedback,
            "preferred_skills": self.preferred_skills,
            "total_sessions": self.total_sessions,
            "total_reps": self.total_reps,
            "skills_practiced": self.skills_practiced,
            "avg_scores": {k: round(v, 1) for k, v in self.avg_scores.items()},
            "last_session": self.last_session,
            "streak_days": self.streak_days,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def get_context_for_agent(self) -> str:
        """Generate a context string for the agent's system prompt."""
        lines = []
        if self.name:
            lines.append(f"User: {self.name}")
        lines.append(f"Fitness level: {self.fitness_level}")
        lines.append(f"Coaching style preference: {self.coaching_style}")
        lines.append(f"Dominant side: {self.dominant_side}")
        if self.injury_history:
            lines.append(f"INJURIES/CAUTION: {', '.join(self.injury_history)}")
        if self.limitations:
            lines.append(f"Limitations: {', '.join(self.limitations)}")
        lines.append(f"Sessions completed: {self.total_sessions}, Total reps: {self.total_reps}")
        if self.avg_scores:
            top = sorted(self.avg_scores.items(), key=lambda x: -x[1])[:5]
            lines.append(f"Best skills: {', '.join(f'{s} ({v:.0f}%)' for s, v in top)}")
        return "\n".join(lines)


class MemoryStore:
    """Persistent memory store with TF-IDF retrieval."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "memory"
            )
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.observations: list[Observation] = []
        self.user_profile = UserProfile()
        self.session_context: dict = {}  # volatile per-session state

        self._load()

    # ── Observation Memory ───────────────────────────────────────────

    def add_observation(self, content: str, category: str = "general",
                        session_id: str = "", metadata: dict = None,
                        relevance_boost: float = 1.0):
        """Store an observation."""
        obs = Observation(
            content=content,
            category=category,
            session_id=session_id,
            metadata=metadata or {},
            relevance_boost=relevance_boost,
        )
        self.observations.append(obs)
        self._save_observations()
        return obs

    def recall(self, query: str, top_k: int = 5,
               category: str = None, max_age_hours: float = None) -> list[Observation]:
        """Retrieve relevant observations using TF-IDF similarity + recency weighting.

        Args:
            query: Natural language query
            top_k: Max results to return
            category: Filter by category (optional)
            max_age_hours: Only return observations within this time window
        """
        candidates = self.observations

        # Filter by category
        if category:
            candidates = [o for o in candidates if o.category == category]

        # Filter by age
        if max_age_hours:
            cutoff = time.time() - max_age_hours * 3600
            candidates = [o for o in candidates if o.timestamp >= cutoff]

        if not candidates:
            return []

        # Score each observation
        query_tokens = _tokenize(query)
        scored = []
        for obs in candidates:
            obs_tokens = _tokenize(obs.content)

            # TF-IDF-like similarity
            sim = _token_similarity(query_tokens, obs_tokens)

            # Recency decay (half-life = 24 hours)
            age_hours = (time.time() - obs.timestamp) / 3600
            recency = math.exp(-0.03 * age_hours)  # ~50% at 24h

            # Final score
            score = sim * 0.6 + recency * 0.3 + obs.relevance_boost * 0.1
            scored.append((obs, score))

        scored.sort(key=lambda x: -x[1])
        return [obs for obs, _ in scored[:top_k]]

    def get_recent(self, n: int = 10, category: str = None) -> list[Observation]:
        """Get most recent observations."""
        candidates = self.observations
        if category:
            candidates = [o for o in candidates if o.category == category]
        return candidates[-n:]

    def get_session_observations(self, session_id: str) -> list[Observation]:
        """Get all observations from a specific session."""
        return [o for o in self.observations if o.session_id == session_id]

    # ── User Profile ─────────────────────────────────────────────────

    def update_profile(self, **kwargs):
        """Update user profile fields."""
        for key, value in kwargs.items():
            if hasattr(self.user_profile, key):
                setattr(self.user_profile, key, value)
        self._save_profile()

    def record_session_complete(self, skill: str, reps: int, avg_score: float):
        """Record a completed coaching session in user profile."""
        self.user_profile.update_from_session(skill, reps, avg_score)
        self._save_profile()

    # ── Session Context (volatile) ───────────────────────────────────

    def set_session(self, key: str, value):
        """Set a volatile session value."""
        self.session_context[key] = value

    def get_session(self, key: str, default=None):
        """Get a volatile session value."""
        return self.session_context.get(key, default)

    def clear_session(self):
        """Clear volatile session state."""
        self.session_context = {}

    # ── Context for Agent ────────────────────────────────────────────

    def get_context_for_agent(self, query: str = "") -> str:
        """Build a memory context block for the agent's system prompt.

        Includes: user profile + relevant observations.
        """
        parts = []

        # User profile
        profile_ctx = self.user_profile.get_context_for_agent()
        if profile_ctx:
            parts.append(f"## User Profile\n{profile_ctx}")

        # Relevant memories
        if query:
            relevant = self.recall(query, top_k=3)
            if relevant:
                mem_lines = []
                for obs in relevant:
                    age = (time.time() - obs.timestamp) / 3600
                    age_str = f"{age:.0f}h ago" if age < 48 else f"{age/24:.0f}d ago"
                    mem_lines.append(f"- [{obs.category}] ({age_str}) {obs.content}")
                parts.append(f"## Relevant Memories\n" + "\n".join(mem_lines))

        # Recent observations (last 3)
        recent = self.get_recent(3)
        if recent:
            recent_lines = [f"- {o.content}" for o in recent]
            parts.append(f"## Recent Observations\n" + "\n".join(recent_lines))

        return "\n\n".join(parts) if parts else ""

    # ── Persistence ──────────────────────────────────────────────────

    def _save_observations(self):
        filepath = os.path.join(self.data_dir, "observations.jsonl")
        with open(filepath, "w") as f:
            for obs in self.observations[-1000:]:  # Keep last 1000
                f.write(json.dumps(obs.to_dict()) + "\n")

    def _save_profile(self):
        filepath = os.path.join(self.data_dir, "user_profile.json")
        with open(filepath, "w") as f:
            json.dump(self.user_profile.to_dict(), f, indent=2)

    def _load(self):
        # Load observations
        obs_path = os.path.join(self.data_dir, "observations.jsonl")
        if os.path.exists(obs_path):
            self.observations = []
            with open(obs_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.observations.append(Observation.from_dict(json.loads(line)))
                        except Exception:
                            continue

        # Load profile
        prof_path = os.path.join(self.data_dir, "user_profile.json")
        if os.path.exists(prof_path):
            try:
                with open(prof_path) as f:
                    self.user_profile = UserProfile.from_dict(json.load(f))
            except Exception:
                pass

    def get_stats(self) -> dict:
        """Memory system stats."""
        cats = Counter(o.category for o in self.observations)
        return {
            "total_observations": len(self.observations),
            "categories": dict(cats),
            "user_profile": self.user_profile.to_dict(),
            "session_keys": list(self.session_context.keys()),
        }


# ═══════════════════════════════════════════════════════════════════════
# TF-IDF-like text similarity (no external dependencies)
# ═══════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return re.findall(r'[a-z0-9]+', text.lower())


def _token_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Jaccard-like similarity with term frequency weighting."""
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b
    union = set_a | set_b
    if not union:
        return 0.0
    return len(intersection) / len(union)
