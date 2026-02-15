"""
Room-based Multiplayer Coaching — AEGIS

Manages coaching rooms where multiple users can practice together.
Each user gets their own coaching session + agent intelligence.
Claude orchestrator compares all users at session end.

Architecture:
  Room (code: str)
  ├── participants: {user_id: ParticipantState}
  │   ├── coaching_session: CoachingSession
  │   ├── display_name: str
  │   └── joined_at: float
  └── created_at: float
"""

import asyncio
import json
import random
import string
import time
from dataclasses import dataclass, field
from typing import Optional

from fastapi import WebSocket

from aegis.pose_comparison import CoachingSession


def _generate_room_code() -> str:
    """Generate a 6-character room code (e.g., 'JAB42X')."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


@dataclass
class ParticipantState:
    """State for a single participant in a room."""
    user_id: str
    display_name: str
    coaching_session: Optional[CoachingSession] = None
    joined_at: float = field(default_factory=time.time)
    ws_clients: list = field(default_factory=list)  # coaching WS clients for this user
    video_ws: Optional[WebSocket] = None
    audio_ws: Optional[WebSocket] = None
    is_active: bool = True

    def get_summary(self) -> dict:
        """Get participant's coaching summary."""
        if self.coaching_session:
            progress = self.coaching_session.get_progress()
            return {
                "user_id": self.user_id,
                "display_name": self.display_name,
                "is_active": self.is_active,
                **progress,
            }
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "is_active": self.is_active,
            "reps_completed": 0,
            "avg_score": 0,
        }


@dataclass
class Room:
    """A coaching room with multiple participants."""
    code: str
    skill_name: str
    created_at: float = field(default_factory=time.time)
    participants: dict[str, ParticipantState] = field(default_factory=dict)
    is_active: bool = True

    @property
    def participant_count(self) -> int:
        return len(self.participants)

    def add_participant(self, user_id: str, display_name: str) -> ParticipantState:
        """Add a participant to the room."""
        if user_id in self.participants:
            return self.participants[user_id]
        participant = ParticipantState(user_id=user_id, display_name=display_name)
        participant.coaching_session = CoachingSession(skill_name=self.skill_name)
        self.participants[user_id] = participant
        return participant

    def remove_participant(self, user_id: str) -> Optional[ParticipantState]:
        """Remove a participant from the room."""
        return self.participants.pop(user_id, None)

    def get_leaderboard(self) -> list[dict]:
        """Get all participants ranked by avg score."""
        summaries = [p.get_summary() for p in self.participants.values()]
        return sorted(summaries, key=lambda x: x.get("avg_score", 0), reverse=True)

    def get_comparison_data(self) -> dict:
        """Get structured comparison data for all participants."""
        leaderboard = self.get_leaderboard()
        return {
            "room_code": self.code,
            "skill": self.skill_name,
            "participant_count": len(leaderboard),
            "leaderboard": leaderboard,
            "duration": round(time.time() - self.created_at, 1),
        }


class RoomManager:
    """Manages all active coaching rooms."""

    def __init__(self):
        self._rooms: dict[str, Room] = {}

    def create_room(self, skill_name: str) -> Room:
        """Create a new room with a unique code."""
        code = _generate_room_code()
        while code in self._rooms:
            code = _generate_room_code()
        room = Room(code=code, skill_name=skill_name)
        self._rooms[code] = room
        print(f"[Rooms] Created room {code} for skill '{skill_name}'")
        return room

    def get_room(self, code: str) -> Optional[Room]:
        """Get a room by code."""
        return self._rooms.get(code.upper())

    def join_room(self, code: str, user_id: str, display_name: str) -> Optional[ParticipantState]:
        """Join an existing room. Returns participant state or None if room not found."""
        room = self.get_room(code)
        if not room or not room.is_active:
            return None
        return room.add_participant(user_id, display_name)

    def close_room(self, code: str) -> Optional[dict]:
        """Close a room and return final comparison data."""
        room = self.get_room(code)
        if not room:
            return None
        room.is_active = False
        comparison = room.get_comparison_data()
        return comparison

    def list_rooms(self) -> list[dict]:
        """List all active rooms."""
        return [
            {
                "code": r.code,
                "skill": r.skill_name,
                "participants": r.participant_count,
                "active": r.is_active,
                "duration": round(time.time() - r.created_at, 1),
            }
            for r in self._rooms.values()
            if r.is_active
        ]

    def cleanup_stale(self, max_age: float = 3600):
        """Remove rooms older than max_age seconds."""
        now = time.time()
        stale = [c for c, r in self._rooms.items() if now - r.created_at > max_age]
        for code in stale:
            del self._rooms[code]
        if stale:
            print(f"[Rooms] Cleaned up {len(stale)} stale rooms")
