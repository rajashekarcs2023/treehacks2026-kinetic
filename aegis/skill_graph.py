"""
AEGIS Skill Progression Graph — Dependency-aware coaching order.

A directed acyclic graph (DAG) of skills where:
- Each node is a skill (e.g., "squat", "pistol_squat")
- Edges represent prerequisites (squat → pistol_squat)
- Each skill has a proficiency level (0-100) based on coaching history
- The graph recommends what to practice next based on:
  1. Prerequisites met (all parents above threshold)
  2. PageRank-inspired priority (skills that unlock the most downstream skills)
  3. Current weakest links (skills below target proficiency)

Built-in skill trees:
  - Fitness fundamentals → intermediate → advanced
  - Yoga poses by difficulty
  - PT rehab progression
  - Custom (user-defined or AI-generated)

Persisted to disk so progress carries across sessions.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Skill:
    """A single skill node in the progression graph."""
    skill_id: str
    name: str
    description: str = ""
    category: str = "general"
    difficulty: int = 1  # 1-5
    primary_angle: str = "left_knee"
    target_score: float = 80.0  # proficiency target
    reference_name: str = ""  # linked expert reference

    # Progress tracking
    proficiency: float = 0.0  # 0-100, computed from coaching sessions
    attempts: int = 0
    best_score: float = 0.0
    last_practiced: float = 0.0
    scores_history: list[float] = field(default_factory=list)

    def update_proficiency(self, session_score: float):
        """Update proficiency based on a coaching session score."""
        self.attempts += 1
        self.scores_history.append(session_score)
        self.best_score = max(self.best_score, session_score)
        self.last_practiced = time.time()

        # Exponential moving average with recency bias
        if len(self.scores_history) == 1:
            self.proficiency = session_score
        else:
            alpha = 0.3  # weight of new score
            self.proficiency = alpha * session_score + (1 - alpha) * self.proficiency

    @property
    def mastered(self) -> bool:
        return self.proficiency >= self.target_score

    @property
    def status(self) -> str:
        if self.attempts == 0:
            return "not_started"
        if self.mastered:
            return "mastered"
        if self.proficiency >= self.target_score * 0.6:
            return "in_progress"
        return "beginner"

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "difficulty": self.difficulty,
            "primary_angle": self.primary_angle,
            "target_score": self.target_score,
            "reference_name": self.reference_name,
            "proficiency": round(self.proficiency, 1),
            "attempts": self.attempts,
            "best_score": round(self.best_score, 1),
            "last_practiced": self.last_practiced,
            "status": self.status,
            "mastered": self.mastered,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        skill = cls(
            skill_id=d["skill_id"],
            name=d["name"],
            description=d.get("description", ""),
            category=d.get("category", "general"),
            difficulty=d.get("difficulty", 1),
            primary_angle=d.get("primary_angle", "left_knee"),
            target_score=d.get("target_score", 80.0),
            reference_name=d.get("reference_name", ""),
        )
        skill.proficiency = d.get("proficiency", 0.0)
        skill.attempts = d.get("attempts", 0)
        skill.best_score = d.get("best_score", 0.0)
        skill.last_practiced = d.get("last_practiced", 0.0)
        skill.scores_history = d.get("scores_history", [])
        return skill


class SkillGraph:
    """Directed acyclic graph of skills with prerequisites and proficiency tracking."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.skills: dict[str, Skill] = {}
        self.edges: dict[str, list[str]] = {}  # parent → [children]
        self.reverse_edges: dict[str, list[str]] = {}  # child → [parents]

    def add_skill(self, skill: Skill):
        """Add a skill to the graph."""
        self.skills[skill.skill_id] = skill
        if skill.skill_id not in self.edges:
            self.edges[skill.skill_id] = []
        if skill.skill_id not in self.reverse_edges:
            self.reverse_edges[skill.skill_id] = []

    def add_prerequisite(self, prerequisite_id: str, skill_id: str):
        """Add a prerequisite edge: prerequisite must be mastered before skill."""
        if prerequisite_id not in self.edges:
            self.edges[prerequisite_id] = []
        if skill_id not in self.reverse_edges:
            self.reverse_edges[skill_id] = []

        if skill_id not in self.edges[prerequisite_id]:
            self.edges[prerequisite_id].append(skill_id)
        if prerequisite_id not in self.reverse_edges[skill_id]:
            self.reverse_edges[skill_id].append(prerequisite_id)

    def is_unlocked(self, skill_id: str) -> bool:
        """Check if a skill's prerequisites are all met."""
        parents = self.reverse_edges.get(skill_id, [])
        if not parents:
            return True  # No prerequisites = always unlocked
        return all(
            self.skills[p].mastered for p in parents if p in self.skills
        )

    def get_available_skills(self) -> list[Skill]:
        """Get skills that are unlocked but not yet mastered — ready to practice."""
        available = []
        for sid, skill in self.skills.items():
            if not skill.mastered and self.is_unlocked(sid):
                available.append(skill)
        return sorted(available, key=lambda s: (s.difficulty, -s.proficiency))

    def get_next_recommended(self, top_n: int = 3) -> list[dict]:
        """Get top N recommended skills to practice next.

        Ranking factors:
        1. Must be unlocked (prerequisites met)
        2. Priority to skills that unlock the most downstream skills
        3. Skills closest to mastery get a boost (finish what you started)
        4. Skills not practiced recently get a boost
        """
        available = self.get_available_skills()
        if not available:
            return []

        scored = []
        for skill in available:
            score = 0.0

            # Factor 1: Downstream unlock count (PageRank-inspired)
            downstream = self._count_downstream(skill.skill_id)
            score += downstream * 10

            # Factor 2: Close to mastery bonus
            if skill.proficiency > 0:
                closeness = skill.proficiency / skill.target_score
                score += closeness * 20

            # Factor 3: Recency penalty (recently practiced = lower priority)
            if skill.last_practiced > 0:
                hours_ago = (time.time() - skill.last_practiced) / 3600
                score += min(hours_ago, 24)  # cap at 24h boost

            # Factor 4: Lower difficulty preferred for beginners
            if skill.attempts == 0:
                score += (6 - skill.difficulty) * 5

            scored.append((skill, score))

        scored.sort(key=lambda x: -x[1])

        return [
            {
                **skill.to_dict(),
                "recommendation_score": round(score, 1),
                "unlocks": self.edges.get(skill.skill_id, []),
                "prerequisites_met": True,
            }
            for skill, score in scored[:top_n]
        ]

    def _count_downstream(self, skill_id: str, visited: set = None) -> int:
        """Count how many skills are downstream (transitively unlocked)."""
        if visited is None:
            visited = set()
        if skill_id in visited:
            return 0
        visited.add(skill_id)
        children = self.edges.get(skill_id, [])
        count = len(children)
        for child in children:
            count += self._count_downstream(child, visited)
        return count

    def update_skill_proficiency(self, skill_id: str, session_score: float):
        """Update a skill's proficiency after a coaching session."""
        if skill_id in self.skills:
            self.skills[skill_id].update_proficiency(session_score)

    def get_skill_tree(self) -> dict:
        """Get the full skill tree for visualization."""
        nodes = []
        links = []

        for sid, skill in self.skills.items():
            nodes.append({
                **skill.to_dict(),
                "unlocked": self.is_unlocked(sid),
                "children": self.edges.get(sid, []),
                "parents": self.reverse_edges.get(sid, []),
            })

        for parent, children in self.edges.items():
            for child in children:
                links.append({
                    "source": parent,
                    "target": child,
                    "met": self.skills[parent].mastered if parent in self.skills else False,
                })

        return {
            "name": self.name,
            "total_skills": len(self.skills),
            "mastered": sum(1 for s in self.skills.values() if s.mastered),
            "in_progress": sum(1 for s in self.skills.values() if s.status == "in_progress"),
            "nodes": nodes,
            "links": links,
        }

    def get_progress_summary(self) -> dict:
        """Overall progress summary."""
        total = len(self.skills)
        mastered = sum(1 for s in self.skills.values() if s.mastered)
        in_progress = sum(1 for s in self.skills.values() if s.status == "in_progress")
        not_started = sum(1 for s in self.skills.values() if s.status == "not_started")

        return {
            "graph_name": self.name,
            "total_skills": total,
            "mastered": mastered,
            "in_progress": in_progress,
            "not_started": not_started,
            "completion_pct": round(mastered / max(total, 1) * 100, 1),
            "total_attempts": sum(s.attempts for s in self.skills.values()),
            "avg_proficiency": round(
                sum(s.proficiency for s in self.skills.values()) / max(total, 1), 1
            ),
        }

    # ── Persistence ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "skills": {sid: s.to_dict() for sid, s in self.skills.items()},
            "edges": self.edges,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkillGraph":
        graph = cls(name=d.get("name", "default"))
        for sid, sdata in d.get("skills", {}).items():
            graph.add_skill(Skill.from_dict(sdata))
        for parent, children in d.get("edges", {}).items():
            for child in children:
                graph.add_prerequisite(parent, child)
        return graph

    def save(self, filepath: str = None) -> str:
        if filepath is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "skill_graphs"
            )
            os.makedirs(data_dir, exist_ok=True)
            filepath = os.path.join(data_dir, f"{self.name}.json")
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath

    @classmethod
    def load(cls, name: str = "default") -> Optional["SkillGraph"]:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "skill_graphs"
        )
        filepath = os.path.join(data_dir, f"{name}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath) as f:
            return cls.from_dict(json.load(f))


# ═══════════════════════════════════════════════════════════════════════
# BUILT-IN SKILL TREES
# ═══════════════════════════════════════════════════════════════════════

def create_fitness_graph() -> SkillGraph:
    """Create the fitness fundamentals skill tree."""
    g = SkillGraph(name="fitness")

    # ── Level 1: Fundamentals ────────────────────────────────────────
    g.add_skill(Skill(
        skill_id="bodyweight_squat", name="Bodyweight Squat",
        description="Basic air squat with proper depth and knee tracking",
        category="lower_body", difficulty=1, primary_angle="left_knee",
    ))
    g.add_skill(Skill(
        skill_id="wall_pushup", name="Wall Push-Up",
        description="Push-up against a wall for upper body foundation",
        category="upper_body", difficulty=1, primary_angle="left_elbow",
    ))
    g.add_skill(Skill(
        skill_id="dead_hang", name="Dead Hang",
        description="Hang from a bar with straight arms for grip and shoulder health",
        category="upper_body", difficulty=1, primary_angle="left_shoulder",
    ))
    g.add_skill(Skill(
        skill_id="glute_bridge", name="Glute Bridge",
        description="Lying hip extension for posterior chain activation",
        category="lower_body", difficulty=1, primary_angle="left_hip",
    ))
    g.add_skill(Skill(
        skill_id="plank", name="Plank",
        description="Isometric core hold with neutral spine",
        category="core", difficulty=1, primary_angle="left_hip",
    ))

    # ── Level 2: Intermediate ────────────────────────────────────────
    g.add_skill(Skill(
        skill_id="goblet_squat", name="Goblet Squat",
        description="Weighted front squat with upright torso",
        category="lower_body", difficulty=2, primary_angle="left_knee",
    ))
    g.add_skill(Skill(
        skill_id="standard_pushup", name="Standard Push-Up",
        description="Full push-up with chest to ground",
        category="upper_body", difficulty=2, primary_angle="left_elbow",
    ))
    g.add_skill(Skill(
        skill_id="lunge", name="Forward Lunge",
        description="Alternating forward lunges with knee over ankle",
        category="lower_body", difficulty=2, primary_angle="left_knee",
    ))
    g.add_skill(Skill(
        skill_id="hip_hinge", name="Hip Hinge",
        description="Romanian deadlift pattern with flat back",
        category="lower_body", difficulty=2, primary_angle="left_hip",
    ))
    g.add_skill(Skill(
        skill_id="side_plank", name="Side Plank",
        description="Lateral core stability hold",
        category="core", difficulty=2, primary_angle="left_hip",
    ))

    # ── Level 3: Advanced ────────────────────────────────────────────
    g.add_skill(Skill(
        skill_id="pistol_squat", name="Pistol Squat",
        description="Single-leg squat with full depth",
        category="lower_body", difficulty=4, primary_angle="left_knee",
    ))
    g.add_skill(Skill(
        skill_id="diamond_pushup", name="Diamond Push-Up",
        description="Close-grip push-up for tricep emphasis",
        category="upper_body", difficulty=3, primary_angle="left_elbow",
    ))
    g.add_skill(Skill(
        skill_id="bulgarian_split", name="Bulgarian Split Squat",
        description="Rear-foot-elevated single-leg squat",
        category="lower_body", difficulty=3, primary_angle="left_knee",
    ))

    # ── Prerequisites ────────────────────────────────────────────────
    # Level 1 → Level 2
    g.add_prerequisite("bodyweight_squat", "goblet_squat")
    g.add_prerequisite("bodyweight_squat", "lunge")
    g.add_prerequisite("wall_pushup", "standard_pushup")
    g.add_prerequisite("glute_bridge", "hip_hinge")
    g.add_prerequisite("plank", "side_plank")

    # Level 2 → Level 3
    g.add_prerequisite("goblet_squat", "pistol_squat")
    g.add_prerequisite("lunge", "pistol_squat")
    g.add_prerequisite("standard_pushup", "diamond_pushup")
    g.add_prerequisite("lunge", "bulgarian_split")
    g.add_prerequisite("goblet_squat", "bulgarian_split")

    return g


def create_yoga_graph() -> SkillGraph:
    """Create a yoga progression skill tree."""
    g = SkillGraph(name="yoga")

    # ── Foundation poses ─────────────────────────────────────────────
    g.add_skill(Skill(
        skill_id="mountain", name="Mountain Pose (Tadasana)",
        description="Standing alignment foundation",
        category="standing", difficulty=1, primary_angle="left_knee",
    ))
    g.add_skill(Skill(
        skill_id="downdog", name="Downward Dog",
        description="Inverted V-shape full body stretch",
        category="inversion", difficulty=1, primary_angle="left_shoulder",
    ))
    g.add_skill(Skill(
        skill_id="child", name="Child's Pose",
        description="Resting pose with forward fold",
        category="floor", difficulty=1, primary_angle="left_hip",
    ))

    # ── Intermediate ─────────────────────────────────────────────────
    g.add_skill(Skill(
        skill_id="warrior1", name="Warrior I (Virabhadrasana I)",
        description="Lunging pose with arms overhead",
        category="standing", difficulty=2, primary_angle="left_knee",
    ))
    g.add_skill(Skill(
        skill_id="warrior2", name="Warrior II (Virabhadrasana II)",
        description="Wide stance with arms extended to sides",
        category="standing", difficulty=2, primary_angle="left_knee",
    ))
    g.add_skill(Skill(
        skill_id="tree", name="Tree Pose (Vrksasana)",
        description="Single-leg balance with foot on inner thigh",
        category="balance", difficulty=2, primary_angle="left_knee",
    ))
    g.add_skill(Skill(
        skill_id="triangle", name="Triangle Pose",
        description="Side stretch with straight legs",
        category="standing", difficulty=2, primary_angle="left_hip",
    ))

    # ── Advanced ─────────────────────────────────────────────────────
    g.add_skill(Skill(
        skill_id="warrior3", name="Warrior III",
        description="Single-leg balance with body parallel to ground",
        category="balance", difficulty=3, primary_angle="left_hip",
    ))
    g.add_skill(Skill(
        skill_id="crow", name="Crow Pose (Bakasana)",
        description="Arm balance with knees on triceps",
        category="arm_balance", difficulty=4, primary_angle="left_elbow",
    ))
    g.add_skill(Skill(
        skill_id="headstand", name="Headstand (Sirsasana)",
        description="Inverted balance on head and forearms",
        category="inversion", difficulty=5, primary_angle="left_shoulder",
    ))

    # ── Prerequisites ────────────────────────────────────────────────
    g.add_prerequisite("mountain", "warrior1")
    g.add_prerequisite("mountain", "warrior2")
    g.add_prerequisite("mountain", "tree")
    g.add_prerequisite("mountain", "triangle")
    g.add_prerequisite("warrior1", "warrior3")
    g.add_prerequisite("warrior2", "warrior3")
    g.add_prerequisite("tree", "warrior3")
    g.add_prerequisite("downdog", "crow")
    g.add_prerequisite("warrior3", "crow")
    g.add_prerequisite("downdog", "headstand")
    g.add_prerequisite("crow", "headstand")

    return g


def create_pt_rehab_graph(focus: str = "knee") -> SkillGraph:
    """Create a PT rehabilitation progression graph.

    Args:
        focus: Body area — "knee", "shoulder", or "back"
    """
    g = SkillGraph(name=f"pt_rehab_{focus}")

    if focus == "knee":
        # Phase 1: ROM and activation
        g.add_skill(Skill(
            skill_id="seated_knee_ext", name="Seated Knee Extension",
            description="Straighten knee from 90° to full extension while seated",
            category="rom", difficulty=1, primary_angle="left_knee", target_score=70.0,
        ))
        g.add_skill(Skill(
            skill_id="quad_set", name="Quad Set",
            description="Isometric quad contraction with straight leg",
            category="activation", difficulty=1, primary_angle="left_knee", target_score=70.0,
        ))
        g.add_skill(Skill(
            skill_id="heel_slide", name="Heel Slide",
            description="Slide heel toward glutes while lying down",
            category="rom", difficulty=1, primary_angle="left_knee", target_score=70.0,
        ))

        # Phase 2: Strength
        g.add_skill(Skill(
            skill_id="mini_squat", name="Mini Squat (Quarter)",
            description="Squat to 45° knee bend with support",
            category="strength", difficulty=2, primary_angle="left_knee", target_score=75.0,
        ))
        g.add_skill(Skill(
            skill_id="step_up", name="Step Up",
            description="Step onto low platform with controlled descent",
            category="strength", difficulty=2, primary_angle="left_knee", target_score=75.0,
        ))

        # Phase 3: Function
        g.add_skill(Skill(
            skill_id="full_squat", name="Full Squat",
            description="Squat to 90° with even weight distribution",
            category="function", difficulty=3, primary_angle="left_knee", target_score=80.0,
        ))
        g.add_skill(Skill(
            skill_id="single_leg_balance", name="Single Leg Balance",
            description="Stand on affected leg for 30 seconds",
            category="balance", difficulty=3, primary_angle="left_knee", target_score=80.0,
        ))

        # Prerequisites
        g.add_prerequisite("seated_knee_ext", "mini_squat")
        g.add_prerequisite("quad_set", "mini_squat")
        g.add_prerequisite("heel_slide", "step_up")
        g.add_prerequisite("mini_squat", "full_squat")
        g.add_prerequisite("step_up", "full_squat")
        g.add_prerequisite("mini_squat", "single_leg_balance")

    elif focus == "shoulder":
        g.add_skill(Skill(
            skill_id="pendulum", name="Pendulum Swing",
            description="Gentle arm swing using gravity",
            category="rom", difficulty=1, primary_angle="left_shoulder", target_score=70.0,
        ))
        g.add_skill(Skill(
            skill_id="wall_walk", name="Wall Walk",
            description="Walk fingers up wall for overhead reach",
            category="rom", difficulty=1, primary_angle="left_shoulder", target_score=70.0,
        ))
        g.add_skill(Skill(
            skill_id="band_external", name="Band External Rotation",
            description="Rotator cuff strengthening with resistance band",
            category="strength", difficulty=2, primary_angle="left_elbow", target_score=75.0,
        ))
        g.add_skill(Skill(
            skill_id="overhead_press", name="Light Overhead Press",
            description="Press light weight overhead with full ROM",
            category="strength", difficulty=3, primary_angle="left_shoulder", target_score=80.0,
        ))

        g.add_prerequisite("pendulum", "band_external")
        g.add_prerequisite("wall_walk", "band_external")
        g.add_prerequisite("band_external", "overhead_press")

    return g


# ═══════════════════════════════════════════════════════════════════════
# GRAPH STORE — Manage multiple skill graphs
# ═══════════════════════════════════════════════════════════════════════

class GraphStore:
    """Manages multiple skill graphs (fitness, yoga, PT, custom)."""

    def __init__(self):
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "skill_graphs"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        self._cache: dict[str, SkillGraph] = {}

    def get_or_create(self, name: str) -> SkillGraph:
        """Load a graph or create a built-in one."""
        if name in self._cache:
            return self._cache[name]

        # Try loading from disk
        graph = SkillGraph.load(name)
        if graph:
            self._cache[name] = graph
            return graph

        # Create built-in graphs
        if name == "fitness":
            graph = create_fitness_graph()
        elif name == "yoga":
            graph = create_yoga_graph()
        elif name.startswith("pt_rehab"):
            focus = name.split("_")[-1] if "_" in name else "knee"
            graph = create_pt_rehab_graph(focus)
        else:
            graph = SkillGraph(name=name)

        graph.save()
        self._cache[name] = graph
        return graph

    def save_graph(self, graph: SkillGraph):
        """Save a graph to disk."""
        graph.save()
        self._cache[graph.name] = graph

    def list_graphs(self) -> list[dict]:
        """List all available skill graphs."""
        graphs = []
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".json"):
                name = fname[:-5]
                graph = self.get_or_create(name)
                graphs.append(graph.get_progress_summary())
        # Include built-in graphs even if not yet on disk
        for builtin in ["fitness", "yoga", "pt_rehab_knee", "pt_rehab_shoulder"]:
            if builtin not in [g["graph_name"] for g in graphs]:
                graph = self.get_or_create(builtin)
                graphs.append(graph.get_progress_summary())
        return graphs

    def delete_graph(self, name: str) -> bool:
        filepath = os.path.join(self.data_dir, f"{name}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            self._cache.pop(name, None)
            return True
        return False
