"""
AI Expert Generator — Generate expert reference poses without needing a video.

Three tiers:
1. Canonical Templates — Biomechanically accurate joint angles for common exercises
2. Claude-Generated — Describe ANY skill → Claude generates ideal joint angles
3. DGX Motion Generation — Text-to-3D skeleton sequences via motion diffusion (when DGX available)

Usage:
    from aegis.ai_expert import get_ai_expert, list_canonical_exercises

    # Tier 1: Canonical template (instant, no API call)
    ref = get_ai_expert("squat")

    # Tier 2: Claude generates for any skill (async, ~1s)
    ref = await generate_expert_from_description("karate front kick")

    # Tier 3: DGX 3D motion generation (async, ~3s)
    ref = await generate_motion_from_dgx("squat", dgx_client)
"""

import math
import time
import json
import numpy as np
from typing import Optional
from dataclasses import dataclass

from aegis.pose_comparison import (
    NormalizedSkeleton, SkeletonSequence, KEY_ANGLES,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX, NOSE, LEFT_EAR,
    LEFT_INDEX, RIGHT_INDEX,
    normalize_skeleton, compute_joint_angles,
)


# ═══════════════════════════════════════════════════════════════════════
# TIER 1: CANONICAL POSE TEMPLATES
# ═══════════════════════════════════════════════════════════════════════
# Each exercise defines ideal joint angles at key phases.
# Angles in degrees. Phase order: preparation → execution → peak → recovery

CANONICAL_EXERCISES = {
    "squat": {
        "display_name": "Barbell Squat",
        "phases": {
            "standing": {
                "left_knee": 175, "right_knee": 175,
                "left_hip": 170, "right_hip": 170,
                "left_ankle": 90, "right_ankle": 90,
                "spine": 170, "torso_lean": 170,
                "left_shoulder": 90, "right_shoulder": 90,
                "left_elbow": 80, "right_elbow": 80,
            },
            "descent": {
                "left_knee": 130, "right_knee": 130,
                "left_hip": 120, "right_hip": 120,
                "left_ankle": 80, "right_ankle": 80,
                "spine": 155, "torso_lean": 140,
                "left_shoulder": 85, "right_shoulder": 85,
                "left_elbow": 80, "right_elbow": 80,
            },
            "bottom": {
                "left_knee": 85, "right_knee": 85,
                "left_hip": 80, "right_hip": 80,
                "left_ankle": 70, "right_ankle": 70,
                "spine": 140, "torso_lean": 110,
                "left_shoulder": 80, "right_shoulder": 80,
                "left_elbow": 80, "right_elbow": 80,
            },
            "ascent": {
                "left_knee": 130, "right_knee": 130,
                "left_hip": 120, "right_hip": 120,
                "left_ankle": 80, "right_ankle": 80,
                "spine": 155, "torso_lean": 140,
                "left_shoulder": 85, "right_shoulder": 85,
                "left_elbow": 80, "right_elbow": 80,
            },
        },
        "primary_angle": "left_knee",
        "coaching_cues": [
            "Keep chest up",
            "Knees track over toes",
            "Push through heels",
            "Brace your core",
        ],
    },
    "pushup": {
        "display_name": "Push-Up",
        "phases": {
            "top": {
                "left_elbow": 170, "right_elbow": 170,
                "left_shoulder": 70, "right_shoulder": 70,
                "left_hip": 175, "right_hip": 175,
                "left_knee": 175, "right_knee": 175,
                "spine": 175, "torso_lean": 175,
            },
            "descent": {
                "left_elbow": 120, "right_elbow": 120,
                "left_shoulder": 55, "right_shoulder": 55,
                "left_hip": 175, "right_hip": 175,
                "left_knee": 175, "right_knee": 175,
                "spine": 175, "torso_lean": 175,
            },
            "bottom": {
                "left_elbow": 85, "right_elbow": 85,
                "left_shoulder": 40, "right_shoulder": 40,
                "left_hip": 175, "right_hip": 175,
                "left_knee": 175, "right_knee": 175,
                "spine": 175, "torso_lean": 175,
            },
        },
        "primary_angle": "left_elbow",
        "coaching_cues": [
            "Keep body straight like a plank",
            "Elbows at 45 degrees",
            "Full range of motion",
            "Don't let hips sag",
        ],
    },
    "lunge": {
        "display_name": "Forward Lunge",
        "phases": {
            "standing": {
                "left_knee": 175, "right_knee": 175,
                "left_hip": 170, "right_hip": 170,
                "spine": 175, "torso_lean": 175,
            },
            "step_forward": {
                "left_knee": 130, "right_knee": 150,
                "left_hip": 130, "right_hip": 140,
                "spine": 175, "torso_lean": 160,
            },
            "bottom": {
                "left_knee": 90, "right_knee": 90,
                "left_hip": 90, "right_hip": 140,
                "spine": 175, "torso_lean": 170,
            },
        },
        "primary_angle": "left_knee",
        "coaching_cues": [
            "Front knee at 90 degrees",
            "Back knee nearly touches ground",
            "Torso stays upright",
            "Step far enough forward",
        ],
    },
    "plank": {
        "display_name": "Plank Hold",
        "phases": {
            "hold": {
                "left_elbow": 90, "right_elbow": 90,
                "left_shoulder": 90, "right_shoulder": 90,
                "left_hip": 175, "right_hip": 175,
                "left_knee": 175, "right_knee": 175,
                "spine": 175, "torso_lean": 175,
                "left_ankle": 90, "right_ankle": 90,
            },
        },
        "primary_angle": "left_hip",
        "coaching_cues": [
            "Straight line from head to heels",
            "Engage your core",
            "Don't let hips drop or pike up",
            "Breathe steadily",
        ],
    },
    "deadlift": {
        "display_name": "Deadlift",
        "phases": {
            "setup": {
                "left_knee": 140, "right_knee": 140,
                "left_hip": 100, "right_hip": 100,
                "spine": 160, "torso_lean": 100,
                "left_shoulder": 170, "right_shoulder": 170,
                "left_elbow": 175, "right_elbow": 175,
            },
            "pull": {
                "left_knee": 155, "right_knee": 155,
                "left_hip": 130, "right_hip": 130,
                "spine": 165, "torso_lean": 130,
                "left_shoulder": 170, "right_shoulder": 170,
                "left_elbow": 175, "right_elbow": 175,
            },
            "lockout": {
                "left_knee": 175, "right_knee": 175,
                "left_hip": 175, "right_hip": 175,
                "spine": 175, "torso_lean": 175,
                "left_shoulder": 160, "right_shoulder": 160,
                "left_elbow": 175, "right_elbow": 175,
            },
        },
        "primary_angle": "left_hip",
        "coaching_cues": [
            "Keep the bar close to your body",
            "Drive through your heels",
            "Neutral spine throughout",
            "Squeeze glutes at lockout",
        ],
    },
    "bicep_curl": {
        "display_name": "Bicep Curl",
        "phases": {
            "bottom": {
                "left_elbow": 170, "right_elbow": 170,
                "left_shoulder": 10, "right_shoulder": 10,
                "left_wrist": 170, "right_wrist": 170,
            },
            "mid": {
                "left_elbow": 90, "right_elbow": 90,
                "left_shoulder": 15, "right_shoulder": 15,
                "left_wrist": 170, "right_wrist": 170,
            },
            "top": {
                "left_elbow": 40, "right_elbow": 40,
                "left_shoulder": 20, "right_shoulder": 20,
                "left_wrist": 160, "right_wrist": 160,
            },
        },
        "primary_angle": "left_elbow",
        "coaching_cues": [
            "Keep elbows pinned to your sides",
            "Control the negative",
            "Full range of motion",
            "Don't swing your body",
        ],
    },
    "shoulder_press": {
        "display_name": "Shoulder Press",
        "phases": {
            "rack": {
                "left_elbow": 80, "right_elbow": 80,
                "left_shoulder": 90, "right_shoulder": 90,
                "left_wrist": 170, "right_wrist": 170,
                "spine": 175,
            },
            "press": {
                "left_elbow": 130, "right_elbow": 130,
                "left_shoulder": 140, "right_shoulder": 140,
                "left_wrist": 170, "right_wrist": 170,
                "spine": 175,
            },
            "lockout": {
                "left_elbow": 170, "right_elbow": 170,
                "left_shoulder": 175, "right_shoulder": 175,
                "left_wrist": 170, "right_wrist": 170,
                "spine": 175,
            },
        },
        "primary_angle": "left_elbow",
        "coaching_cues": [
            "Press straight overhead",
            "Keep core braced",
            "Don't arch your back",
            "Full lockout at top",
        ],
    },
    "warrior_pose": {
        "display_name": "Warrior II (Yoga)",
        "phases": {
            "hold": {
                "left_knee": 90, "right_knee": 175,
                "left_hip": 90, "right_hip": 160,
                "left_shoulder": 175, "right_shoulder": 175,
                "left_elbow": 175, "right_elbow": 175,
                "spine": 175, "torso_lean": 175,
            },
        },
        "primary_angle": "left_knee",
        "coaching_cues": [
            "Front knee at 90 degrees over ankle",
            "Arms parallel to ground",
            "Gaze over front hand",
            "Hips open to the side",
        ],
    },
    "tree_pose": {
        "display_name": "Tree Pose (Yoga)",
        "phases": {
            "hold": {
                "left_knee": 175, "right_knee": 90,
                "left_hip": 175, "right_hip": 90,
                "left_shoulder": 175, "right_shoulder": 175,
                "left_elbow": 175, "right_elbow": 175,
                "spine": 175, "torso_lean": 175,
            },
        },
        "primary_angle": "right_knee",
        "coaching_cues": [
            "Standing leg locked straight",
            "Foot on inner thigh, not knee",
            "Hands in prayer or overhead",
            "Fix gaze on a point",
        ],
    },
    "jumping_jack": {
        "display_name": "Jumping Jack",
        "phases": {
            "closed": {
                "left_shoulder": 10, "right_shoulder": 10,
                "left_elbow": 175, "right_elbow": 175,
                "left_knee": 175, "right_knee": 175,
                "hip_width": 20,
            },
            "open": {
                "left_shoulder": 170, "right_shoulder": 170,
                "left_elbow": 170, "right_elbow": 170,
                "left_knee": 170, "right_knee": 170,
                "hip_width": 60,
            },
        },
        "primary_angle": "left_shoulder",
        "coaching_cues": [
            "Arms fully extended overhead",
            "Land softly on toes",
            "Keep rhythm steady",
            "Full range of motion",
        ],
    },
}


def list_canonical_exercises() -> list[dict]:
    """List all available canonical exercise templates."""
    return [
        {
            "id": name,
            "display_name": ex["display_name"],
            "phases": list(ex["phases"].keys()),
            "primary_angle": ex["primary_angle"],
        }
        for name, ex in CANONICAL_EXERCISES.items()
    ]


# Semantic aliases: common names → canonical exercise key
# Covers synonyms, abbreviations, and related terms
EXERCISE_ALIASES = {
    # Squat variants
    "squat": "squat", "squats": "squat", "barbell_squat": "squat",
    "back_squat": "squat", "air_squat": "squat", "bodyweight_squat": "squat",
    "goblet_squat": "squat", "front_squat": "squat",
    # Pushup variants
    "pushup": "pushup", "push_up": "pushup", "pushups": "pushup",
    "push_ups": "pushup", "chest_press": "pushup", "press_up": "pushup",
    # Lunge variants
    "lunge": "lunge", "lunges": "lunge", "forward_lunge": "lunge",
    "walking_lunge": "lunge", "reverse_lunge": "lunge", "split_squat": "lunge",
    # Plank
    "plank": "plank", "forearm_plank": "plank", "high_plank": "plank",
    "elbow_plank": "plank",
    # Deadlift
    "deadlift": "deadlift", "deadlifts": "deadlift", "rdl": "deadlift",
    "romanian_deadlift": "deadlift", "hip_hinge": "deadlift",
    # Bicep curl
    "bicep_curl": "bicep_curl", "curl": "bicep_curl", "curls": "bicep_curl",
    "bicep_curls": "bicep_curl", "arm_curl": "bicep_curl",
    "dumbbell_curl": "bicep_curl", "hammer_curl": "bicep_curl",
    # Shoulder press
    "shoulder_press": "shoulder_press", "overhead_press": "shoulder_press",
    "ohp": "shoulder_press", "military_press": "shoulder_press",
    "press": "shoulder_press",
    # Yoga - Warrior
    "warrior_pose": "warrior_pose", "warrior": "warrior_pose",
    "warrior_2": "warrior_pose", "warrior_ii": "warrior_pose",
    "virabhadrasana": "warrior_pose",
    # Yoga - Tree
    "tree_pose": "tree_pose", "tree": "tree_pose",
    "vrksasana": "tree_pose",
    # Jumping jack
    "jumping_jack": "jumping_jack", "jumping_jacks": "jumping_jack",
    "star_jump": "jumping_jack", "star_jumps": "jumping_jack",
}


def get_ai_expert(exercise_name: str) -> Optional[dict]:
    """Get canonical expert template for an exercise.

    Uses semantic alias lookup (O(1)) — no fuzzy string matching.
    For unknown exercises, returns None → caller falls through to Claude generation.
    """
    name = exercise_name.lower().strip().replace(" ", "_").replace("-", "_")

    # Direct canonical match
    if name in CANONICAL_EXERCISES:
        return CANONICAL_EXERCISES[name]

    # Alias lookup
    canonical_key = EXERCISE_ALIASES.get(name)
    if canonical_key and canonical_key in CANONICAL_EXERCISES:
        return CANONICAL_EXERCISES[canonical_key]

    # Also try without underscores as alias key
    name_no_sep = name.replace("_", "")
    for alias, key in EXERCISE_ALIASES.items():
        if alias.replace("_", "") == name_no_sep:
            return CANONICAL_EXERCISES.get(key)

    # Not found → caller should use Claude generation (Tier 2)
    return None


def angles_to_skeleton(angles: dict[str, float],
                       frame_width: int = 640,
                       frame_height: int = 480) -> list:
    """Convert a set of ideal joint angles to approximate skeleton points.

    Uses inverse kinematics approximation to place joints in a plausible
    configuration matching the given angles. Returns 33 points in
    MediaPipe format: [(x, y, vis), ...] × 33.
    """
    # Start from hip center
    cx, cy = frame_width / 2, frame_height * 0.55

    # Approximate bone lengths (normalized to frame height)
    h = frame_height
    torso_len = h * 0.22
    upper_arm = h * 0.14
    lower_arm = h * 0.12
    upper_leg = h * 0.20
    lower_leg = h * 0.20
    foot_len = h * 0.06
    head_len = h * 0.10
    shoulder_w = h * 0.15
    hip_w = h * 0.08

    points = [(0.0, 0.0, 0.0)] * 33

    # Hips
    l_hip = (cx - hip_w, cy)
    r_hip = (cx + hip_w, cy)

    # Shoulders
    l_shoulder = (cx - shoulder_w, cy - torso_len)
    r_shoulder = (cx + shoulder_w, cy - torso_len)

    # Head
    nose = (cx, cy - torso_len - head_len)
    l_ear = (cx - head_len * 0.4, cy - torso_len - head_len * 0.8)

    # Convert angles to radians for placement
    def place_limb(origin, length, angle_deg, direction="down"):
        rad = math.radians(angle_deg)
        if direction == "down":
            dx = length * math.sin(rad * 0.3)
            dy = length * math.cos(rad * 0.1)
        elif direction == "up":
            dx = length * math.sin(rad * 0.3)
            dy = -length * math.cos(rad * 0.1)
        else:
            dx = length * math.cos(rad * 0.5)
            dy = length * math.sin(rad * 0.2)
        return (origin[0] + dx, origin[1] + dy)

    # Get angles with defaults
    def a(name, default=170):
        return angles.get(name, default)

    # Knees from hip angles
    knee_angle_l = a("left_knee")
    knee_angle_r = a("right_knee")
    hip_angle_l = a("left_hip")

    # Simple forward kinematics approximation
    l_knee = (l_hip[0] - upper_leg * 0.1, l_hip[1] + upper_leg * (knee_angle_l / 180))
    r_knee = (r_hip[0] + upper_leg * 0.1, r_hip[1] + upper_leg * (knee_angle_r / 180))

    l_ankle = (l_knee[0] - lower_leg * 0.05, l_knee[1] + lower_leg * 0.9)
    r_ankle = (r_knee[0] + lower_leg * 0.05, r_knee[1] + lower_leg * 0.9)

    l_foot = (l_ankle[0] - foot_len * 0.3, l_ankle[1] + foot_len * 0.5)
    r_foot = (r_ankle[0] + foot_len * 0.3, r_ankle[1] + foot_len * 0.5)

    # Arms from shoulder angles
    elbow_angle_l = a("left_elbow")
    elbow_angle_r = a("right_elbow")

    l_elbow = (l_shoulder[0] - upper_arm * 0.5, l_shoulder[1] + upper_arm * (elbow_angle_l / 180))
    r_elbow = (r_shoulder[0] + upper_arm * 0.5, r_shoulder[1] + upper_arm * (elbow_angle_r / 180))

    l_wrist = (l_elbow[0] - lower_arm * 0.3, l_elbow[1] + lower_arm * 0.5)
    r_wrist = (r_elbow[0] + lower_arm * 0.3, r_elbow[1] + lower_arm * 0.5)

    l_index = (l_wrist[0] - 5, l_wrist[1] + 10)
    r_index = (r_wrist[0] + 5, r_wrist[1] + 10)

    # Assign to MediaPipe 33-point format
    vis = 0.95
    points[NOSE] = (nose[0], nose[1], vis)
    points[LEFT_EAR] = (l_ear[0], l_ear[1], vis)
    points[7] = (l_ear[0], l_ear[1], vis)  # left ear (idx 7)
    points[8] = (cx + head_len * 0.4, cy - torso_len - head_len * 0.8, vis)  # right ear
    points[2] = (cx - head_len * 0.2, nose[1] - head_len * 0.2, vis)  # left eye
    points[5] = (cx + head_len * 0.2, nose[1] - head_len * 0.2, vis)  # right eye
    points[1] = (cx - head_len * 0.15, nose[1] - head_len * 0.25, vis)  # left eye inner
    points[3] = (cx - head_len * 0.25, nose[1] - head_len * 0.15, vis)  # left eye outer
    points[4] = (cx + head_len * 0.15, nose[1] - head_len * 0.25, vis)  # right eye inner
    points[6] = (cx + head_len * 0.25, nose[1] - head_len * 0.15, vis)  # right eye outer
    points[9] = (cx - head_len * 0.1, nose[1] + head_len * 0.1, vis)   # mouth left
    points[10] = (cx + head_len * 0.1, nose[1] + head_len * 0.1, vis)  # mouth right

    points[LEFT_SHOULDER] = (l_shoulder[0], l_shoulder[1], vis)
    points[RIGHT_SHOULDER] = (r_shoulder[0], r_shoulder[1], vis)
    points[LEFT_ELBOW] = (l_elbow[0], l_elbow[1], vis)
    points[RIGHT_ELBOW] = (r_elbow[0], r_elbow[1], vis)
    points[LEFT_WRIST] = (l_wrist[0], l_wrist[1], vis)
    points[RIGHT_WRIST] = (r_wrist[0], r_wrist[1], vis)
    points[LEFT_INDEX] = (l_index[0], l_index[1], vis)
    points[RIGHT_INDEX] = (r_index[0], r_index[1], vis)
    points[17] = (l_wrist[0] + 5, l_wrist[1] - 5, vis)  # left pinky
    points[18] = (r_wrist[0] - 5, r_wrist[1] - 5, vis)  # right pinky
    points[19] = (l_index[0] + 5, l_index[1], vis)  # left thumb
    points[20] = (r_index[0] - 5, r_index[1], vis)  # right thumb
    points[21] = (l_index[0], l_index[1] + 3, vis)  # left pinky (dup)
    points[22] = (r_index[0], r_index[1] + 3, vis)  # right pinky (dup)

    points[LEFT_HIP] = (l_hip[0], l_hip[1], vis)
    points[RIGHT_HIP] = (r_hip[0], r_hip[1], vis)
    points[LEFT_KNEE] = (l_knee[0], l_knee[1], vis)
    points[RIGHT_KNEE] = (r_knee[0], r_knee[1], vis)
    points[LEFT_ANKLE] = (l_ankle[0], l_ankle[1], vis)
    points[RIGHT_ANKLE] = (r_ankle[0], r_ankle[1], vis)
    points[LEFT_FOOT_INDEX] = (l_foot[0], l_foot[1], vis)
    points[RIGHT_FOOT_INDEX] = (r_foot[0], r_foot[1], vis)
    points[29] = (l_ankle[0] - 3, l_ankle[1] + 3, vis)  # left heel
    points[30] = (r_ankle[0] + 3, r_ankle[1] + 3, vis)  # right heel

    return points


def generate_expert_sequence(exercise_name: str,
                             frames_per_phase: int = 15) -> Optional[SkeletonSequence]:
    """Generate a full expert SkeletonSequence from canonical templates.

    Interpolates between phases to create smooth motion.
    Returns a SkeletonSequence ready for comparison.
    """
    template = get_ai_expert(exercise_name)
    if template is None:
        return None

    phases = template["phases"]
    phase_names = list(phases.keys())
    all_skeletons = []

    for i in range(len(phase_names)):
        current_angles = phases[phase_names[i]]
        next_angles = phases[phase_names[(i + 1) % len(phase_names)]]

        for f in range(frames_per_phase):
            t = f / frames_per_phase

            # Smooth interpolation (ease in-out)
            t_smooth = (1 - math.cos(t * math.pi)) / 2

            # Interpolate angles
            interp_angles = {}
            all_keys = set(list(current_angles.keys()) + list(next_angles.keys()))
            for key in all_keys:
                a_val = current_angles.get(key, 170)
                b_val = next_angles.get(key, 170)
                interp_angles[key] = a_val + (b_val - a_val) * t_smooth

            # Convert to skeleton points
            points = angles_to_skeleton(interp_angles)
            skel = normalize_skeleton(points)
            skel.timestamp = (i * frames_per_phase + f) / 30.0  # 30 FPS
            all_skeletons.append(skel)

    seq = SkeletonSequence(name=f"ai_expert_{exercise_name}", skeletons=all_skeletons)
    seq.metadata = {
        "source": "ai_generated",
        "exercise": exercise_name,
        "display_name": template["display_name"],
        "phases": phase_names,
        "coaching_cues": template.get("coaching_cues", []),
    }
    return seq


# ═══════════════════════════════════════════════════════════════════════
# TIER 2: CLAUDE-GENERATED EXPERT
# ═══════════════════════════════════════════════════════════════════════

ANGLE_GENERATION_PROMPT = """You are a biomechanics expert. Given a physical skill or exercise,
generate the ideal joint angles (in degrees) for each phase of the movement.

Return ONLY valid JSON in this exact format:
{
    "display_name": "Exercise Name",
    "phases": {
        "phase1_name": {"left_knee": 170, "right_knee": 170, "left_hip": 170, ...},
        "phase2_name": {"left_knee": 90, ...}
    },
    "primary_angle": "left_knee",
    "coaching_cues": ["Cue 1", "Cue 2", "Cue 3"]
}

Available joint angles: left_knee, right_knee, left_hip, right_hip,
left_shoulder, right_shoulder, left_elbow, right_elbow,
left_ankle, right_ankle, spine, neck, left_wrist, right_wrist,
torso_lean, hip_width.

Angles are measured at the vertex joint (e.g., left_knee = angle at knee joint).
170-180 = fully extended, 90 = right angle, <90 = acute.

Be biomechanically accurate. Include 2-4 phases for dynamic movements, 1 for static holds.
"""


async def generate_expert_from_claude(skill_description: str,
                                       anthropic_client=None) -> Optional[dict]:
    """Use Claude to generate ideal joint angles for ANY skill.

    Args:
        skill_description: Natural language description of the skill
        anthropic_client: Optional anthropic.AsyncAnthropic client

    Returns:
        Exercise template dict (same format as CANONICAL_EXERCISES entries)
    """
    if anthropic_client is None:
        try:
            import anthropic
            anthropic_client = anthropic.AsyncAnthropic()
        except Exception:
            return None

    try:
        response = await anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"{ANGLE_GENERATION_PROMPT}\n\nSkill: {skill_description}",
            }],
        )

        text = response.content[0].text.strip()
        # Extract JSON from response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        template = json.loads(text)

        # Validate structure
        if "phases" in template and "primary_angle" in template:
            return template

    except Exception as e:
        print(f"[AI Expert] Claude generation failed: {e}")

    return None


SEMANTIC_MAPPING_PROMPT = """You are a fitness expert. Given a user's description of an exercise or physical skill,
determine if it matches any of these canonical exercises:

{exercise_list}

If it matches one, respond with ONLY the exercise ID (e.g. "squat").
If it doesn't match any, respond with "GENERATE".

Examples:
- "back squat" → squat
- "barbell bench press" → pushup
- "sun salutation" → GENERATE
- "karate front kick" → GENERATE
- "overhead press" → shoulder_press

User input: {skill}
"""


async def resolve_exercise_with_claude(skill_description: str,
                                        anthropic_client=None) -> Optional[str]:
    """Use Claude to semantically map a skill description to a canonical exercise.

    Returns canonical exercise key if matched, None if should generate new.
    """
    if anthropic_client is None:
        try:
            import anthropic
            anthropic_client = anthropic.AsyncAnthropic()
        except Exception:
            return None

    exercise_list = "\n".join(
        f"- {key}: {ex['display_name']}" for key, ex in CANONICAL_EXERCISES.items()
    )

    try:
        response = await anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": SEMANTIC_MAPPING_PROMPT.format(
                    exercise_list=exercise_list,
                    skill=skill_description,
                ),
            }],
        )
        result = response.content[0].text.strip().lower().replace(" ", "_")

        if result != "generate" and result in CANONICAL_EXERCISES:
            return result
    except Exception as e:
        print(f"[AI Expert] Claude mapping failed: {e}")

    return None


async def generate_expert_sequence_from_description(
    skill_description: str,
    anthropic_client=None,
    frames_per_phase: int = 15,
) -> Optional[SkeletonSequence]:
    """Generate full expert sequence from natural language description.

    Three-step resolution:
    1. Alias lookup (instant, O(1))
    2. Claude semantic mapping → canonical exercise (async, ~0.5s)
    3. Claude angle generation for truly novel skills (async, ~1s)
    """
    # Step 1: Canonical / alias lookup (instant)
    seq = generate_expert_sequence(skill_description)
    if seq:
        return seq

    # Step 2: Claude semantic mapping — does this match a canonical exercise?
    canonical_key = await resolve_exercise_with_claude(skill_description, anthropic_client)
    if canonical_key:
        seq = generate_expert_sequence(canonical_key)
        if seq:
            # Cache the alias for next time
            alias = skill_description.lower().strip().replace(" ", "_")
            EXERCISE_ALIASES[alias] = canonical_key
            return seq

    # Step 3: Claude angle generation — truly novel skill
    template = await generate_expert_from_claude(skill_description, anthropic_client)
    if template is None:
        return None

    # Cache it in canonical exercises for future use
    name = skill_description.lower().strip().replace(" ", "_")
    CANONICAL_EXERCISES[name] = template

    return generate_expert_sequence(name, frames_per_phase)


# ═══════════════════════════════════════════════════════════════════════
# TIER 3: DGX MOTION GENERATION
# ═══════════════════════════════════════════════════════════════════════

async def generate_motion_from_dgx(skill_description: str,
                                    dgx_client=None) -> Optional[SkeletonSequence]:
    """Generate 3D motion sequence using motion diffusion model on DGX.

    When DGX is available with a motion generation model (MDM/MotionGPT),
    this sends a text prompt and receives a full 3D skeleton sequence.

    Falls back to Tier 1/2 if DGX motion generation is unavailable.
    """
    if dgx_client is None:
        # Fall back to canonical + Claude
        return generate_expert_sequence(skill_description)

    try:
        # Check if DGX has motion generation endpoint
        if hasattr(dgx_client, '_client') and dgx_client._client:
            client = dgx_client._client
            resp = await client.post(
                f"{dgx_client.base_url}/generate_motion",
                json={"prompt": skill_description, "num_frames": 60, "fps": 30},
                timeout=10.0,
            )

            if resp.status_code == 200:
                data = resp.json()
                keypoints = data.get("keypoints", [])
                if keypoints:
                    dgx_skeletons = []
                    for i, frame_kps in enumerate(keypoints):
                        points = [(kp["x"], kp["y"], kp.get("vis", 0.95))
                                  for kp in frame_kps]
                        if len(points) >= 33:
                            skel = normalize_skeleton(points)
                            skel.timestamp = i / 30.0
                            dgx_skeletons.append(skel)
                    if dgx_skeletons:
                        seq = SkeletonSequence(
                            name=f"dgx_motion_{skill_description}",
                            skeletons=dgx_skeletons,
                        )
                        seq.metadata = {
                            "source": "dgx_motion_generation",
                            "prompt": skill_description,
                            "gpu": data.get("gpu", "unknown"),
                            "generation_ms": data.get("generation_ms", 0),
                        }
                        return seq

    except Exception as e:
        print(f"[DGX Motion] Generation failed, falling back: {e}")

    # Fallback
    return generate_expert_sequence(skill_description)


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════

async def get_best_expert(skill: str, dgx_client=None,
                          anthropic_client=None) -> Optional[SkeletonSequence]:
    """Get the best available expert reference for a skill.

    Priority: DGX motion generation → Canonical template → Claude generation
    """
    # Tier 3: DGX (if available)
    if dgx_client and dgx_client.is_available:
        result = await generate_motion_from_dgx(skill, dgx_client)
        if result:
            return result

    # Tier 1: Canonical (instant)
    result = generate_expert_sequence(skill)
    if result:
        return result

    # Tier 2: Claude (async)
    return await generate_expert_sequence_from_description(
        skill, anthropic_client
    )
