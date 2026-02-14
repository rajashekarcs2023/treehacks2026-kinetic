---
name: skill-coaching
description: Real-time physical skill coaching with expert motion transfer. Use when the user wants to practice a movement, get form feedback, or be coached on any physical skill like squats, yoga, PT exercises.
---

# Skill Coaching

## Coaching Workflow

1. **Start session**: Use `start_coaching_session` with skill name and optional expert reference
2. **Monitor form**: Use `compare_to_reference` every few seconds during movement
3. **Voice feedback**: Use `speak_to_user` for SHORT real-time cues ("Knees wider", "Good!")
4. **Track reps**: Use `get_rep_count` and announce completions
5. **Safety check**: Use `detect_compensation_patterns` periodically
6. **Quality check**: Use `get_movement_quality_analysis` for smoothness/symmetry
7. **End session**: Use `end_coaching_session` for full summary

## Voice Cue Guidelines

- During movement: MAX 4 words ("Deeper!", "Chest up!", "Great form!")
- Between reps: One sentence corrections
- After set: Detailed feedback with specific angles

## Joint Angles Available

- `left_knee`, `right_knee` — squats, lunges, leg exercises
- `left_hip`, `right_hip` — deadlifts, hip hinges
- `left_elbow`, `right_elbow` — curls, presses
- `left_shoulder`, `right_shoulder` — overhead movements
- `left_ankle`, `right_ankle` — calf raises, balance

## Zero-Shot Coaching (No Reference)

When no expert reference exists:
1. Use `analyze_skill_from_description` with the skill name
2. Apply biomechanics knowledge to determine ideal angles
3. Compare user's current angles to your determined ideals
4. Coach corrections via `speak_to_user`

## Common Coaching Scenarios

### Squat
- Primary angle: `left_knee` (rep detection)
- Watch: knee tracking over toes, hip depth, back angle
- Common mistake: knees caving inward (check L/R knee symmetry)

### Yoga Warrior II
- Watch: front knee at 90°, back leg straight, arms level
- Common mistake: front knee past ankle, torso leaning

### PT Rehab
- ALWAYS check `detect_compensation_patterns` — injury risk is highest here
- Be gentle with corrections
- Track smaller ROM improvements over sessions
