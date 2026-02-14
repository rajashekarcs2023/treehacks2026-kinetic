---
name: skill-progress
description: Skill progression tracking, expert references, goals, and memory management. Use when the user asks about their history, wants to set goals, manage references, or review progress.
---

# Skill Progress

## Expert References

- `list_references` — show all stored expert movements
- `load_reference_from_current` — capture live camera as reference
- `record_reference_start` / `record_reference_stop` — record a reference sequence

## Goal Management

12 preset goals available:
- **Spatial**: desk_watch, posture_coach, driver_monitor, study_focus, elderly_care, general
- **Coaching**: skill_coach, pt_rehab, fitness_trainer, dance_teacher, sports_coach, zero_shot_coach

- `get_current_goal` — see active goal
- `update_goal` — change goal via natural language
- `get_goal_presets` — list all presets

## Memory

- `save_observation` — store important observations for future sessions
- `get_observations` — recall past observations by tag

## Skill Document Parsing

- `parse_skill_document` — feed PT protocols, yoga guides, exercise docs
- Extracts target angles, phases, rep counts, safety boundaries
