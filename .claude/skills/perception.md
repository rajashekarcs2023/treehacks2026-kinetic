---
name: spatial-perception
description: Spatial awareness and scene understanding via computer vision. Use when analyzing what's in the scene, tracking people, checking posture, or monitoring activities and zones.
---

# Spatial Perception

## Scene Analysis

- `get_spatial_state` — full JSON state (persons, objects, poses)
- `get_spatial_summary` — human-readable scene summary
- `get_scene_changes` — what changed in the last N seconds
- `get_objects_in_scene` — detected objects (80 COCO classes)
- `count_objects` — count of specific object class

## Person Tracking

- Each person has a persistent `track_id`
- `get_person_detail` — deep info: position, velocity, activity, pose
- People are tracked across frames via ByteTrack

## Pose Analysis

- 33 MediaPipe landmarks per person
- `analyze_posture` — computed metrics (spine angle, shoulder tilt, knee angles)
- `get_pose_landmarks` — raw landmark positions
- `check_body_alignment` — exercise-specific alignment checks

## Activity Recognition

Activities detected: standing, sitting, walking, running, fallen, waving, reaching, crouching

- `get_activity_timeline` — recent activity history
- `get_time_in_activity` — duration in specific activity

## Zone Monitoring

- `set_watch_zone` — define rectangular danger/interest zones
- `check_zone_status` — who is in/near zones
- Useful for safety monitoring, boundary detection
