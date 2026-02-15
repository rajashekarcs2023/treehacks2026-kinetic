# AEGIS — Use Cases, Goals & Tool Call Flows

> Every use case maps to a goal. Every goal drives specific tool calls.
> This document shows exactly what happens inside AEGIS for each scenario.

---

## How It Works

```
User sets goal (voice, text, or preset)
        ↓
Claude receives goal-specific system prompt
        ↓
Monitor triggers agent every N seconds OR on events
        ↓
Agent picks tools from 25 available based on goal context
        ↓
Tools read spatial state → Agent reasons → Agent acts (alert, speak, save)
```

---

## Goal 1: Desk Guardian (`desk_watch`)

**User says:** "Watch my desk and alert if anyone approaches"

### Use Cases

#### UC1.1 — Someone approaches the desk
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_scene_changes()` | Detects "new person entered scene" |
| 2 | `get_person_detail(person_id)` | Gets position, speed, direction, activity |
| 3 | `check_zone_status()` | Checks if person is approaching the desk zone |
| 4 | `send_telegram_alert("Someone approaching your desk")` | Alerts owner |
| 5 | `capture_photo(annotated=True)` | Captures evidence with bounding boxes |
| 6 | `speak_to_user("Alert: someone is near your desk", urgency="high")` | Voice warning |
| 7 | `save_observation("Person 3 approached desk at 2:15pm", tags=["intrusion"])` | Logs it |

#### UC1.2 — Owner leaves the desk
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_scene_changes()` | Detects "person left scene" |
| 2 | `get_objects_in_scene()` | Notes what's on the desk (laptop, phone, bag) |
| 3 | `set_watch_zone(x1, y1, x2, y2, label="Desk Area")` | Sets up perimeter around desk |
| 4 | `save_observation("Owner left. Objects: laptop, backpack", tags=["owner_left"])` | Records state |
| 5 | `send_telegram_alert("You left your desk. I'm watching: laptop, backpack visible")` | Confirms |

#### UC1.3 — Someone reaches toward desk objects
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_spatial_state()` | Full scene snapshot |
| 2 | `get_pose_landmarks(person_id)` | Checks arm extension (wrist landmarks) |
| 3 | `analyze_posture(person_id)` | Arm reach angle toward desk |
| 4 | `capture_photo(annotated=True)` | Evidence |
| 5 | `send_telegram_alert("⚠️ Someone is reaching toward your desk!", include_photo=True)` | Urgent alert |
| 6 | `speak_to_user("Please don't touch that", urgency="high")` | Verbal deterrent |

#### UC1.4 — Periodic all-clear check
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_scene_changes()` | No meaningful changes |
| 2 | `get_objects_in_scene()` | Same objects as before |
| 3 | *(no alert — nothing changed)* | Agent stays quiet |

**Tools used by this goal:** `get_scene_changes`, `get_person_detail`, `get_objects_in_scene`, `check_zone_status`, `set_watch_zone`, `get_pose_landmarks`, `analyze_posture`, `capture_photo`, `send_telegram_alert`, `speak_to_user`, `save_observation`

---

## Goal 2: Posture & Form Coach (`posture_coach`)

**User says:** "Coach my posture while I work" or "Watch my squat form"

### Use Cases

#### UC2.1 — Sitting posture check (desk worker)
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_spatial_state()` | Confirms person is sitting |
| 2 | `analyze_posture(person_id)` | Returns shoulder_tilt, spine_angle, head_forward |
| 3 | `get_pose_landmarks(person_id)` | Raw landmarks for detailed analysis |
| 4 | `check_body_alignment(person_id)` | Deviation from ideal sitting posture |
| 5 | `speak_to_user("Your shoulders are uneven — try leveling them")` | Real-time voice coaching |
| 6 | `save_observation("Posture check: 15° shoulder tilt, head forward", tags=["posture"])` | Track over time |

#### UC2.2 — Exercise form correction (squats)
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_spatial_state()` | Person detected, activity=standing/crouching |
| 2 | `get_pose_landmarks(person_id)` | Full 33-point skeleton |
| 3 | `check_body_alignment(person_id, exercise_type="squat")` | Knee-over-toe, back angle, depth |
| 4 | `analyze_posture(person_id)` | Knee angle, hip angle measurements |
| 5 | `speak_to_user("Good depth! But push your knees out more — they're caving in")` | Voice correction |
| 6 | `get_activity_timeline(person_id)` | How many reps, rest periods |
| 7 | `save_observation("Squat set: 8 reps, knee cave on last 3", tags=["exercise"])` | Session log |

#### UC2.3 — Yoga pose hold
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_pose_landmarks(person_id)` | Full skeleton |
| 2 | `check_body_alignment(person_id, exercise_type="yoga")` | Balance, symmetry |
| 3 | `analyze_posture(person_id)` | Joint angles |
| 4 | `get_time_in_activity(person_id, activity="standing")` | How long holding pose |
| 5 | `speak_to_user("Beautiful warrior pose! Hold for 10 more seconds")` | Encouragement |

#### UC2.4 — Progress report
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_observations(tag="posture")` | Recall past posture checks |
| 2 | `get_session_stats()` | Duration, checks done |
| 3 | `speak_to_user("Your shoulder alignment improved 20% over the last hour")` | Progress feedback |
| 4 | `send_telegram_alert("Session summary: 45 min, posture improved")` | Async report |

**Tools used by this goal:** `get_spatial_state`, `analyze_posture`, `get_pose_landmarks`, `check_body_alignment`, `get_activity_timeline`, `get_time_in_activity`, `get_session_stats`, `get_observations`, `save_observation`, `speak_to_user`, `send_telegram_alert`

---

## Goal 3: Driver Alertness Monitor (`driver_monitor`)

**User says:** "Keep me awake while driving"

### Use Cases

#### UC3.1 — Early drowsiness signs (head nodding)
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_spatial_state()` | Person detected, sitting |
| 2 | `get_pose_landmarks(person_id)` | Head position (landmark 0) vs shoulders (11/12) |
| 3 | `analyze_posture(person_id)` | Head tilt angle — forward tilt = nodding |
| 4 | `speak_to_user("Hey, you seem a bit tired. Consider a break.", urgency="normal")` | Gentle alert |
| 5 | `save_observation("Slight head nod detected", tags=["drowsiness", "level1"])` | Track pattern |

#### UC3.2 — Repeated drowsiness (escalation)
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_pose_landmarks(person_id)` | Head dropping further |
| 2 | `get_observations(tag="drowsiness")` | 3 drowsiness events in last 5 min |
| 3 | `speak_to_user("WARNING: Signs of drowsiness. Please pull over.", urgency="high")` | Escalated |
| 4 | `send_telegram_alert("⚠️ Driver showing repeated drowsiness signs")` | Alert emergency contact |

#### UC3.3 — Critical: falling asleep
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_spatial_state()` | Activity = sitting, near-zero movement |
| 2 | `get_pose_landmarks(person_id)` | Head fully dropped, eyes-level landmarks down |
| 3 | `speak_to_user("DANGER! You appear to be falling asleep! PULL OVER NOW!", urgency="critical")` | Maximum urgency |
| 4 | `send_telegram_alert("🚨 CRITICAL: Driver appears to be falling asleep!", include_photo=True)` | Emergency alert |
| 5 | `capture_photo()` | Evidence for emergency contact |

#### UC3.4 — Distraction detection (looking at phone)
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_objects_in_scene(class_filter="cell phone")` | Phone detected in hand |
| 2 | `get_pose_landmarks(person_id)` | Head turned away from forward |
| 3 | `speak_to_user("Eyes on the road! Put your phone down.", urgency="high")` | Immediate |
| 4 | `save_observation("Driver distracted by phone", tags=["distraction"])` | Log |

**Tools used by this goal:** `get_spatial_state`, `get_pose_landmarks`, `analyze_posture`, `get_objects_in_scene`, `get_observations`, `save_observation`, `speak_to_user`, `send_telegram_alert`, `capture_photo`

---

## Goal 4: Study Focus Assistant (`study_focus`)

**User says:** "Help me stay focused while studying"

### Use Cases

#### UC4.1 — Phone distraction detected
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_scene_changes()` | Object change detected |
| 2 | `get_objects_in_scene(class_filter="cell phone")` | Phone now visible / in hand |
| 3 | `get_time_in_activity(person_id, activity="sitting")` | Was studying for 18 min |
| 4 | `speak_to_user("Hey, I noticed you picked up your phone. Back to the books?")` | Gentle nudge |
| 5 | `save_observation("Phone distraction at 18 min mark", tags=["distraction", "phone"])` | Track |

#### UC4.2 — Left the study area
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_scene_changes()` | Person left scene |
| 2 | `get_time_in_activity(person_id, activity="sitting")` | Was sitting for 45 min |
| 3 | `save_observation("Left desk after 45 min focus session", tags=["break"])` | Log |
| 4 | `get_current_time()` | 3:45 PM |
| 5 | `send_telegram_alert("Good focus session! 45 min. Taking a break at 3:45 PM")` | Summary |

#### UC4.3 — Pomodoro timer reminder
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_time_in_activity(person_id, activity="sitting")` | 25 minutes continuous |
| 2 | `get_current_time()` | Check time |
| 3 | `speak_to_user("25 minutes done! Time for a 5-minute break. Stretch!")` | Pomodoro alert |
| 4 | `save_observation("Pomodoro cycle 3 complete", tags=["pomodoro"])` | Track cycles |

#### UC4.4 — Return from break
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_scene_changes()` | Person entered scene |
| 2 | `get_observations(tag="break")` | Last break was 7 min ago |
| 3 | `speak_to_user("Welcome back! Let's get another focus session going.")` | Motivate |

#### UC4.5 — End-of-day study report
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_session_stats()` | Total session info |
| 2 | `get_observations(tag="distraction")` | Count distractions |
| 3 | `get_observations(tag="pomodoro")` | Count completed cycles |
| 4 | `send_telegram_alert("Study report: 3h 20m total, 6 Pomodoro cycles, 4 phone distractions")` | Daily summary |

**Tools used by this goal:** `get_scene_changes`, `get_objects_in_scene`, `get_time_in_activity`, `get_current_time`, `get_session_stats`, `get_observations`, `save_observation`, `speak_to_user`, `send_telegram_alert`

---

## Goal 5: Elderly Care Guardian (`elderly_care`)

**User says:** "Watch over my grandmother"

### Use Cases

#### UC5.1 — FALL DETECTED (highest priority)
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_spatial_state()` | Activity = "fallen" or "lying_down" |
| 2 | `capture_photo(annotated=True)` | Immediate photo evidence |
| 3 | `send_telegram_alert("⚠️ FALL DETECTED! Person appears fallen at living room. Photo attached.", include_photo=True)` | Emergency alert to caregiver |
| 4 | `speak_to_user("Are you okay? I've alerted your family.", urgency="critical")` | Speak to the person |
| 5 | `save_observation("FALL at 2:30pm, position=(320,400)", tags=["fall", "emergency"])` | Critical log |
| 6 | `get_pose_landmarks(person_id)` | Body position analysis for paramedic info |

#### UC5.2 — Prolonged inactivity
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_spatial_state()` | Person sitting, speed = 0 |
| 2 | `get_time_in_activity(person_id, activity="sitting")` | 45 minutes no movement |
| 3 | `get_scene_changes()` | No changes in a long time |
| 4 | `speak_to_user("It's been a while since you moved. How about a short walk?")` | Gentle prompt |
| 5 | `save_observation("Inactive for 45 min, sitting", tags=["inactivity"])` | Track |
| 6 | `send_telegram_alert("Mom hasn't moved in 45 min. She's sitting in the living room.")` | Inform caregiver |

#### UC5.3 — Person left the room
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_scene_changes()` | Person left scene |
| 2 | `get_current_time()` | 2:00 AM (unusual hour) |
| 3 | `save_observation("Person left room at 2:00 AM", tags=["left_room", "night"])` | Log |
| 4 | `send_telegram_alert("Dad left the room at 2:00 AM — unusual for this hour")` | Alert |

#### UC5.4 — Unsteady movement
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_person_detail(person_id)` | Speed fluctuating, erratic direction changes |
| 2 | `get_pose_landmarks(person_id)` | Check balance (center of mass) |
| 3 | `analyze_posture(person_id)` | Body lean, instability indicators |
| 4 | `save_observation("Unsteady movement pattern detected", tags=["unsteady", "risk"])` | Early warning |
| 5 | `send_telegram_alert("Noticing some unsteady movement. Keeping close watch.")` | Heads up |

#### UC5.5 — Regular wellness check (caregiver gets update)
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_spatial_summary()` | Quick scene overview |
| 2 | `get_activity_timeline(person_id)` | What they've been doing |
| 3 | `get_session_stats()` | Overall session stats |
| 4 | `send_telegram_alert("All clear. Mom is up and walking around. Had breakfast at 8am.")` | Peace of mind |

**Tools used by this goal:** `get_spatial_state`, `get_spatial_summary`, `get_person_detail`, `get_scene_changes`, `get_pose_landmarks`, `analyze_posture`, `get_time_in_activity`, `get_activity_timeline`, `get_session_stats`, `get_current_time`, `capture_photo`, `send_telegram_alert`, `speak_to_user`, `save_observation`

---

## Goal 6: General Spatial Awareness (`general`)

**User says:** "Just tell me what's happening"

### Use Cases

#### UC6.1 — New person enters
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_scene_changes()` | New person detected |
| 2 | `get_person_detail(person_id)` | Activity, position, direction |
| 3 | `speak_to_user("Someone just walked in from the left. They're standing near the door.")` | Narrate |

#### UC6.2 — User asks "what do you see?"
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_spatial_state()` | Full scene |
| 2 | `get_objects_in_scene()` | All objects |
| 3 | `count_objects("person")` | People count |
| 4 | `speak_to_user("I see 2 people: one sitting at the desk, one standing by the window. There's a laptop, a cup, and a backpack on the desk.")` | Full description |

#### UC6.3 — Risk event
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_scene_changes()` | New risk event |
| 2 | `check_zone_status()` | Someone in danger zone |
| 3 | `send_telegram_alert("Person approaching monitored zone")` | Alert |

**Tools used by this goal:** `get_spatial_state`, `get_spatial_summary`, `get_scene_changes`, `get_person_detail`, `get_objects_in_scene`, `count_objects`, `check_zone_status`, `speak_to_user`, `send_telegram_alert`

---

## Custom Dynamic Goals (examples)

### "Count people walking through the hallway"
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_scene_changes()` | Detect new people entering/leaving |
| 2 | `get_person_detail(person_id)` | Confirm activity = walking |
| 3 | `save_observation("Person 5 walked through at 3:12pm", tags=["count", "hallway"])` | Count each one |
| 4 | `get_observations(tag="count")` | Running total |
| 5 | `send_telegram_alert("Hourly update: 12 people walked through in the last hour")` | Report |

### "Make sure my cat doesn't jump on the counter"
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_objects_in_scene(class_filter="cat")` | Look for cat |
| 2 | `set_watch_zone(x1, y1, x2, y2, label="Counter")` | Define counter area |
| 3 | `check_zone_status()` | Is cat near counter? |
| 4 | `speak_to_user("Your cat is on the counter again!", urgency="normal")` | Alert |
| 5 | `capture_photo()` | Catch them in the act |

### "Remind me to take a break every 20 minutes"
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `get_time_in_activity(person_id, activity="sitting")` | Check sitting duration |
| 2 | `get_current_time()` | Track time |
| 3 | `speak_to_user("20 minutes! Time to stand up and stretch.")` | Break reminder |
| 4 | `save_observation("Break reminder sent at 2:40pm", tags=["break"])` | Track breaks taken |

### "Watch for deliveries at the front door"
| Step | Tool Call | What Happens |
|------|-----------|-------------|
| 1 | `set_watch_zone(x1, y1, x2, y2, label="Front Door")` | Define door area |
| 2 | `get_scene_changes()` | New person near door |
| 3 | `get_person_detail(person_id)` | Check if they stop and leave (delivery behavior) |
| 4 | `get_objects_in_scene()` | Look for package/box |
| 5 | `capture_photo(annotated=True)` | Snapshot |
| 6 | `send_telegram_alert("Someone stopped at your door! Possible delivery. Photo attached.", include_photo=True)` | Alert with photo |

---

## Tool Usage Heatmap by Goal

| Tool | Desk | Posture | Driver | Study | Elderly | General |
|------|:----:|:-------:|:------:|:-----:|:-------:|:-------:|
| `get_spatial_state` | ● | ● | ● | | ● | ● |
| `get_spatial_summary` | | | | | ● | ● |
| `get_person_detail` | ● | | | | ● | ● |
| `get_scene_changes` | ● | | | ● | ● | ● |
| `get_objects_in_scene` | ● | | ● | ● | | ● |
| `count_objects` | | | | | | ● |
| `analyze_posture` | ● | ● | ● | | ● | |
| `get_pose_landmarks` | ● | ● | ● | | ● | |
| `check_body_alignment` | | ● | | | | |
| `get_activity_timeline` | | ● | | | ● | |
| `get_time_in_activity` | | ● | | ● | ● | |
| `get_session_stats` | | ● | | ● | ● | |
| `set_watch_zone` | ● | | | | | |
| `clear_watch_zones` | ● | | | | | |
| `check_zone_status` | ● | | | | | ● |
| `send_telegram_alert` | ● | ● | ● | ● | ● | ● |
| `speak_to_user` | ● | ● | ● | ● | ● | ● |
| `capture_photo` | ● | | ● | | ● | |
| `save_observation` | ● | ● | ● | ● | ● | |
| `get_observations` | | ● | ● | ● | | |
| `web_search` | | | | | | |
| `get_current_time` | | | | ● | ● | |
| `get_current_goal` | | | | | | |
| `update_goal` | | | | | | |
| `get_goal_presets` | | | | | | |

**Every goal uses:** `speak_to_user`, `send_telegram_alert`
**Perception-heavy:** Desk, Elderly, General
**Pose-heavy:** Posture, Driver
**Time-tracking-heavy:** Study, Elderly, Posture
**Zone-heavy:** Desk

---

## Tool Call Frequency (expected per minute)

| Goal | Checks/min | Alerts/min | Typical tools per check |
|------|-----------|-----------|------------------------|
| Desk Guardian | 4 | 0-1 | 2-3 (scene_changes + zone_status) |
| Posture Coach | 2-4 | 1-2 | 3-4 (posture + landmarks + alignment) |
| Driver Monitor | 6-10 | 0-2 | 2-3 (landmarks + posture) — HIGH frequency, safety-critical |
| Study Focus | 1-2 | 0-1 | 2-3 (scene_changes + time_in_activity) |
| Elderly Care | 2-4 | 0-1 | 2-4 (state + time + scene_changes) |
| General | 2 | 0-1 | 1-2 (scene_changes or summary) |
