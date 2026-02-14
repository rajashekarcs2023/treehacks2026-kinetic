"""
PREVENT: Real-Time Predictive Spatial Intelligence Engine
==========================================================
Main pipeline runner. Integrates all 6 layers:
  1. Perception (YOLO11n + ByteTrack + MediaPipe Pose)
  2. Motion Modeling (velocity + trajectory prediction)
  3. Risk Estimation (TTC + danger zone collision)
  4. Decision Engine (intervention triggers)
  5. Intervention Renderer (visual overlays + alerts)
  6. Dashboard (live web UI at http://localhost:5555)

Usage:
  python run.py                     # default: pose ON, depth OFF
  python run.py --depth             # enable depth estimation
  python run.py --no-pose           # disable pose overlay
  python run.py --no-dashboard      # disable web dashboard

Controls:
  q = quit
  z = draw new danger zone (click 2 corners)
  c = clear all danger zones
  d = toggle depth overlay
  p = toggle pose overlay
  r = reset intervention log
"""

import argparse
import sys
import time
import cv2
import numpy as np

from src.models import SceneState, DangerZone, BBox
from src.perception import PerceptionLayer
from src.motion import MotionModeler
from src.risk import RiskEstimator
from src.decision import DecisionEngine
from src.intervention import InterventionRenderer
from src.dashboard import DashboardBroadcaster, run_dashboard_server


def parse_args():
    parser = argparse.ArgumentParser(description="PREVENT Pipeline")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose estimation")
    parser.add_argument("--depth", action="store_true", help="Enable depth estimation")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable web dashboard")
    parser.add_argument("--port", type=int, default=5555, help="Dashboard port")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--prediction-horizon", type=float, default=2.0, help="Prediction horizon in seconds")
    parser.add_argument("--ttc-threshold", type=float, default=2.0, help="TTC warning threshold in seconds")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Initialize all layers ──────────────────────────────────────────

    print("Initializing PREVENT pipeline...")
    print(f"  Pose:       {'ON' if not args.no_pose else 'OFF'}")
    print(f"  Depth:      {'ON' if args.depth else 'OFF'}")
    print(f"  Dashboard:  {'ON (port {})'.format(args.port) if not args.no_dashboard else 'OFF'}")
    print(f"  Prediction: {args.prediction_horizon}s horizon")
    print(f"  TTC thresh: {args.ttc_threshold}s")
    print()

    # Layer 1: Perception
    perception = PerceptionLayer(
        enable_pose=not args.no_pose,
        enable_depth=args.depth,
    )

    # Layer 2: Motion
    motion = MotionModeler(
        prediction_horizon=args.prediction_horizon,
    )

    # Layer 3: Risk
    risk_estimator = RiskEstimator(
        ttc_threshold=args.ttc_threshold,
    )

    # Layer 4: Decision
    decision_engine = DecisionEngine(
        risk_threshold=0.3,
        cooldown_seconds=2.0,
    )

    # Layer 5: Intervention
    renderer = InterventionRenderer()

    # Layer 6: Dashboard
    broadcaster = None
    if not args.no_dashboard:
        broadcaster = DashboardBroadcaster()
        run_dashboard_server(broadcaster, port=args.port)

    # ── Open camera ────────────────────────────────────────────────────

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {width}x{height}")
    print("Pipeline running. Press 'q' to quit.")
    print()

    # ── Default danger zone (right third of frame) ─────────────────────

    danger_zones = [
        DangerZone(
            zone_id="zone_0",
            bbox=BBox(int(width * 0.65), int(height * 0.15), int(width * 0.95), int(height * 0.85)),
            label="DANGER ZONE",
        ),
    ]
    zone_counter = 1

    # ── State for interactive zone drawing ─────────────────────────────

    drawing_zone = False
    zone_pts = []
    show_depth_overlay = False
    show_pose = not args.no_pose

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing_zone, zone_pts, danger_zones, zone_counter
        if drawing_zone and event == cv2.EVENT_LBUTTONDOWN:
            zone_pts.append((x, y))
            if len(zone_pts) == 2:
                x1 = min(zone_pts[0][0], zone_pts[1][0])
                y1 = min(zone_pts[0][1], zone_pts[1][1])
                x2 = max(zone_pts[0][0], zone_pts[1][0])
                y2 = max(zone_pts[0][1], zone_pts[1][1])
                danger_zones.append(DangerZone(
                    zone_id=f"zone_{zone_counter}",
                    bbox=BBox(x1, y1, x2, y2),
                    label=f"ZONE {zone_counter}",
                ))
                print(f"Added danger zone_{zone_counter}: ({x1},{y1})-({x2},{y2})")
                zone_counter += 1
                zone_pts = []
                drawing_zone = False

    cv2.namedWindow("PREVENT")
    cv2.setMouseCallback("PREVENT", mouse_callback)

    # ── Main loop ──────────────────────────────────────────────────────

    prev_time = time.time()
    fps = 0
    frame_count = 0
    depth_map = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - prev_time
        prev_time = now
        instant_fps = 1.0 / dt if dt > 0 else 0
        fps = 0.9 * fps + 0.1 * instant_fps
        frame_count += 1

        # ── Layer 1: Perception ────────────────────────────────────────
        persons, pose_landmarks, new_depth = perception.process(frame, now)

        if new_depth is not None:
            depth_map = new_depth

        # ── Layer 2: Motion ────────────────────────────────────────────
        persons = motion.update(persons)

        # ── Layer 3: Risk ──────────────────────────────────────────────
        risk_events = risk_estimator.assess(persons, danger_zones)

        # ── Layer 4: Decision ──────────────────────────────────────────
        interventions = decision_engine.evaluate(risk_events)

        # ── Build scene state ──────────────────────────────────────────
        scene = SceneState(
            frame_number=frame_count,
            timestamp=now,
            persons=persons,
            danger_zones=danger_zones,
            risk_events=risk_events,
            interventions=interventions,
            fps=fps,
            frame_width=width,
            frame_height=height,
        )

        # ── Layer 5: Intervention (render) ─────────────────────────────
        display_frame = frame.copy()

        # Optional depth overlay
        if show_depth_overlay and depth_map is not None:
            depth_color = cv2.applyColorMap(
                (depth_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
            )
            display_frame = cv2.addWeighted(display_frame, 0.6, depth_color, 0.4, 0)

        trails = motion.get_all_trails()
        display_pose = pose_landmarks if show_pose else []
        display_frame = renderer.render(display_frame, scene, trails, display_pose,
                                        decision_engine.get_stats())

        if drawing_zone:
            cv2.putText(display_frame, "CLICK 2 CORNERS for danger zone",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ── Layer 6: Dashboard ─────────────────────────────────────────
        if broadcaster:
            broadcaster.update(scene, display_frame)

        # ── Display ────────────────────────────────────────────────────
        cv2.imshow("PREVENT", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z'):
            drawing_zone = True
            zone_pts = []
            print("Zone drawing mode: click two corners.")
        elif key == ord('c'):
            danger_zones = []
            print("Cleared all danger zones.")
        elif key == ord('d'):
            show_depth_overlay = not show_depth_overlay
            print(f"Depth overlay: {'ON' if show_depth_overlay else 'OFF'}")
        elif key == ord('p'):
            show_pose = not show_pose
            print(f"Pose overlay: {'ON' if show_pose else 'OFF'}")
        elif key == ord('r'):
            decision_engine.intervention_log.clear()
            print("Intervention log reset.")

    # ── Cleanup ────────────────────────────────────────────────────────

    cap.release()
    cv2.destroyAllWindows()
    perception.close()

    # Print final stats
    stats = decision_engine.get_stats()
    print("\n" + "=" * 60)
    print("PREVENT SESSION SUMMARY")
    print("=" * 60)
    print(f"  Total frames:       {frame_count}")
    print(f"  Total interventions: {stats['total']}")
    print(f"    Critical:          {stats['critical']}")
    print(f"    Urgent:            {stats['urgent']}")
    print(f"    Early warning:     {stats['early']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
