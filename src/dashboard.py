"""
Layer 6: Dashboard & Explainability
=====================================
WebSocket-based live dashboard that streams scene state to a browser UI.
Runs a lightweight Flask server with SocketIO.

The dashboard shows:
- Live video feed with overlays
- Active tracked persons table
- Risk events timeline
- Intervention log
- FPS and system stats
"""

import json
import time
import threading
import base64
import cv2
import numpy as np
from src.models import SceneState, RiskEvent, Intervention


class DashboardBroadcaster:
    """
    Collects scene state each frame and broadcasts to connected dashboard clients.
    Uses a simple shared state approach — the main loop writes, the server reads.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_state: dict | None = None
        self._latest_frame_jpg: bytes | None = None
        self._event_history: list[dict] = []
        self._max_history = 200

    def update(self, scene: SceneState, frame: np.ndarray):
        """Called each frame from the main pipeline loop."""
        # Encode frame as JPEG
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        jpg_bytes = jpg.tobytes()

        # Serialize scene state
        state = {
            "frame_number": scene.frame_number,
            "timestamp": scene.timestamp,
            "fps": round(scene.fps, 1),
            "num_persons": len(scene.persons),
            "persons": [
                {
                    "id": p.track_id,
                    "cx": p.bbox.cx,
                    "cy": p.bbox.cy,
                    "speed": round(p.speed, 1),
                    "direction": round(p.direction_deg, 1),
                    "confidence": round(p.confidence, 2),
                    "depth": round(p.depth_estimate, 3) if p.depth_estimate else None,
                }
                for p in scene.persons
            ],
            "risk_events": [
                {
                    "event_id": e.event_id,
                    "person_id": e.person_id,
                    "zone_id": e.zone_id,
                    "ttc": round(e.ttc, 2),
                    "risk_score": round(e.risk_score, 2),
                    "description": e.description,
                }
                for e in scene.risk_events
            ],
            "interventions": [
                {
                    "id": i.intervention_id,
                    "action_type": i.action_type,
                    "person_id": i.risk_event.person_id,
                    "ttc": round(i.risk_event.ttc, 2),
                    "risk_score": round(i.risk_event.risk_score, 2),
                }
                for i in scene.interventions
            ],
            "num_zones": len(scene.danger_zones),
        }

        with self._lock:
            self._latest_state = state
            self._latest_frame_jpg = jpg_bytes

            # Append to event history
            for evt in state["risk_events"]:
                evt["frame"] = scene.frame_number
                self._event_history.append(evt)
            self._event_history = self._event_history[-self._max_history:]

    def get_latest(self) -> tuple[dict | None, bytes | None]:
        """Get latest state and frame (thread-safe)."""
        with self._lock:
            return self._latest_state, self._latest_frame_jpg

    def get_event_history(self) -> list[dict]:
        with self._lock:
            return list(self._event_history)


def create_dashboard_app(broadcaster: DashboardBroadcaster):
    """Create a Flask app that serves the dashboard."""
    from flask import Flask, Response, render_template_string, jsonify

    app = Flask(__name__)

    DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PREVENT Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0a; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; }
  .header { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 16px 24px; display: flex; align-items: center; gap: 16px; border-bottom: 2px solid #0f3460; }
  .header h1 { font-size: 22px; font-weight: 700; color: #fff; }
  .header .badge { background: #00c853; color: #000; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .container { display: grid; grid-template-columns: 1fr 380px; gap: 16px; padding: 16px; height: calc(100vh - 72px); }
  .video-panel { background: #111; border-radius: 12px; overflow: hidden; position: relative; }
  .video-panel img { width: 100%; height: 100%; object-fit: contain; }
  .side-panel { display: flex; flex-direction: column; gap: 12px; overflow-y: auto; }
  .card { background: #1a1a1a; border-radius: 10px; padding: 16px; border: 1px solid #2a2a2a; }
  .card h3 { font-size: 13px; text-transform: uppercase; color: #888; letter-spacing: 1px; margin-bottom: 10px; }
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat { background: #222; border-radius: 8px; padding: 12px; text-align: center; }
  .stat .value { font-size: 28px; font-weight: 700; color: #fff; }
  .stat .label { font-size: 11px; color: #888; margin-top: 2px; }
  .stat.alert .value { color: #ff4444; }
  .stat.ok .value { color: #00c853; }
  .person-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2a2a2a; font-size: 13px; }
  .person-row:last-child { border: none; }
  .risk-item { padding: 8px 10px; margin-bottom: 6px; border-radius: 6px; font-size: 12px; }
  .risk-item.high { background: rgba(255, 68, 68, 0.15); border-left: 3px solid #ff4444; }
  .risk-item.medium { background: rgba(255, 165, 0, 0.15); border-left: 3px solid orange; }
  .risk-item.low { background: rgba(0, 200, 83, 0.15); border-left: 3px solid #00c853; }
  .log-entry { font-size: 11px; color: #aaa; padding: 3px 0; border-bottom: 1px solid #1a1a1a; font-family: monospace; }
  #no-signal { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); color: #555; font-size: 18px; }
</style>
</head>
<body>
<div class="header">
  <h1>PREVENT</h1>
  <span class="badge" id="status-badge">CONNECTING...</span>
  <span style="color:#666; font-size:13px; margin-left:auto;">Real-Time Predictive Spatial Intelligence</span>
</div>
<div class="container">
  <div class="video-panel">
    <img id="video-feed" src="" alt="Live Feed">
    <div id="no-signal">Waiting for video stream...</div>
  </div>
  <div class="side-panel">
    <div class="card">
      <h3>System Stats</h3>
      <div class="stat-grid">
        <div class="stat ok"><div class="value" id="fps">--</div><div class="label">FPS</div></div>
        <div class="stat"><div class="value" id="persons">0</div><div class="label">Persons</div></div>
        <div class="stat alert"><div class="value" id="risks">0</div><div class="label">Active Risks</div></div>
        <div class="stat"><div class="value" id="interventions">0</div><div class="label">Interventions</div></div>
      </div>
    </div>
    <div class="card">
      <h3>Tracked Persons</h3>
      <div id="persons-list"><span style="color:#555">No persons detected</span></div>
    </div>
    <div class="card" style="flex:1; overflow-y:auto;">
      <h3>Risk Events</h3>
      <div id="risk-list"><span style="color:#555">No active risks</span></div>
    </div>
    <div class="card" style="max-height: 180px; overflow-y: auto;">
      <h3>Intervention Log</h3>
      <div id="log-list"><span style="color:#555">No interventions yet</span></div>
    </div>
  </div>
</div>
<script>
let totalInterventions = 0;
let logEntries = [];

function updateDashboard() {
  fetch('/api/state')
    .then(r => r.json())
    .then(data => {
      if (!data) return;
      document.getElementById('status-badge').textContent = 'LIVE';
      document.getElementById('status-badge').style.background = '#00c853';
      document.getElementById('fps').textContent = data.fps || '--';
      document.getElementById('persons').textContent = data.num_persons || 0;
      document.getElementById('risks').textContent = (data.risk_events || []).length;

      // Persons list
      const pl = document.getElementById('persons-list');
      if (data.persons && data.persons.length > 0) {
        pl.innerHTML = data.persons.map(p =>
          `<div class="person-row">
            <span>ID: ${p.id}</span>
            <span>${p.speed} px/s</span>
            <span>conf: ${p.confidence}</span>
          </div>`
        ).join('');
      } else {
        pl.innerHTML = '<span style="color:#555">No persons detected</span>';
      }

      // Risk events
      const rl = document.getElementById('risk-list');
      if (data.risk_events && data.risk_events.length > 0) {
        rl.innerHTML = data.risk_events.map(e => {
          const cls = e.risk_score > 0.7 ? 'high' : e.risk_score > 0.3 ? 'medium' : 'low';
          return `<div class="risk-item ${cls}">
            <strong>Person ${e.person_id}</strong> → ${e.zone_id}<br>
            TTC: ${e.ttc}s | Risk: ${(e.risk_score*100).toFixed(0)}%
          </div>`;
        }).join('');
      } else {
        rl.innerHTML = '<span style="color:#555">No active risks</span>';
      }

      // Interventions
      if (data.interventions && data.interventions.length > 0) {
        totalInterventions += data.interventions.length;
        data.interventions.forEach(i => {
          logEntries.unshift(`[${new Date().toLocaleTimeString()}] ${i.action_type} → Person ${i.person_id} (TTC: ${i.ttc}s)`);
        });
        logEntries = logEntries.slice(0, 50);
      }
      document.getElementById('interventions').textContent = totalInterventions;
      const ll = document.getElementById('log-list');
      if (logEntries.length > 0) {
        ll.innerHTML = logEntries.map(e => `<div class="log-entry">${e}</div>`).join('');
      }
    })
    .catch(() => {
      document.getElementById('status-badge').textContent = 'DISCONNECTED';
      document.getElementById('status-badge').style.background = '#ff4444';
    });
}

function updateVideo() {
  const img = document.getElementById('video-feed');
  img.src = '/api/frame?' + Date.now();
  img.onload = () => {
    document.getElementById('no-signal').style.display = 'none';
  };
}

setInterval(updateDashboard, 200);
setInterval(updateVideo, 100);
</script>
</body>
</html>
"""

    @app.route('/')
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route('/api/state')
    def api_state():
        state, _ = broadcaster.get_latest()
        if state is None:
            return jsonify({})
        return jsonify(state)

    @app.route('/api/frame')
    def api_frame():
        _, jpg = broadcaster.get_latest()
        if jpg is None:
            # Return 1x1 transparent pixel
            return Response(b'', mimetype='image/jpeg')
        return Response(jpg, mimetype='image/jpeg')

    @app.route('/api/history')
    def api_history():
        return jsonify(broadcaster.get_event_history())

    return app


def run_dashboard_server(broadcaster: DashboardBroadcaster, port: int = 5555):
    """Run the dashboard server in a background thread."""
    app = create_dashboard_app(broadcaster)

    def _run():
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    print(f"Dashboard running at http://localhost:{port}")
    return thread
