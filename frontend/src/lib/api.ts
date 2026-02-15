const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

// ── Scene & Perception ─────────────────────────────────────────────
export async function getState() {
  return fetchAPI<SpatialState>("/api/state");
}

export async function getSummary() {
  return fetchAPI<{ summary: string }>("/api/summary");
}

// ── Activity ───────────────────────────────────────────────────────
export async function getActivityStats() {
  return fetchAPI<ActivityStats>("/api/activity/stats");
}

export async function trainActivityModel(nPerClass = 150, epochs = 60) {
  return fetchAPI<TrainResult>(
    `/api/activity/train?n_per_class=${nPerClass}&epochs=${epochs}`,
    { method: "POST" }
  );
}

// ── Coaching ───────────────────────────────────────────────────────
export async function startCoaching(skill: string, reference?: string) {
  return fetchAPI<CoachingStatus>(
    "/api/coaching/start",
    {
      method: "POST",
      body: JSON.stringify({
        skill_name: skill,
        reference_name: reference || null,
      }),
    }
  );
}

export async function ingestYouTube(url: string, name: string, keyAngle = "left_knee") {
  return fetchAPI<{
    status: string;
    name?: string;
    frames?: number;
    phases?: number;
    duration?: number;
    error?: string;
  }>("/api/references/from_youtube", {
    method: "POST",
    body: JSON.stringify({ url, name, key_angle: keyAngle }),
  });
}

export async function stopCoaching() {
  return fetchAPI<CoachingSummary>("/api/coaching/stop", { method: "POST" });
}

export async function getCoachingStatus() {
  return fetchAPI<CoachingStatus>("/api/coaching/status");
}

export async function getCoachingProgress() {
  return fetchAPI<CoachingProgress>("/api/coaching/progress");
}

export async function getCoachingScore() {
  return fetchAPI<{ score: number; details: Record<string, number> }>("/api/coaching/score");
}

export async function getCoachingQuality() {
  return fetchAPI<MovementQuality>("/api/coaching/quality");
}

export async function getCoachingAngles() {
  return fetchAPI<{ angles: Record<string, number> }>("/api/coaching/angles");
}

// ── References ─────────────────────────────────────────────────────
export async function getReferences() {
  return fetchAPI<{ references: string[] }>("/api/references");
}

export async function getReference(name: string) {
  return fetchAPI<ReferenceDetail>(`/api/references/${encodeURIComponent(name)}`);
}

// ── Skill Graphs ───────────────────────────────────────────────────
export async function getGraphs() {
  return fetchAPI<{ graphs: string[] }>("/api/graphs");
}

export async function getGraph(name: string) {
  return fetchAPI<SkillGraph>(`/api/graphs/${encodeURIComponent(name)}`);
}

export async function getRecommendations(graphName: string) {
  return fetchAPI<{ recommendations: Recommendation[] }>(
    `/api/graphs/${encodeURIComponent(graphName)}/recommend`
  );
}

export async function getGraphProgress(graphName: string) {
  return fetchAPI<GraphProgress>(
    `/api/graphs/${encodeURIComponent(graphName)}/progress`
  );
}

export async function updateSkillProficiency(
  graphName: string,
  skillId: string,
  score: number
) {
  return fetchAPI<{ proficiency: number }>(
    `/api/graphs/${encodeURIComponent(graphName)}/skills/${encodeURIComponent(skillId)}/update?score=${score}`,
    { method: "POST" }
  );
}

// ── Goals ──────────────────────────────────────────────────────────
export async function getGoals() {
  return fetchAPI<{ goals: Goal[] }>("/api/goals");
}

export async function setGoal(goalId: string) {
  return fetchAPI<Goal>(`/api/goals/${encodeURIComponent(goalId)}`, {
    method: "POST",
  });
}

// ── Training & Model ───────────────────────────────────────────────
export async function getTrainingStats() {
  return fetchAPI<TrainingStats>("/api/training/stats");
}

export async function bootstrapModel(nPerSkill = 30, epochs = 50) {
  return fetchAPI<BootstrapResult>(
    `/api/model/bootstrap?n_per_skill=${nPerSkill}&epochs=${epochs}`,
    { method: "POST" }
  );
}

export async function getModelStatus() {
  return fetchAPI<ModelStatus>("/api/model/status");
}

// ── Agent ──────────────────────────────────────────────────────────
export async function getAgentStatus() {
  return fetchAPI<AgentStatus>("/api/agent/status");
}

export async function sendMessage(message: string) {
  return fetchAPI<{ response: string }>("/api/agent/message", {
    method: "POST",
    body: JSON.stringify({ message }),
  });
}

// ── Room APIs (Multiplayer) ───────────────────────────────────────
export async function createRoom(skillName: string, displayName: string) {
  return fetchAPI<{ room_code: string; skill: string; user_id: string; participants: number }>("/api/rooms/create", {
    method: "POST",
    body: JSON.stringify({ skill_name: skillName, display_name: displayName }),
  });
}

export async function joinRoom(roomCode: string, displayName: string) {
  return fetchAPI<{ room_code: string; skill: string; user_id: string; participants: number; leaderboard: unknown[] }>("/api/rooms/join", {
    method: "POST",
    body: JSON.stringify({ room_code: roomCode, display_name: displayName }),
  });
}

export async function getRoomStatus(code: string) {
  return fetchAPI<{ room_code: string; skill: string; participant_count: number; leaderboard: unknown[] }>(`/api/rooms/${code}`);
}

export async function getRoomLeaderboard(code: string) {
  return fetchAPI<{ leaderboard: unknown[] }>(`/api/rooms/${code}/leaderboard`);
}

export async function compareRoom(code: string) {
  return fetchAPI<{ leaderboard: unknown[]; agent_analysis: string }>(`/api/rooms/${code}/compare`, { method: "POST" });
}

export async function closeRoom(code: string) {
  return fetchAPI<{ leaderboard: unknown[]; agent_verdict?: string }>(`/api/rooms/${code}/close`, { method: "POST" });
}

// ── AI Expert Generation ─────────────────────────────────────────
export interface AIExpertResult {
  skill: string;
  source: string;
  frame_count: number;
  phases: string[];
  coaching_cues: string[];
  keyframes: number[][][]; // [frame][joint][x,y,vis]
  generation_ms: number;
  model: string;
  error?: string;
}

export async function generateAIExpert(skillDescription: string, useDgx = false): Promise<AIExpertResult> {
  return fetchAPI<AIExpertResult>("/api/ai-expert/generate", {
    method: "POST",
    body: JSON.stringify({ skill_description: skillDescription, use_dgx: useDgx }),
  });
}

// ── Monitoring / Goals ───────────────────────────────────────────
export async function sendAlertToTelegram(message: string) {
  return fetchAPI<{ status: string }>("/api/alert", {
    method: "POST",
    body: JSON.stringify({ message }),
  });
}

// ── WebSocket helpers ──────────────────────────────────────────────
export function createCoachingWS(): WebSocket {
  const wsBase = API_BASE.replace(/^http/, "ws");
  return new WebSocket(`${wsBase}/ws/coaching`);
}

export function createVideoWS(): WebSocket {
  const wsBase = API_BASE.replace(/^http/, "ws");
  return new WebSocket(`${wsBase}/ws/video`);
}

export function createAudioWS(): WebSocket {
  const wsBase = API_BASE.replace(/^http/, "ws");
  return new WebSocket(`${wsBase}/ws/audio`);
}

// ── Types ──────────────────────────────────────────────────────────
export interface SpatialState {
  persons: PersonState[];
  objects: ObjectState[];
  timestamp: number;
  fps: number;
}

export interface PersonState {
  track_id: number;
  bbox: [number, number, number, number];
  activity: string;
  confidence: number;
  speed: number;
  pose_detected: boolean;
}

export interface ObjectState {
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
}

export interface ActivityStats {
  ml_available: boolean;
  ml_accuracy: number | null;
  classes: string[];
  total_predictions: number;
  heuristic_fallbacks: number;
}

export interface TrainResult {
  accuracy: number;
  epochs: number;
  classes: number;
}

export interface CoachingStatus {
  active: boolean;
  skill: string;
  reps: number;
  current_score: number;
  phase: string;
  elapsed: number;
}

export interface CoachingSummary {
  skill: string;
  total_reps: number;
  avg_score: number;
  best_score: number;
  duration: number;
  improvements: string[];
}

export interface CoachingProgress {
  reps: number;
  scores: number[];
  avg_score: number;
  best_score: number;
  trend: "improving" | "stable" | "declining";
}

export interface MovementQuality {
  smoothness: number;
  symmetry: number;
  range_of_motion: number;
  tempo_consistency: number;
  overall: number;
}

export interface ReferenceDetail {
  name: string;
  skill: string;
  frames: number;
  duration: number;
}

export interface SkillGraph {
  name: string;
  skills: SkillNode[];
}

export interface SkillNode {
  id: string;
  name: string;
  category: string;
  proficiency: number;
  unlocked: boolean;
  prerequisites: string[];
  description: string;
}

export interface Recommendation {
  skill_id: string;
  skill_name: string;
  reason: string;
  priority: number;
}

export interface GraphProgress {
  total_skills: number;
  unlocked: number;
  mastered: number;
  avg_proficiency: number;
  categories: Record<string, { count: number; avg: number }>;
}

export interface Goal {
  id: string;
  name: string;
  description: string;
  category: string;
}

export interface TrainingStats {
  skills: Record<string, number>;
  total_samples: number;
}

export interface BootstrapResult {
  samples_generated: number;
  model_trained: boolean;
  rmse: number;
}

export interface ModelStatus {
  trained: boolean;
  skill_count: number;
  sample_count: number;
}

export interface AgentStatus {
  goal: string;
  uptime: number;
  tool_calls: number;
  decisions: number;
}

