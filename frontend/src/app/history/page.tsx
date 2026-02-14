"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScoreRing } from "@/components/score-ring";
import {
  Calendar,
  TrendingUp,
  TrendingDown,
  Minus,
  Clock,
  Dumbbell,
  Filter,
  ChevronDown,
} from "lucide-react";

interface Session {
  id: string;
  date: string;
  time: string;
  skill: string;
  score: number;
  reps: number;
  duration: string;
  trend: "up" | "down" | "same";
  improvements: string[];
  quality: {
    smoothness: number;
    symmetry: number;
    rangeOfMotion: number;
    tempo: number;
  };
}

const MOCK_HISTORY: Session[] = [
  {
    id: "1", date: "Today", time: "10:30 AM", skill: "Tennis Serve", score: 74, reps: 8,
    duration: "5:30", trend: "up",
    improvements: ["Shoulder rotation improved 20%", "Ball toss height more consistent"],
    quality: { smoothness: 72, symmetry: 70, rangeOfMotion: 78, tempo: 80 },
  },
  {
    id: "2", date: "Today", time: "9:15 AM", skill: "Salsa Basic Step", score: 88, reps: 12,
    duration: "7:00", trend: "up",
    improvements: ["Hip movement more fluid", "Timing 88% on beat"],
    quality: { smoothness: 90, symmetry: 85, rangeOfMotion: 82, tempo: 92 },
  },
  {
    id: "3", date: "Today", time: "8:30 AM", skill: "Squat", score: 82, reps: 15,
    duration: "4:30", trend: "up",
    improvements: ["Better depth consistency", "Knee tracking improved"],
    quality: { smoothness: 78, symmetry: 85, rangeOfMotion: 80, tempo: 90 },
  },
  {
    id: "4", date: "Yesterday", time: "7:45 PM", skill: "Warrior Pose", score: 91, reps: 6,
    duration: "6:15", trend: "same",
    improvements: ["Excellent balance", "Arm alignment perfect"],
    quality: { smoothness: 92, symmetry: 88, rangeOfMotion: 85, tempo: 95 },
  },
  {
    id: "5", date: "Yesterday", time: "6:30 PM", skill: "Boxing Jab", score: 80, reps: 20,
    duration: "4:00", trend: "up",
    improvements: ["Faster retraction", "Better weight transfer"],
    quality: { smoothness: 75, symmetry: 82, rangeOfMotion: 70, tempo: 88 },
  },
  {
    id: "6", date: "Yesterday", time: "5:00 PM", skill: "Knee Extension", score: 85, reps: 10,
    duration: "3:00", trend: "up",
    improvements: ["ROM improved to 95%", "No compensation detected"],
    quality: { smoothness: 88, symmetry: 90, rangeOfMotion: 95, tempo: 85 },
  },
  {
    id: "7", date: "Feb 12", time: "8:00 AM", skill: "Guitar Posture", score: 62, reps: 5,
    duration: "8:00", trend: "up",
    improvements: ["Wrist angle improved", "Neck less strained"],
    quality: { smoothness: 60, symmetry: 65, rangeOfMotion: 55, tempo: 70 },
  },
  {
    id: "8", date: "Feb 12", time: "7:00 AM", skill: "ASL Alphabet", score: 80, reps: 26,
    duration: "5:30", trend: "up",
    improvements: ["Cleaner hand shapes for D, G, P", "Speed increased"],
    quality: { smoothness: 78, symmetry: 82, rangeOfMotion: 75, tempo: 85 },
  },
  {
    id: "9", date: "Feb 11", time: "6:00 PM", skill: "Front Kick", score: 72, reps: 15,
    duration: "4:00", trend: "same",
    improvements: ["Better chamber position", "Work on hip extension"],
    quality: { smoothness: 68, symmetry: 72, rangeOfMotion: 75, tempo: 70 },
  },
];

const WEEKLY_DATA = [
  { day: "Mon", score: 72, sessions: 2 },
  { day: "Tue", score: 78, sessions: 3 },
  { day: "Wed", score: 75, sessions: 2 },
  { day: "Thu", score: 82, sessions: 3 },
  { day: "Fri", score: 80, sessions: 2 },
  { day: "Sat", score: 85, sessions: 4 },
  { day: "Sun", score: 87, sessions: 3 },
];

export default function HistoryPage() {
  const [expandedSession, setExpandedSession] = useState<string | null>(null);
  const [filter, setFilter] = useState("all");

  const filteredHistory =
    filter === "all"
      ? MOCK_HISTORY
      : MOCK_HISTORY.filter((s) => s.skill.toLowerCase() === filter);

  const uniqueSkills = [...new Set(MOCK_HISTORY.map((s) => s.skill))];

  const totalReps = MOCK_HISTORY.reduce((a, s) => a + s.reps, 0);
  const avgScore = Math.round(
    MOCK_HISTORY.reduce((a, s) => a + s.score, 0) / MOCK_HISTORY.length
  );
  const totalSessions = MOCK_HISTORY.length;
  const bestScore = Math.max(...MOCK_HISTORY.map((s) => s.score));

  return (
    <div className="p-6 lg:p-8 space-y-6 max-w-7xl mx-auto">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">History</h1>
        <p className="text-muted-foreground mt-1">
          Review your sessions, track improvement, celebrate progress.
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-card border-border">
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold">{totalSessions}</p>
            <p className="text-xs text-muted-foreground">Total Sessions</p>
          </CardContent>
        </Card>
        <Card className="bg-card border-border">
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold">{totalReps}</p>
            <p className="text-xs text-muted-foreground">Total Reps</p>
          </CardContent>
        </Card>
        <Card className="bg-card border-border">
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold">{avgScore}</p>
            <p className="text-xs text-muted-foreground">Avg Score</p>
          </CardContent>
        </Card>
        <Card className="bg-card border-border">
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold text-primary">{bestScore}</p>
            <p className="text-xs text-muted-foreground">Personal Best</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Left: Weekly chart */}
        <div className="space-y-6">
          {/* Weekly Overview */}
          <Card className="bg-card border-border">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Calendar className="h-4 w-4" />
                This Week
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-end gap-2 h-32">
                {WEEKLY_DATA.map((d, i) => (
                  <div key={d.day} className="flex-1 flex flex-col items-center gap-1">
                    <span className="text-[10px] text-muted-foreground tabular-nums">
                      {d.score}
                    </span>
                    <div className="w-full rounded-t-sm bg-secondary relative" style={{ height: "100%" }}>
                      <div
                        className="absolute bottom-0 left-0 right-0 rounded-t-sm transition-all duration-500"
                        style={{
                          height: `${(d.score / 100) * 100}%`,
                          backgroundColor:
                            i === WEEKLY_DATA.length - 1 ? "#06b6d4" : "#06b6d4",
                          opacity: 0.3 + (i / WEEKLY_DATA.length) * 0.7,
                        }}
                      />
                    </div>
                    <span className="text-[10px] text-muted-foreground">{d.day}</span>
                  </div>
                ))}
              </div>
              <div className="mt-4 flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Weekly Trend</span>
                <span className="flex items-center gap-1 text-green-500">
                  <TrendingUp className="h-3 w-3" /> +8% from last week
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Per-Skill Averages */}
          <Card className="bg-card border-border">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Skill Averages
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {uniqueSkills.map((skill) => {
                const skillSessions = MOCK_HISTORY.filter((s) => s.skill === skill);
                const avg = Math.round(
                  skillSessions.reduce((a, s) => a + s.score, 0) / skillSessions.length
                );
                return (
                  <div key={skill} className="flex items-center gap-3">
                    <ScoreRing score={avg} size={40} strokeWidth={3} />
                    <div className="flex-1">
                      <p className="text-sm font-medium">{skill}</p>
                      <p className="text-[10px] text-muted-foreground">
                        {skillSessions.length} sessions
                      </p>
                    </div>
                    <span className="text-sm font-medium tabular-nums">{avg}%</span>
                  </div>
                );
              })}
            </CardContent>
          </Card>
        </div>

        {/* Right: Session List */}
        <div className="lg:col-span-2 space-y-4">
          {/* Filter */}
          <div className="flex items-center gap-2 flex-wrap">
            <Button
              variant={filter === "all" ? "default" : "outline"}
              size="sm"
              className="text-xs"
              onClick={() => setFilter("all")}
            >
              All
            </Button>
            {uniqueSkills.map((skill) => (
              <Button
                key={skill}
                variant={filter === skill.toLowerCase() ? "default" : "outline"}
                size="sm"
                className="text-xs"
                onClick={() => setFilter(skill.toLowerCase())}
              >
                {skill}
              </Button>
            ))}
          </div>

          {/* Session List */}
          <div className="space-y-3">
            {filteredHistory.map((session) => {
              const isExpanded = expandedSession === session.id;
              return (
                <Card
                  key={session.id}
                  className="bg-card border-border transition-all hover:border-primary/20 cursor-pointer"
                  onClick={() => setExpandedSession(isExpanded ? null : session.id)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center gap-4">
                      <ScoreRing score={session.score} size={56} strokeWidth={4} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="font-medium">{session.skill}</p>
                          <Badge
                            variant="outline"
                            className={
                              session.score >= 80
                                ? "border-primary/40 text-primary text-[10px]"
                                : session.score >= 60
                                ? "border-green-500/40 text-green-500 text-[10px]"
                                : "border-yellow-500/40 text-yellow-500 text-[10px]"
                            }
                          >
                            {session.score >= 80
                              ? "Excellent"
                              : session.score >= 60
                              ? "Good"
                              : "Needs Work"}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground mt-1">
                          <span className="flex items-center gap-1">
                            <Dumbbell className="h-3 w-3" /> {session.reps} reps
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" /> {session.duration}
                          </span>
                          <span>
                            {session.date} · {session.time}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {session.trend === "up" && (
                          <TrendingUp className="h-4 w-4 text-green-500" />
                        )}
                        {session.trend === "down" && (
                          <TrendingDown className="h-4 w-4 text-red-500" />
                        )}
                        {session.trend === "same" && (
                          <Minus className="h-4 w-4 text-muted-foreground" />
                        )}
                        <ChevronDown
                          className={`h-4 w-4 text-muted-foreground transition-transform ${
                            isExpanded ? "rotate-180" : ""
                          }`}
                        />
                      </div>
                    </div>

                    {/* Expanded Details */}
                    {isExpanded && (
                      <div className="mt-4 pt-4 border-t border-border space-y-4">
                        {/* Quality Metrics */}
                        <div className="grid grid-cols-4 gap-3">
                          {[
                            { label: "Smoothness", value: session.quality.smoothness },
                            { label: "Symmetry", value: session.quality.symmetry },
                            { label: "ROM", value: session.quality.rangeOfMotion },
                            { label: "Tempo", value: session.quality.tempo },
                          ].map((m) => (
                            <div key={m.label} className="text-center">
                              <p className="text-lg font-semibold tabular-nums">{m.value}%</p>
                              <p className="text-[10px] text-muted-foreground">{m.label}</p>
                            </div>
                          ))}
                        </div>

                        {/* Improvements */}
                        {session.improvements.length > 0 && (
                          <div>
                            <p className="text-xs text-muted-foreground mb-1.5">Notes</p>
                            <div className="space-y-1">
                              {session.improvements.map((imp, i) => (
                                <p key={i} className="text-xs text-foreground/80">
                                  • {imp}
                                </p>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// filters
// weekly chart
