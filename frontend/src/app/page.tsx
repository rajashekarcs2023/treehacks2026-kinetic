"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScoreRing } from "@/components/score-ring";
import {
  Video,
  Flame,
  TrendingUp,
  Clock,
  ChevronRight,
  Brain,
  Mic,
  Upload,
  FileText,
  Sparkles,
  Dumbbell,
  Music,
  Swords,
  HeartPulse,
  Hand,
  Monitor,
} from "lucide-react";

interface SessionSummary {
  id: string;
  skill: string;
  category: string;
  score: number;
  reps: number;
  duration: string;
  time: string;
  trend: "up" | "down" | "same";
}

const MOCK_SESSIONS: SessionSummary[] = [
  { id: "1", skill: "Tennis Serve", category: "Sports", score: 74, reps: 8, duration: "5:30", time: "1h ago", trend: "up" },
  { id: "2", skill: "Salsa Basic Step", category: "Dance", score: 88, reps: 12, duration: "7:00", time: "3h ago", trend: "up" },
  { id: "3", skill: "Squat", category: "Fitness", score: 82, reps: 15, duration: "4:30", time: "5h ago", trend: "up" },
  { id: "4", skill: "Warrior Pose", category: "Yoga", score: 91, reps: 6, duration: "6:15", time: "Yesterday", trend: "same" },
  { id: "5", skill: "Knee Extension", category: "PT Rehab", score: 85, reps: 10, duration: "3:00", time: "Yesterday", trend: "up" },
];

const SKILL_CATEGORIES = [
  { name: "Fitness", icon: Dumbbell, color: "text-blue-400", bg: "bg-blue-500/10", examples: "Squats, Deadlifts, Push-ups" },
  { name: "Dance", icon: Music, color: "text-pink-400", bg: "bg-pink-500/10", examples: "Salsa, Hip-hop, Ballet" },
  { name: "Sports", icon: Swords, color: "text-orange-400", bg: "bg-orange-500/10", examples: "Tennis, Golf, Boxing" },
  { name: "Yoga", icon: Sparkles, color: "text-purple-400", bg: "bg-purple-500/10", examples: "Warrior, Tree, Crow" },
  { name: "PT & Rehab", icon: HeartPulse, color: "text-emerald-400", bg: "bg-emerald-500/10", examples: "Knee, Shoulder, Gait" },
  { name: "Sign Language", icon: Hand, color: "text-yellow-400", bg: "bg-yellow-500/10", examples: "ASL Signs, Alphabet" },
  { name: "Music", icon: Music, color: "text-cyan-400", bg: "bg-cyan-500/10", examples: "Guitar, Piano, Drums" },
  { name: "Ergonomics", icon: Monitor, color: "text-slate-400", bg: "bg-slate-500/10", examples: "Posture, Lifting, Typing" },
];

const CATEGORY_BADGE_COLOR: Record<string, string> = {
  Sports: "border-orange-500/40 text-orange-400",
  Dance: "border-pink-500/40 text-pink-400",
  Fitness: "border-blue-500/40 text-blue-400",
  Yoga: "border-purple-500/40 text-purple-400",
  "PT Rehab": "border-emerald-500/40 text-emerald-400",
};

export default function Dashboard() {
  const [greeting, setGreeting] = useState("");

  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting("Good morning");
    else if (hour < 17) setGreeting("Good afternoon");
    else setGreeting("Good evening");
  }, []);

  const todayScore = 79;
  const streak = 7;
  const skillsLearned = 12;
  const totalTime = "48 min";

  return (
    <div className="p-6 lg:p-8 space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">{greeting}</h1>
          <p className="text-muted-foreground mt-1">
            Learn any physical skill from any expert — in real-time, through voice.
          </p>
        </div>
        <Link href="/coach">
          <Button size="lg" className="gap-2 animate-pulse-glow">
            <Video className="h-4 w-4" />
            Start Coaching
          </Button>
        </Link>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-card border-border">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
              <TrendingUp className="h-6 w-6 text-primary" />
            </div>
            <div>
              <p className="text-2xl font-bold">{todayScore}</p>
              <p className="text-xs text-muted-foreground">Avg Score Today</p>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-orange-500/10">
              <Flame className="h-6 w-6 text-orange-500" />
            </div>
            <div>
              <p className="text-2xl font-bold">{streak}</p>
              <p className="text-xs text-muted-foreground">Day Streak</p>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-green-500/10">
              <Sparkles className="h-6 w-6 text-green-500" />
            </div>
            <div>
              <p className="text-2xl font-bold">{skillsLearned}</p>
              <p className="text-xs text-muted-foreground">Skills Learning</p>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-purple-500/10">
              <Clock className="h-6 w-6 text-purple-500" />
            </div>
            <div>
              <p className="text-2xl font-bold">{totalTime}</p>
              <p className="text-xs text-muted-foreground">Practice Time</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 3 Input Paths — the key product differentiator */}
      <Card className="bg-gradient-to-r from-primary/5 via-card to-purple-500/5 border-primary/20">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Start Learning Any Skill</CardTitle>
          <p className="text-xs text-muted-foreground">Three ways to begin — no training data needed</p>
        </CardHeader>
        <CardContent>
          <div className="grid sm:grid-cols-3 gap-3">
            <Link href="/coach?mode=video">
              <div className="group flex flex-col items-center gap-2 p-4 rounded-xl border border-border bg-secondary/20 hover:border-primary/40 hover:bg-primary/5 transition-all cursor-pointer">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 group-hover:bg-primary/20 transition-colors">
                  <Upload className="h-5 w-5 text-primary" />
                </div>
                <p className="text-sm font-medium">Expert Video</p>
                <p className="text-[10px] text-muted-foreground text-center leading-relaxed">
                  Upload a video or record live — AI extracts the skeleton and teaches you to match it
                </p>
              </div>
            </Link>

            <Link href="/coach?mode=voice">
              <div className="group flex flex-col items-center gap-2 p-4 rounded-xl border border-border bg-secondary/20 hover:border-primary/40 hover:bg-primary/5 transition-all cursor-pointer">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 group-hover:bg-primary/20 transition-colors">
                  <Mic className="h-5 w-5 text-primary" />
                </div>
                <p className="text-sm font-medium">Just Describe It</p>
                <p className="text-[10px] text-muted-foreground text-center leading-relaxed">
                  Say &ldquo;Coach my tennis serve&rdquo; — AI reasons about biomechanics, no video needed
                </p>
              </div>
            </Link>

            <Link href="/coach?mode=document">
              <div className="group flex flex-col items-center gap-2 p-4 rounded-xl border border-border bg-secondary/20 hover:border-primary/40 hover:bg-primary/5 transition-all cursor-pointer">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 group-hover:bg-primary/20 transition-colors">
                  <FileText className="h-5 w-5 text-primary" />
                </div>
                <p className="text-sm font-medium">From a Document</p>
                <p className="text-[10px] text-muted-foreground text-center leading-relaxed">
                  Upload a PT protocol, yoga manual, or coaching guide — instant live coaching
                </p>
              </div>
            </Link>
          </div>
        </CardContent>
      </Card>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Left column */}
        <div className="space-y-6">
          {/* Performance Ring */}
          <Card className="bg-card border-border">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Overall Performance
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col items-center pb-6">
              <ScoreRing score={todayScore} size={160} strokeWidth={10} label="score" />
              <div className="mt-4 flex gap-6 text-center">
                <div>
                  <p className="text-lg font-semibold text-green-500">+5</p>
                  <p className="text-[10px] text-muted-foreground">vs yesterday</p>
                </div>
                <div>
                  <p className="text-lg font-semibold">92</p>
                  <p className="text-[10px] text-muted-foreground">personal best</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Skill Categories */}
          <Card className="bg-card border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Explore Skills
              </CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-2">
              {SKILL_CATEGORIES.slice(0, 6).map((cat) => (
                <Link key={cat.name} href={`/skills?category=${encodeURIComponent(cat.name)}`}>
                  <Button
                    variant="outline"
                    className="w-full h-auto py-2.5 px-2 flex flex-col items-center gap-1 hover:border-primary/50 hover:bg-primary/5"
                  >
                    <cat.icon className={`h-4 w-4 ${cat.color}`} />
                    <span className="text-[11px] font-medium">{cat.name}</span>
                  </Button>
                </Link>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Right 2 columns */}
        <div className="lg:col-span-2 space-y-6">
          {/* AI Insight */}
          <Card className="bg-gradient-to-r from-primary/5 via-primary/10 to-purple-500/5 border-primary/20">
            <CardContent className="p-5 flex items-start gap-4">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/15">
                <Brain className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="font-medium text-sm">AI Coach Insight</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Your tennis serve shoulder rotation improved 20% since last session. For salsa,
                  your hip movement is getting more fluid — timing is 88% on beat. I noticed your
                  squat knee tracking is limited by ankle mobility. Try calf stretches before your
                  next set.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Recent Sessions — diverse skills */}
          <Card className="bg-card border-border">
            <CardHeader className="flex flex-row items-center justify-between pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Recent Sessions
              </CardTitle>
              <Link href="/history">
                <Button variant="ghost" size="sm" className="text-xs gap-1">
                  View all <ChevronRight className="h-3 w-3" />
                </Button>
              </Link>
            </CardHeader>
            <CardContent className="space-y-3">
              {MOCK_SESSIONS.map((session) => (
                <div
                  key={session.id}
                  className="flex items-center gap-4 rounded-lg border border-border bg-secondary/30 p-3 transition-colors hover:bg-secondary/50"
                >
                  <ScoreRing score={session.score} size={48} strokeWidth={4} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="font-medium text-sm">{session.skill}</p>
                      <Badge
                        variant="outline"
                        className={`text-[10px] ${CATEGORY_BADGE_COLOR[session.category] || "border-primary/40 text-primary"}`}
                      >
                        {session.category}
                      </Badge>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {session.reps} reps · {session.duration}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="flex items-center gap-1">
                      {session.trend === "up" && (
                        <TrendingUp className="h-3 w-3 text-green-500" />
                      )}
                      <span className="text-xs text-muted-foreground">{session.time}</span>
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Progress across diverse skills */}
          <Card className="bg-card border-border">
            <CardHeader className="flex flex-row items-center justify-between pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Skill Progress
              </CardTitle>
              <Link href="/skills">
                <Button variant="ghost" size="sm" className="text-xs gap-1">
                  All skills <ChevronRight className="h-3 w-3" />
                </Button>
              </Link>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { name: "Salsa Basic Step", category: "Dance", level: 88 },
                  { name: "Squat Form", category: "Fitness", level: 82 },
                  { name: "Tennis Serve", category: "Sports", level: 74 },
                  { name: "Warrior Pose", category: "Yoga", level: 91 },
                  { name: "Knee Extension", category: "PT", level: 85 },
                  { name: "Guitar Posture", category: "Music", level: 62 },
                ].map((skill) => (
                  <div key={skill.name} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span>
                        {skill.name}{" "}
                        <span className="text-muted-foreground">· {skill.category}</span>
                      </span>
                      <span className="text-muted-foreground tabular-nums">{skill.level}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-secondary">
                      <div
                        className="h-full rounded-full score-gradient transition-all duration-500"
                        style={{ width: `${skill.level}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

// stats cards + score ring
// quick start
// ai insight card
// recent sessions
// skill progress
