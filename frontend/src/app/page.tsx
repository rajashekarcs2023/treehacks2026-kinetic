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
  Sparkles,
  Dumbbell,
  Music,
  Swords,
  HeartPulse,
  Hand,
  Monitor,
  Globe,
  FileText,
  Layers,
  Eye,
  Shield,
  Stethoscope,
  Activity,
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
  { id: "1", skill: "Knee Extension", category: "PT Rehab", score: 85, reps: 10, duration: "4:30", time: "1h ago", trend: "up" },
  { id: "2", skill: "Warrior Pose", category: "Yoga", score: 91, reps: 6, duration: "6:15", time: "2h ago", trend: "up" },
  { id: "3", skill: "Sit-to-Stand", category: "Elderly", score: 78, reps: 8, duration: "3:00", time: "3h ago", trend: "up" },
  { id: "4", skill: "ASL Alphabet", category: "Sign Lang", score: 88, reps: 12, duration: "5:00", time: "Yesterday", trend: "up" },
  { id: "5", skill: "Salsa Basic Step", category: "Dance", score: 82, reps: 15, duration: "7:00", time: "Yesterday", trend: "same" },
];

const SKILL_CATEGORIES = [
  { name: "PT & Rehab", icon: HeartPulse, color: "text-emerald-400", bg: "bg-emerald-500/10", examples: "Knee, Shoulder, Hip" },
  { name: "Yoga", icon: Sparkles, color: "text-purple-400", bg: "bg-purple-500/10", examples: "Warrior, Tree, Sun" },
  { name: "Sign Language", icon: Hand, color: "text-yellow-400", bg: "bg-yellow-500/10", examples: "ASL, Greetings" },
  { name: "Elderly Care", icon: HeartPulse, color: "text-rose-400", bg: "bg-rose-500/10", examples: "Balance, Sit-Stand" },
  { name: "Ergonomics", icon: Monitor, color: "text-slate-400", bg: "bg-slate-500/10", examples: "Posture, Lifting" },
  { name: "Dance", icon: Music, color: "text-pink-400", bg: "bg-pink-500/10", examples: "Salsa, Ballet" },
  { name: "Fitness", icon: Dumbbell, color: "text-blue-400", bg: "bg-blue-500/10", examples: "Squats, Push-ups" },
  { name: "Sports", icon: Swords, color: "text-orange-400", bg: "bg-orange-500/10", examples: "Tennis, Golf" },
];

const CATEGORY_BADGE_COLOR: Record<string, string> = {
  "PT Rehab": "border-emerald-500/40 text-emerald-400",
  Yoga: "border-purple-500/40 text-purple-400",
  "Sign Lang": "border-yellow-500/40 text-yellow-400",
  Elderly: "border-rose-500/40 text-rose-400",
  Ergo: "border-slate-500/40 text-slate-400",
  Dance: "border-pink-500/40 text-pink-400",
  Fitness: "border-blue-500/40 text-blue-400",
  Sports: "border-orange-500/40 text-orange-400",
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
        <div className="flex items-start gap-6">
          {/* Showcase buttons — LEFT of greeting */}
          <div className="flex flex-col gap-2 shrink-0">
            <Link href="/product">
              <Button size="sm" className="gap-2 w-full justify-start bg-purple-600 hover:bg-purple-700 text-white">
                <Globe className="h-4 w-4" />
                Product
              </Button>
            </Link>
            <Link href="/paper">
              <Button size="sm" className="gap-2 w-full justify-start bg-blue-600 hover:bg-blue-700 text-white">
                <FileText className="h-4 w-4" />
                Tech Paper
              </Button>
            </Link>
            <Link href="/architecture">
              <Button size="sm" className="gap-2 w-full justify-start bg-emerald-600 hover:bg-emerald-700 text-white">
                <Layers className="h-4 w-4" />
                Architecture
              </Button>
            </Link>
          </div>
          <div>
            <h1 className="text-3xl font-bold tracking-tight">{greeting}</h1>
            <p className="text-muted-foreground mt-1">
              Physical Movement Intelligence — coaching, monitoring, rehab, and safety in real-time.
            </p>
          </div>
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

      {/* ── 4 Intelligence Modes ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Mode 1: AI Skill Coach */}
        <Link href="/coach">
          <Card className="group h-full bg-gradient-to-br from-primary/10 to-purple-500/5 border-primary/30 hover:border-primary/60 hover:shadow-lg hover:shadow-primary/10 transition-all cursor-pointer">
            <CardContent className="p-5 flex flex-col items-center text-center gap-3">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/20 group-hover:bg-primary/30 transition-colors">
                <Video className="h-7 w-7 text-primary" />
              </div>
              <div>
                <p className="text-sm font-bold">AI Skill Coach</p>
                <p className="text-[11px] text-muted-foreground mt-1 leading-tight">
                  Learn any movement from video, text, or AI-generated expert motion
                </p>
              </div>
              <Badge className="bg-primary/15 text-primary text-[10px] border-0">Voice + Skeleton</Badge>
            </CardContent>
          </Card>
        </Link>

        {/* Mode 2: Physical Therapy */}
        <Link href="/coach?mode=pt">
          <Card className="group h-full bg-gradient-to-br from-emerald-500/10 to-green-500/5 border-emerald-500/30 hover:border-emerald-500/60 hover:shadow-lg hover:shadow-emerald-500/10 transition-all cursor-pointer">
            <CardContent className="p-5 flex flex-col items-center text-center gap-3">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-500/20 group-hover:bg-emerald-500/30 transition-colors">
                <Stethoscope className="h-7 w-7 text-emerald-500" />
              </div>
              <div>
                <p className="text-sm font-bold">Physical Therapy</p>
                <p className="text-[11px] text-muted-foreground mt-1 leading-tight">
                  Guided rehab — knee, shoulder, hip, ankle, post-surgery recovery
                </p>
              </div>
              <Badge className="bg-emerald-500/15 text-emerald-400 text-[10px] border-0">PT Rehab</Badge>
            </CardContent>
          </Card>
        </Link>

        {/* Mode 3: Goal-Based Intelligence */}
        <Link href="/monitor">
          <Card className="group h-full bg-gradient-to-br from-blue-500/10 to-cyan-500/5 border-blue-500/30 hover:border-blue-500/60 hover:shadow-lg hover:shadow-blue-500/10 transition-all cursor-pointer">
            <CardContent className="p-5 flex flex-col items-center text-center gap-3">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-500/20 group-hover:bg-blue-500/30 transition-colors">
                <Eye className="h-7 w-7 text-blue-500" />
              </div>
              <div>
                <p className="text-sm font-bold">Goal-Based Intelligence</p>
                <p className="text-[11px] text-muted-foreground mt-1 leading-tight">
                  Give AI a goal — it watches, reasons, and acts autonomously
                </p>
              </div>
              <Badge className="bg-blue-500/15 text-blue-400 text-[10px] border-0">Autonomous Agent</Badge>
            </CardContent>
          </Card>
        </Link>

        {/* Mode 4: Clinical Patient Safety */}
        <Link href="/monitor?goal=elderly_care">
          <Card className="group h-full bg-gradient-to-br from-red-500/10 to-rose-500/5 border-red-500/30 hover:border-red-500/60 hover:shadow-lg hover:shadow-red-500/10 transition-all cursor-pointer">
            <CardContent className="p-5 flex flex-col items-center text-center gap-3">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-red-500/20 group-hover:bg-red-500/30 transition-colors">
                <Shield className="h-7 w-7 text-red-500" />
              </div>
              <div>
                <p className="text-sm font-bold">Clinical Patient Safety</p>
                <p className="text-[11px] text-muted-foreground mt-1 leading-tight">
                  Hospital fall detection, bed exit alerts, nurse Telegram notifications
                </p>
              </div>
              <Badge className="bg-red-500/15 text-red-400 text-[10px] border-0">Hospital Grade</Badge>
            </CardContent>
          </Card>
        </Link>
      </div>

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
                  Your knee extension range of motion improved 15° since last session — great rehab progress!
                  Warrior pose balance is excellent at 91%. For your elderly mobility routine,
                  sit-to-stand form is getting stronger. I recommend adding heel raises next to
                  improve ankle stability.
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
                  { name: "Warrior Pose", category: "Yoga", level: 91 },
                  { name: "Knee Extension", category: "PT Rehab", level: 85 },
                  { name: "ASL Alphabet", category: "Sign Lang", level: 88 },
                  { name: "Sit-to-Stand", category: "Elderly", level: 78 },
                  { name: "Salsa Basic", category: "Dance", level: 82 },
                  { name: "Desk Posture", category: "Ergo", level: 70 },
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

