"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { ScoreRing } from "@/components/score-ring";
import {
  Lock,
  Unlock,
  ChevronRight,
  Star,
  Dumbbell,
  HeartPulse,
  Sparkles,
  ArrowRight,
  Music,
  Swords,
  Hand,
  Monitor,
  Footprints,
} from "lucide-react";
import Link from "next/link";

interface Skill {
  id: string;
  name: string;
  category: string;
  proficiency: number;
  unlocked: boolean;
  prerequisites: string[];
  description: string;
}

const SKILL_TREES: Record<string, { icon: typeof Dumbbell; color: string; skills: Skill[] }> = {
  fitness: {
    icon: Dumbbell,
    color: "text-blue-400",
    skills: [
      { id: "squat", name: "Squat", category: "lower", proficiency: 82, unlocked: true, prerequisites: [], description: "Fundamental lower body movement" },
      { id: "lunge", name: "Lunge", category: "lower", proficiency: 58, unlocked: true, prerequisites: ["squat"], description: "Single-leg stability and strength" },
      { id: "bulgarian_split", name: "Bulgarian Split Squat", category: "lower", proficiency: 0, unlocked: false, prerequisites: ["lunge"], description: "Advanced single-leg squat variant" },
      { id: "push_up", name: "Push-up", category: "upper", proficiency: 45, unlocked: true, prerequisites: [], description: "Core upper body push movement" },
      { id: "deadlift", name: "Deadlift", category: "compound", proficiency: 35, unlocked: true, prerequisites: ["squat"], description: "Full posterior chain movement" },
      { id: "plank", name: "Plank", category: "core", proficiency: 68, unlocked: true, prerequisites: [], description: "Isometric core stability" },
      { id: "burpee", name: "Burpee", category: "compound", proficiency: 55, unlocked: true, prerequisites: ["push_up", "squat"], description: "Full body conditioning" },
      { id: "clean", name: "Power Clean", category: "compound", proficiency: 0, unlocked: false, prerequisites: ["deadlift"], description: "Explosive Olympic lift" },
    ],
  },
  dance: {
    icon: Music,
    color: "text-pink-400",
    skills: [
      { id: "salsa_basic", name: "Salsa Basic Step", category: "latin", proficiency: 88, unlocked: true, prerequisites: [], description: "Foundation of salsa dancing — forward/back basic" },
      { id: "salsa_cross", name: "Salsa Cross Body Lead", category: "latin", proficiency: 62, unlocked: true, prerequisites: ["salsa_basic"], description: "Lead partner across the slot" },
      { id: "bachata_basic", name: "Bachata Basic", category: "latin", proficiency: 75, unlocked: true, prerequisites: [], description: "Side-to-side with hip motion" },
      { id: "hiphop_groove", name: "Hip-hop Groove", category: "street", proficiency: 70, unlocked: true, prerequisites: [], description: "Bounce and rock fundamentals" },
      { id: "hiphop_wave", name: "Body Wave", category: "street", proficiency: 45, unlocked: true, prerequisites: ["hiphop_groove"], description: "Fluid body isolation wave" },
      { id: "ballet_plie", name: "Ballet Plié", category: "classical", proficiency: 55, unlocked: true, prerequisites: [], description: "Foundation of classical technique" },
      { id: "ballet_tendu", name: "Ballet Tendu", category: "classical", proficiency: 0, unlocked: false, prerequisites: ["ballet_plie"], description: "Foot extension from position" },
      { id: "popping", name: "Popping & Locking", category: "street", proficiency: 0, unlocked: false, prerequisites: ["hiphop_groove"], description: "Isolation and hit techniques" },
    ],
  },
  sports: {
    icon: Swords,
    color: "text-orange-400",
    skills: [
      { id: "tennis_serve", name: "Tennis Serve", category: "racket", proficiency: 74, unlocked: true, prerequisites: [], description: "Full kinetic chain serve motion" },
      { id: "tennis_forehand", name: "Tennis Forehand", category: "racket", proficiency: 68, unlocked: true, prerequisites: [], description: "Topspin forehand drive" },
      { id: "golf_swing", name: "Golf Swing", category: "club", proficiency: 42, unlocked: true, prerequisites: [], description: "Full swing — rotation and weight transfer" },
      { id: "boxing_jab", name: "Boxing Jab", category: "combat", proficiency: 80, unlocked: true, prerequisites: [], description: "Lead hand straight punch" },
      { id: "boxing_cross", name: "Boxing Cross", category: "combat", proficiency: 65, unlocked: true, prerequisites: ["boxing_jab"], description: "Rear hand power punch" },
      { id: "batting_stance", name: "Batting Stance", category: "bat", proficiency: 58, unlocked: true, prerequisites: [], description: "Baseball/cricket batting setup" },
      { id: "basketball_shot", name: "Free Throw", category: "ball", proficiency: 52, unlocked: true, prerequisites: [], description: "Basketball shooting form" },
      { id: "boxing_combo", name: "Jab-Cross-Hook", category: "combat", proficiency: 0, unlocked: false, prerequisites: ["boxing_jab", "boxing_cross"], description: "Three-punch power combination" },
    ],
  },
  yoga: {
    icon: Sparkles,
    color: "text-purple-400",
    skills: [
      { id: "mountain", name: "Mountain Pose", category: "standing", proficiency: 95, unlocked: true, prerequisites: [], description: "Foundation of all standing poses" },
      { id: "warrior1", name: "Warrior I", category: "standing", proficiency: 91, unlocked: true, prerequisites: ["mountain"], description: "Strength and focus" },
      { id: "warrior2", name: "Warrior II", category: "standing", proficiency: 85, unlocked: true, prerequisites: ["warrior1"], description: "Open hip stance with strength" },
      { id: "tree", name: "Tree Pose", category: "balance", proficiency: 78, unlocked: true, prerequisites: ["mountain"], description: "Single-leg balance" },
      { id: "downdog", name: "Downward Dog", category: "inversion", proficiency: 82, unlocked: true, prerequisites: [], description: "Full body stretch" },
      { id: "crow", name: "Crow Pose", category: "arm_balance", proficiency: 0, unlocked: false, prerequisites: ["tree"], description: "Arm balance foundation" },
      { id: "headstand", name: "Headstand", category: "inversion", proficiency: 0, unlocked: false, prerequisites: ["downdog", "crow"], description: "King of inversions" },
    ],
  },
  martial_arts: {
    icon: Footprints,
    color: "text-red-400",
    skills: [
      { id: "front_kick", name: "Front Kick", category: "kicks", proficiency: 72, unlocked: true, prerequisites: [], description: "Basic linear push kick" },
      { id: "roundhouse", name: "Roundhouse Kick", category: "kicks", proficiency: 55, unlocked: true, prerequisites: ["front_kick"], description: "Circular power kick" },
      { id: "side_kick", name: "Side Kick", category: "kicks", proficiency: 0, unlocked: false, prerequisites: ["front_kick"], description: "Lateral thrust kick" },
      { id: "horse_stance", name: "Horse Stance", category: "stances", proficiency: 85, unlocked: true, prerequisites: [], description: "Wide squat stance — build leg endurance" },
      { id: "kata_basic", name: "Basic Kata", category: "forms", proficiency: 40, unlocked: true, prerequisites: ["horse_stance", "front_kick"], description: "Choreographed movement sequence" },
      { id: "spinning_kick", name: "Spinning Back Kick", category: "kicks", proficiency: 0, unlocked: false, prerequisites: ["roundhouse", "side_kick"], description: "Advanced rotational kick" },
    ],
  },
  pt_rehab: {
    icon: HeartPulse,
    color: "text-emerald-400",
    skills: [
      { id: "wall_sit", name: "Wall Sit", category: "knee", proficiency: 90, unlocked: true, prerequisites: [], description: "Isometric quad strengthening" },
      { id: "knee_ext", name: "Knee Extension", category: "knee", proficiency: 85, unlocked: true, prerequisites: [], description: "Controlled quad activation" },
      { id: "step_up", name: "Step Up", category: "knee", proficiency: 72, unlocked: true, prerequisites: ["wall_sit"], description: "Functional leg strength" },
      { id: "clamshell", name: "Clamshell", category: "hip", proficiency: 88, unlocked: true, prerequisites: [], description: "Glute med activation" },
      { id: "bridge", name: "Glute Bridge", category: "hip", proficiency: 78, unlocked: true, prerequisites: ["clamshell"], description: "Posterior chain activation" },
      { id: "shoulder_raise", name: "Lateral Raise", category: "shoulder", proficiency: 65, unlocked: true, prerequisites: [], description: "Shoulder mobility and strength" },
      { id: "single_leg_bridge", name: "Single Leg Bridge", category: "hip", proficiency: 0, unlocked: false, prerequisites: ["bridge"], description: "Advanced glute activation" },
    ],
  },
  music: {
    icon: Music,
    color: "text-cyan-400",
    skills: [
      { id: "guitar_posture", name: "Guitar Posture", category: "guitar", proficiency: 62, unlocked: true, prerequisites: [], description: "Seated position, neck angle, wrist alignment" },
      { id: "guitar_chords", name: "Chord Transitions", category: "guitar", proficiency: 45, unlocked: true, prerequisites: ["guitar_posture"], description: "Smooth fretting hand movement" },
      { id: "piano_hands", name: "Piano Hand Position", category: "piano", proficiency: 70, unlocked: true, prerequisites: [], description: "Curved fingers, relaxed wrists" },
      { id: "drum_grip", name: "Drum Grip & Stroke", category: "drums", proficiency: 55, unlocked: true, prerequisites: [], description: "Matched grip, controlled rebound" },
      { id: "violin_bow", name: "Violin Bow Hold", category: "strings", proficiency: 38, unlocked: true, prerequisites: [], description: "Pinky curve, relaxed thumb" },
      { id: "piano_scales", name: "Piano Scales", category: "piano", proficiency: 0, unlocked: false, prerequisites: ["piano_hands"], description: "Thumb-under technique" },
    ],
  },
  sign_language: {
    icon: Hand,
    color: "text-yellow-400",
    skills: [
      { id: "asl_alphabet", name: "ASL Alphabet", category: "basics", proficiency: 80, unlocked: true, prerequisites: [], description: "Finger spelling A-Z" },
      { id: "asl_numbers", name: "ASL Numbers", category: "basics", proficiency: 75, unlocked: true, prerequisites: [], description: "Counting 1-100 in ASL" },
      { id: "asl_greetings", name: "Common Greetings", category: "phrases", proficiency: 60, unlocked: true, prerequisites: ["asl_alphabet"], description: "Hello, thank you, please, sorry" },
      { id: "asl_questions", name: "Question Signs", category: "phrases", proficiency: 42, unlocked: true, prerequisites: ["asl_greetings"], description: "Who, what, where, when, why" },
      { id: "asl_conversation", name: "Basic Conversation", category: "advanced", proficiency: 0, unlocked: false, prerequisites: ["asl_greetings", "asl_questions"], description: "Fluid conversational signing" },
    ],
  },
};

const SUB_CATEGORIES: Record<string, string> = {
  lower: "Lower Body", upper: "Upper Body", compound: "Compound", core: "Core",
  latin: "Latin", street: "Street", classical: "Classical",
  racket: "Racket Sports", club: "Club Sports", combat: "Combat", bat: "Bat Sports", ball: "Ball Sports",
  standing: "Standing", balance: "Balance", arm_balance: "Arm Balance", inversion: "Inversion",
  kicks: "Kicks", stances: "Stances", forms: "Forms",
  knee: "Knee", hip: "Hip", shoulder: "Shoulder",
  guitar: "Guitar", piano: "Piano", drums: "Drums", strings: "Strings",
  basics: "Basics", phrases: "Phrases", advanced: "Advanced",
};

const SUB_CATEGORY_COLORS: Record<string, string> = {
  lower: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  upper: "bg-purple-500/10 text-purple-400 border-purple-500/20",
  compound: "bg-orange-500/10 text-orange-400 border-orange-500/20",
  core: "bg-green-500/10 text-green-400 border-green-500/20",
  latin: "bg-pink-500/10 text-pink-400 border-pink-500/20",
  street: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  classical: "bg-rose-500/10 text-rose-400 border-rose-500/20",
  racket: "bg-orange-500/10 text-orange-400 border-orange-500/20",
  combat: "bg-red-500/10 text-red-400 border-red-500/20",
  club: "bg-lime-500/10 text-lime-400 border-lime-500/20",
  bat: "bg-amber-500/10 text-amber-400 border-amber-500/20",
  ball: "bg-sky-500/10 text-sky-400 border-sky-500/20",
  standing: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
  balance: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  arm_balance: "bg-red-500/10 text-red-400 border-red-500/20",
  inversion: "bg-pink-500/10 text-pink-400 border-pink-500/20",
  kicks: "bg-red-500/10 text-red-400 border-red-500/20",
  stances: "bg-amber-500/10 text-amber-400 border-amber-500/20",
  forms: "bg-violet-500/10 text-violet-400 border-violet-500/20",
  knee: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  hip: "bg-teal-500/10 text-teal-400 border-teal-500/20",
  shoulder: "bg-sky-500/10 text-sky-400 border-sky-500/20",
  guitar: "bg-amber-500/10 text-amber-400 border-amber-500/20",
  piano: "bg-slate-500/10 text-slate-400 border-slate-500/20",
  drums: "bg-orange-500/10 text-orange-400 border-orange-500/20",
  strings: "bg-rose-500/10 text-rose-400 border-rose-500/20",
  basics: "bg-green-500/10 text-green-400 border-green-500/20",
  phrases: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  advanced: "bg-purple-500/10 text-purple-400 border-purple-500/20",
};

function getTreeStats(skills: Skill[]) {
  const total = skills.length;
  const unlocked = skills.filter((s) => s.unlocked).length;
  const mastered = skills.filter((s) => s.proficiency >= 80).length;
  const avg = skills.filter((s) => s.unlocked).reduce((a, s) => a + s.proficiency, 0) / (unlocked || 1);
  return { total, unlocked, mastered, avg };
}

const TREE_NAMES: Record<string, string> = {
  fitness: "Fitness", dance: "Dance", sports: "Sports", yoga: "Yoga",
  martial_arts: "Martial Arts", pt_rehab: "PT & Rehab", music: "Music", sign_language: "Sign Language",
};

export default function SkillsPage() {
  const [activeTree, setActiveTree] = useState("fitness");
  const treeData = SKILL_TREES[activeTree];
  const skills = treeData.skills;
  const stats = getTreeStats(skills);
  const categories = [...new Set(skills.map((s) => s.category))];

  return (
    <div className="p-6 lg:p-8 space-y-6 max-w-7xl mx-auto">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Skills</h1>
        <p className="text-muted-foreground mt-1">
          Any physical skill. Track progression, unlock new movements, master your body.
        </p>
      </div>

      {/* Tree Selector — scrollable tabs for all categories */}
      <Tabs value={activeTree} onValueChange={setActiveTree}>
        <ScrollArea className="w-full">
          <TabsList className="bg-secondary/50 inline-flex w-max">
            {Object.entries(SKILL_TREES).map(([key, tree]) => (
              <TabsTrigger key={key} value={key} className="gap-1.5">
                <tree.icon className={`h-3.5 w-3.5`} />
                {TREE_NAMES[key]}
              </TabsTrigger>
            ))}
          </TabsList>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>

        <TabsContent value={activeTree} className="mt-6 space-y-6">
          {/* Stats Overview */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="bg-card border-border">
              <CardContent className="p-4 text-center">
                <p className="text-3xl font-bold">{stats.total}</p>
                <p className="text-xs text-muted-foreground">Total Skills</p>
              </CardContent>
            </Card>
            <Card className="bg-card border-border">
              <CardContent className="p-4 text-center">
                <p className="text-3xl font-bold text-green-500">{stats.unlocked}</p>
                <p className="text-xs text-muted-foreground">Unlocked</p>
              </CardContent>
            </Card>
            <Card className="bg-card border-border">
              <CardContent className="p-4 text-center">
                <p className="text-3xl font-bold text-primary">{stats.mastered}</p>
                <p className="text-xs text-muted-foreground">Mastered</p>
              </CardContent>
            </Card>
            <Card className="bg-card border-border">
              <CardContent className="p-4 text-center">
                <p className="text-3xl font-bold">{Math.round(stats.avg)}%</p>
                <p className="text-xs text-muted-foreground">Avg Proficiency</p>
              </CardContent>
            </Card>
          </div>

          {/* Skills by Sub-Category */}
          {categories.map((cat) => {
            const catSkills = skills.filter((s) => s.category === cat);
            return (
              <div key={cat}>
                <div className="flex items-center gap-2 mb-3">
                  <Badge variant="outline" className={SUB_CATEGORY_COLORS[cat] || ""}>
                    {SUB_CATEGORIES[cat] || cat}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {catSkills.filter((s) => s.unlocked).length}/{catSkills.length} unlocked
                  </span>
                </div>
                <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {catSkills.map((skill) => (
                    <Card
                      key={skill.id}
                      className={`bg-card border-border transition-all hover:border-primary/30 ${
                        !skill.unlocked ? "opacity-50" : ""
                      }`}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              {skill.unlocked ? (
                                <Unlock className="h-3.5 w-3.5 text-green-500" />
                              ) : (
                                <Lock className="h-3.5 w-3.5 text-muted-foreground" />
                              )}
                              <h3 className="font-medium text-sm">{skill.name}</h3>
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                              {skill.description}
                            </p>
                            {skill.prerequisites.length > 0 && (
                              <div className="flex items-center gap-1 mt-2 flex-wrap">
                                <span className="text-[10px] text-muted-foreground">Requires:</span>
                                {skill.prerequisites.map((p) => (
                                  <Badge
                                    key={p}
                                    variant="outline"
                                    className="text-[10px] py-0 px-1.5"
                                  >
                                    {p.replace(/_/g, " ")}
                                  </Badge>
                                ))}
                              </div>
                            )}
                          </div>
                          <ScoreRing
                            score={skill.proficiency}
                            size={48}
                            strokeWidth={4}
                          />
                        </div>
                        {skill.unlocked && skill.proficiency > 0 && (
                          <div className="mt-3">
                            <div className="h-1.5 rounded-full bg-secondary">
                              <div
                                className="h-full rounded-full score-gradient transition-all duration-500"
                                style={{ width: `${skill.proficiency}%` }}
                              />
                            </div>
                          </div>
                        )}
                        {skill.unlocked && (
                          <Link href={`/coach?skill=${encodeURIComponent(skill.name)}`}>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="w-full mt-3 text-xs gap-1 text-primary hover:text-primary hover:bg-primary/10"
                            >
                              Practice <ArrowRight className="h-3 w-3" />
                            </Button>
                          </Link>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            );
          })}

          {/* Recommendations */}
          <Card className="bg-gradient-to-r from-primary/5 via-primary/10 to-purple-500/5 border-primary/20">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Star className="h-4 w-4 text-primary" />
                Recommended Next
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {skills
                  .filter((s) => !s.unlocked)
                  .slice(0, 3)
                  .map((skill) => (
                    <div
                      key={skill.id}
                      className="flex items-center justify-between p-2 rounded-lg bg-black/20"
                    >
                      <div>
                        <p className="text-sm font-medium">{skill.name}</p>
                        <p className="text-xs text-muted-foreground">
                          Unlock by mastering: {skill.prerequisites.join(", ").replace(/_/g, " ")}
                        </p>
                      </div>
                      <ChevronRight className="h-4 w-4 text-muted-foreground" />
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
