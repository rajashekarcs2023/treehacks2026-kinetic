"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Globe } from "lucide-react";

export default function PaperPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Top bar */}
      <div className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-sm">
        <div className="mx-auto max-w-4xl flex items-center justify-between px-6 py-3">
          <Link href="/">
            <Button variant="ghost" size="sm" className="gap-2">
              <ArrowLeft className="h-4 w-4" /> Back
            </Button>
          </Link>
          <span className="text-sm font-medium text-white/80">Technical Paper</span>
          <Link href="/product">
            <Button variant="outline" size="sm" className="gap-2">
              <Globe className="h-4 w-4" /> Product Page
            </Button>
          </Link>
        </div>
      </div>

      {/* Paper content */}
      <div className="mx-auto max-w-4xl px-6 py-12">
        <article className="rounded-xl border border-border bg-card p-8 md:p-12 shadow-lg space-y-8">
          {/* Title */}
          <div className="text-center space-y-4 pb-6 border-b border-border">
            <h1 className="text-2xl md:text-3xl font-bold leading-tight">
              Kinetic: A Cognitive Layer for Real-Time Physical Movement Intelligence
            </h1>
            <p className="text-base font-medium">
              Rajashekar Vennavelli
            </p>
            <p className="text-sm text-white/80">
              TreeHacks 2026, Stanford University, Stanford, 94305, CA, USA
            </p>
          </div>

          {/* Abstract */}
          <section>
            <h2 className="text-lg font-bold text-primary mb-3">Abstract</h2>
            <p className="text-sm leading-relaxed text-white/80">
              We present <strong className="text-foreground">Kinetic</strong>, the first real-time physical movement intelligence
              system that perceives, reasons about, and enhances human physical capability across multiple domains:
              skill coaching, physical therapy rehabilitation, autonomous fall detection, and spatial monitoring.
              Kinetic introduces a 4-tier expert generation pipeline that synthesizes biomechanically accurate
              reference poses from natural language descriptions, leveraging canonical biomechanical templates,
              large language model semantic mapping (Claude Sonnet 4), and state-of-the-art text-to-3D motion
              generation (HY-Motion 1.0-Lite on NVIDIA A100). Our system evaluates movement using a novel
              triple-metric scoring engine combining Gaussian joint angle comparison, cosine spatial similarity,
              and COCO Object Keypoint Similarity (OKS). Real-time feedback is delivered through conversational
              voice coaching (GPT-4o Realtime API) with 3-layer interruption handling. In autonomous monitoring
              mode, Kinetic watches physical spaces for falls, inactivity, and safety events — sending alerts
              via Telegram with photos. Edge AI inference on NVIDIA DGX Spark&apos;s GB10 Superchip provides
              sub-50ms pose estimation latency. The system orchestrates 44 MCP tools across 12 categories through
              the Anthropic Claude Agent SDK with 3 sub-agents. Kinetic demonstrates that AI-driven physical
              intelligence can augment human physical agency — helping people move, recover, express, and
              perform at their full potential.
            </p>
            <p className="text-sm text-white/80 mt-3">
              <strong className="text-foreground">Keywords:</strong> Physical Movement Intelligence, Pose Estimation, Motion Generation,
              Fall Detection, Autonomous Monitoring, Edge AI, Multi-Agent Systems, Real-Time Voice, Computer Vision, DGX Spark, Modal GPU, Claude Agent SDK
            </p>
          </section>

          {/* 1. Introduction */}
          <section>
            <h2 className="text-lg font-bold text-primary mb-3">1. Introduction</h2>
            <p className="text-sm leading-relaxed text-white/80">
              Today, AI augments cognitive intelligence — helping us write, code, and reason. Yet human physical
              capability remains fundamentally unaugmented. Movement literacy, motor skill development, rehabilitation,
              injury prevention, and safety monitoring still depend on expensive human experts ($80–200/hour for
              personal trainers, $100+/hour for physical therapists) or crude apps that count reps without
              understanding biomechanics. An estimated 1.7 billion people worldwide desire to learn physical
              skills, 55 million elderly Americans need fall monitoring, and millions need accessible PT rehab.
            </p>
            <p className="text-sm leading-relaxed text-white/80 mt-3">
              Recent advances in computer vision (MediaPipe [1], YOLOv8 [2]), large language models (Claude [3],
              GPT-4o [4]), and motion generation (HY-Motion 1.0 [5]) create an unprecedented opportunity to build AI
              coaching systems that see, understand, and communicate corrections in real-time. However, existing
              approaches suffer from three key limitations: (1) they require pre-recorded expert demonstrations,
              limiting skill coverage; (2) they use simplistic single-metric scoring that fails to capture the nuances
              of human movement; and (3) they lack natural communication interfaces, relying on visual overlays that
              users cannot observe during active movement.
            </p>
            <p className="text-sm leading-relaxed text-white/80 mt-3">
              Kinetic addresses all three limitations and goes further — introducing autonomous spatial monitoring
              for fall detection, goal-based reasoning for safety-critical environments, and multi-modal intelligence
              that spans coaching, rehabilitation, and patient monitoring. Our 4-tier expert generation pipeline
              eliminates the need for demonstration videos. Our triple-metric scoring engine provides robust,
              multi-dimensional form evaluation. And our voice-first interface delivers corrections naturally
              during active movement.
            </p>
          </section>

          {/* 2. System Architecture */}
          <section>
            <h2 className="text-lg font-bold text-primary mb-3">2. System Architecture</h2>
            <p className="text-sm leading-relaxed text-white/80">
              Kinetic&apos;s architecture spans six infrastructure pillars designed for real-time, multi-modal coaching:
            </p>
            <ol className="text-sm leading-relaxed text-white/80 mt-3 space-y-2 list-decimal list-inside">
              <li><strong className="text-foreground">NVIDIA DGX Spark (Edge AI)</strong> — YOLOv8n-pose on the GB10 Superchip for 17-keypoint pose estimation at 15+ FPS with &lt;50ms latency.</li>
              <li><strong className="text-foreground">Modal + NVIDIA A100 (Cloud GPU)</strong> — HY-Motion 1.0-Lite (0.46B params) deployed for text-to-3D motion generation, producing 30-frame skeleton sequences in ~26 seconds.</li>
              <li><strong className="text-foreground">Anthropic Claude Agent SDK</strong> — Multi-agent orchestration with 44 MCP tools across 12 categories, 3 sub-agents (Perception, Coach, Communicator), and 3 hooks (PreToolUse, PostToolUse, Stop).</li>
              <li><strong className="text-foreground">OpenAI GPT-4o Realtime API</strong> — Bidirectional voice coaching with 3-layer interruption handling and real-time pose score injection.</li>
              <li><strong className="text-foreground">Google MediaPipe + Ultralytics YOLO</strong> — 33 body + 21 hand landmarks tracked at 30 FPS with phase detection (preparation → execution → peak → recovery).</li>
              <li><strong className="text-foreground">Triple-Metric Scoring Engine</strong> — Gaussian joint angles (σ=15°) + Cosine spatial similarity + COCO OKS, weighted 40/30/30.</li>
            </ol>
            <p className="text-sm leading-relaxed text-white/80 mt-3">
              Data flows from camera frames through the CV pipeline to the scoring engine, with Claude orchestrating
              the coaching logic and GPT-4o delivering voice corrections. The architecture supports both edge-first
              inference (DGX Spark) for latency-critical pose estimation and cloud GPU inference (Modal A100) for
              computationally intensive motion generation.
            </p>
          </section>

          {/* 3. AI Expert Generation */}
          <section>
            <h2 className="text-lg font-bold text-primary mb-3">3. AI Expert Generation Pipeline</h2>
            <p className="text-sm leading-relaxed text-white/80">
              The core innovation of Kinetic is generating expert references from text alone, eliminating the
              requirement for demonstration videos. Our 4-tier pipeline provides graceful degradation with increasing
              latency and sophistication:
            </p>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-sm border border-border rounded-lg overflow-hidden">
                <thead>
                  <tr className="bg-muted/50">
                    <th className="text-left p-3 font-semibold">Tier</th>
                    <th className="text-left p-3 font-semibold">Method</th>
                    <th className="text-left p-3 font-semibold">Latency</th>
                    <th className="text-left p-3 font-semibold">Coverage</th>
                  </tr>
                </thead>
                <tbody className="text-white/80">
                  <tr className="border-t border-border">
                    <td className="p-3">1</td>
                    <td className="p-3">Canonical Templates (10+ exercises)</td>
                    <td className="p-3">&lt;1ms</td>
                    <td className="p-3">Common exercises</td>
                  </tr>
                  <tr className="border-t border-border">
                    <td className="p-3">2</td>
                    <td className="p-3">Claude Semantic Mapping</td>
                    <td className="p-3">~500ms</td>
                    <td className="p-3">Aliases &amp; variations</td>
                  </tr>
                  <tr className="border-t border-border">
                    <td className="p-3">3</td>
                    <td className="p-3">Claude Angle Generation</td>
                    <td className="p-3">~1s</td>
                    <td className="p-3">Any describable skill</td>
                  </tr>
                  <tr className="border-t border-border">
                    <td className="p-3">4</td>
                    <td className="p-3">HY-Motion 1.0 on A100</td>
                    <td className="p-3">~26s</td>
                    <td className="p-3">SOTA 3D motion sequences</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-sm leading-relaxed text-white/80 mt-3">
              Each tier is attempted in order, falling through to the next only when necessary. Tier 1 handles
              ~70% of coaching requests instantly. Tier 2 resolves natural language variations (&quot;barbell back
              squat&quot; → squat). Tier 3 uses Claude&apos;s biomechanical reasoning to generate per-phase joint
              angles for truly novel skills. Tier 4 leverages Tencent&apos;s HY-Motion 1.0-Lite, a 0.46B parameter
              text-to-3D motion diffusion model, to generate full skeleton sequences with temporal dynamics.
            </p>
          </section>

          {/* 4. Triple-Metric Scoring */}
          <section>
            <h2 className="text-lg font-bold text-primary mb-3">4. Triple-Metric Scoring Engine</h2>
            <p className="text-sm leading-relaxed text-white/80">
              Traditional pose scoring relies on a single metric (typically joint angle difference), which fails to
              capture spatial relationships and overall pose shape. We propose a triple-metric approach:
            </p>
            <ul className="text-sm leading-relaxed text-white/80 mt-3 space-y-3">
              <li>
                <strong className="text-foreground">Gaussian Joint Angles (weight: 0.4)</strong> — Each of 16 key
                joint angles is scored using a Gaussian function with σ=15°. This provides smooth, interpretable
                per-joint scores that naturally penalize large deviations while being tolerant of minor variations
                within the acceptable range.
              </li>
              <li>
                <strong className="text-foreground">Cosine Spatial Similarity (weight: 0.3)</strong> — Normalized
                skeleton vectors are compared using cosine similarity, capturing overall pose shape independent of
                body proportions. This metric detects global pose errors (e.g., leaning too far forward) that
                individual joint angles might miss.
              </li>
              <li>
                <strong className="text-foreground">COCO OKS (weight: 0.3)</strong> — Object Keypoint Similarity,
                the standard metric in academic pose estimation research (used in the COCO benchmark [6]), provides
                a weighted evaluation based on joint importance and localization accuracy. Each keypoint has a
                per-type standard deviation (σ<sub>k</sub>) reflecting its annotation variance.
              </li>
            </ul>
            <p className="text-sm leading-relaxed text-white/80 mt-3">
              The final score is a weighted combination: <code className="text-foreground bg-muted px-1.5 py-0.5 rounded text-xs">
              S = 0.4 × S_gaussian + 0.3 × S_cosine + 0.3 × S_oks</code>. This multi-metric approach provides
              a more robust and nuanced evaluation than any single metric alone.
            </p>
          </section>

          {/* 5. Autonomous Monitoring */}
          <section>
            <h2 className="text-lg font-bold text-primary mb-3">5. Autonomous Monitoring Mode</h2>
            <p className="text-sm leading-relaxed text-white/80">
              Beyond coaching, Kinetic operates as an autonomous spatial intelligence agent. In monitoring mode,
              the system accepts a goal (fall detection, desk security, posture watch, study focus) and runs a
              continuous perception→reasoning→action loop:
            </p>
            <ol className="text-sm leading-relaxed text-white/80 mt-3 space-y-2 list-decimal list-inside">
              <li><strong className="text-foreground">Perception</strong> — MediaPipe pose + YOLO detection at 2 FPS (efficiency-optimized for 24/7 monitoring).</li>
              <li><strong className="text-foreground">Activity Classification</strong> — Multi-signal approach with temporal smoothing: standing, walking, sitting, fallen, exercising, lying_down.</li>
              <li><strong className="text-foreground">Goal-Based Reasoning</strong> — Claude Agent SDK evaluates whether detected activity matches the configured goal&apos;s alert triggers.</li>
              <li><strong className="text-foreground">Autonomous Action</strong> — Sends Telegram alerts with snapshot photos, voice alerts via OpenAI Realtime, and logs to the tool call panel.</li>
            </ol>
            <p className="text-sm leading-relaxed text-white/80 mt-3">
              The monitoring loop is fully bidirectional: caregivers can send Telegram commands (/status, /goals,
              /photo, /start, /stop) to control the system remotely. For hospital deployment, the elderly_care
              goal prioritizes fall detection with immediate photo alerts — designed for patient safety scenarios
              where response time is critical.
            </p>
          </section>

          {/* 6. Edge AI */}
          <section>
            <h2 className="text-lg font-bold text-primary mb-3">6. Edge AI Inference on DGX Spark</h2>
            <p className="text-sm leading-relaxed text-white/80">
              Real-time coaching demands sub-100ms feedback latency. Cloud-based pose estimation introduces
              200-500ms of network latency, which creates a perceptible delay between movement and feedback.
              We deploy YOLOv8n-pose on the NVIDIA DGX Spark&apos;s GB10 Superchip, achieving 17-keypoint
              skeleton extraction at 15+ FPS with &lt;50ms end-to-end latency — a 4x improvement over cloud
              alternatives.
            </p>
            <p className="text-sm leading-relaxed text-white/80 mt-3">
              The DGX Spark runs our custom inference server that handles concurrent pose estimation requests
              while maintaining real-time frame rates. The pipeline processes each frame in a single pass:
              detection → keypoint extraction → skeleton normalization → joint angle computation. This tight
              feedback loop is critical for movement correction during active exercise.
            </p>
          </section>

          {/* 7. Voice Coaching */}
          <section>
            <h2 className="text-lg font-bold text-primary mb-3">7. Voice-First Coaching Interface</h2>
            <p className="text-sm leading-relaxed text-white/80">
              When users are actively performing physical movements, they cannot look at a screen. Voice is
              the only interface modality that works during active exercise. Kinetic uses OpenAI&apos;s GPT-4o
              Realtime API for bidirectional audio streaming with three key innovations:
            </p>
            <ol className="text-sm leading-relaxed text-white/80 mt-3 space-y-2 list-decimal list-inside">
              <li><strong className="text-foreground">3-Layer Interruption Handling</strong> — Users can interrupt mid-sentence; the AI pauses gracefully and adjusts its response based on the interruption context.</li>
              <li><strong className="text-foreground">Context Injection</strong> — Real-time pose scores are injected into the voice model&apos;s context window, ensuring corrections reference the user&apos;s current form rather than stale data.</li>
              <li><strong className="text-foreground">Fallback TTS</strong> — If the Realtime API is unavailable, the system falls back to standard TTS to maintain coaching continuity.</li>
            </ol>
          </section>

          {/* 8. Claude Agent SDK */}
          <section>
            <h2 className="text-lg font-bold text-primary mb-3">8. Multi-Agent Orchestration with Claude Agent SDK</h2>
            <p className="text-sm leading-relaxed text-white/80">
              Kinetic uses the Anthropic Claude Agent SDK for deep multi-agent orchestration. The physical
              intelligence brain consists of Claude Sonnet 4 with 46 MCP tools organized across 12 categories:
              spatial analysis, pose comparison, skill coaching, expert generation, recording, reference
              management, phase detection, rep counting, training data, document parsing, skill intelligence,
              and system configuration.
            </p>
            <p className="text-sm leading-relaxed text-white/80 mt-3">
              Three sub-agents specialize in different aspects: the <em>Perception Agent</em> handles
              computer vision and pose analysis; the <em>Coach Agent</em> manages form evaluation, corrections,
              and skill pedagogy; and the <em>Communicator Agent</em> orchestrates voice delivery and user
              interaction. Pre-tool and post-tool hooks enforce safety guardrails and maintain an audit log
              of all tool invocations.
            </p>
          </section>

          {/* 9. Results */}
          <section>
            <h2 className="text-lg font-bold text-primary mb-3">9. Implementation &amp; Results</h2>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-sm border border-border rounded-lg overflow-hidden">
                <thead>
                  <tr className="bg-muted/50">
                    <th className="text-left p-3 font-semibold">Component</th>
                    <th className="text-left p-3 font-semibold">Metric</th>
                    <th className="text-left p-3 font-semibold">Value</th>
                  </tr>
                </thead>
                <tbody className="text-white/80">
                  <tr className="border-t border-border"><td className="p-3">DGX Pose Estimation</td><td className="p-3">Latency</td><td className="p-3">&lt;50ms</td></tr>
                  <tr className="border-t border-border"><td className="p-3">DGX Pose Estimation</td><td className="p-3">Frame Rate</td><td className="p-3">15+ FPS</td></tr>
                  <tr className="border-t border-border"><td className="p-3">Modal HY-Motion</td><td className="p-3">Generation Time</td><td className="p-3">~26s (30 frames)</td></tr>
                  <tr className="border-t border-border"><td className="p-3">Modal HY-Motion</td><td className="p-3">Output Shape</td><td className="p-3">[30, 52, 3]</td></tr>
                  <tr className="border-t border-border"><td className="p-3">Expert Pipeline Tier 1</td><td className="p-3">Latency</td><td className="p-3">&lt;1ms</td></tr>
                  <tr className="border-t border-border"><td className="p-3">Expert Pipeline Tier 3</td><td className="p-3">Latency</td><td className="p-3">~1s</td></tr>
                  <tr className="border-t border-border"><td className="p-3">Claude Agent</td><td className="p-3">MCP Tools</td><td className="p-3">46 (HTTP-exposed)</td></tr>
                  <tr className="border-t border-border"><td className="p-3">Intelligence Modes</td><td className="p-3">Count</td><td className="p-3">4 (coach, PT, monitor, hospital)</td></tr>
                  <tr className="border-t border-border"><td className="p-3">Telegram Bot</td><td className="p-3">Commands</td><td className="p-3">8 bidirectional</td></tr>
                  <tr className="border-t border-border"><td className="p-3">Codebase</td><td className="p-3">Lines of Code</td><td className="p-3">17,000+</td></tr>
                  <tr className="border-t border-border"><td className="p-3">Build Time</td><td className="p-3">Duration</td><td className="p-3">20 hours (solo)</td></tr>
                </tbody>
              </table>
            </div>
          </section>

          {/* 10. Conclusion */}
          <section>
            <h2 className="text-lg font-bold text-primary mb-3">10. Conclusion</h2>
            <p className="text-sm leading-relaxed text-white/80">
              Kinetic demonstrates that real-time physical movement intelligence is achievable through the
              combination of edge AI inference, multi-agent LLM orchestration, and state-of-the-art motion
              generation. By unifying skill coaching, physical therapy, autonomous monitoring, and hospital
              safety into a single platform, Kinetic shows that the same CV + AI stack that coaches a squat
              can detect a fall, guide PT rehab, and monitor a hospital room. The triple-metric scoring
              engine provides robust form evaluation, the voice-first interface enables natural coaching
              during active movement, and the Telegram integration enables autonomous operation without
              any human in the loop.
            </p>
            <p className="text-sm leading-relaxed text-white/80 mt-3">
              AI has transformed cognitive work. Kinetic brings that transformation to physical capability.
              Future work will explore wearable AR integration for projected skeleton overlays,
              multi-camera 3D reconstruction for depth-aware fall detection, clinical PT dashboards for
              remote rehabilitation monitoring, and hospital deployment with ceiling-mounted cameras
              for 24/7 patient safety.
            </p>
          </section>

          {/* References */}
          <section className="pt-4 border-t border-border">
            <h2 className="text-lg font-bold text-primary mb-3">References</h2>
            <ol className="text-xs leading-relaxed text-white/80 space-y-1.5 list-decimal list-inside">
              <li>C. Lugaresi et al. &quot;MediaPipe: A Framework for Building Perception Pipelines.&quot; arXiv:1906.08172, 2019.</li>
              <li>G. Jocher et al. &quot;Ultralytics YOLOv8.&quot; https://github.com/ultralytics/ultralytics, 2023.</li>
              <li>Anthropic. &quot;Claude Agent SDK Documentation.&quot; https://docs.anthropic.com/agent-sdk, 2025.</li>
              <li>OpenAI. &quot;GPT-4o Realtime API.&quot; https://platform.openai.com/docs/guides/realtime, 2025.</li>
              <li>Tencent. &quot;HY-Motion 1.0: Text-to-3D Motion Generation.&quot; https://huggingface.co/tencent/HY-Motion-1.0, 2025.</li>
              <li>T. Lin et al. &quot;Microsoft COCO: Common Objects in Context.&quot; ECCV, 2014.</li>
              <li>NVIDIA. &quot;DGX Spark with GB10 Superchip.&quot; https://www.nvidia.com/dgx-spark, 2025.</li>
              <li>Modal Labs. &quot;Modal: Serverless GPU Infrastructure.&quot; https://modal.com, 2025.</li>
            </ol>
          </section>
        </article>
      </div>
    </div>
  );
}
