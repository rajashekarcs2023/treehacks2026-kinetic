"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Play, Pause, RotateCcw, Sparkles } from "lucide-react";

const SKELETON_CONNECTIONS = [
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // torso sides
  [23, 24], // hips
  [23, 25], [25, 27], // left leg
  [24, 26], [26, 28], // right leg
  [27, 29], [28, 30], // ankles to heels
  [27, 31], [28, 32], // ankles to toes
];

interface SkeletonPlayerProps {
  keyframes: number[][][]; // [frame][joint][x,y,vis]
  source: string;
  skill: string;
  generationMs?: number;
  model?: string;
  width?: number;
  height?: number;
}

export default function SkeletonPlayer({
  keyframes,
  source,
  skill,
  generationMs = 0,
  model = "",
  width = 320,
  height = 400,
}: SkeletonPlayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [playing, setPlaying] = useState(true);
  const [frameIdx, setFrameIdx] = useState(0);
  const animRef = useRef<number | null>(null);
  const lastTimeRef = useRef(0);

  const drawFrame = useCallback(
    (idx: number) => {
      const canvas = canvasRef.current;
      if (!canvas || !keyframes || keyframes.length === 0) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      canvas.width = width;
      canvas.height = height;
      ctx.clearRect(0, 0, width, height);

      // Background gradient
      const grad = ctx.createLinearGradient(0, 0, 0, height);
      grad.addColorStop(0, "rgba(124, 58, 237, 0.05)");
      grad.addColorStop(1, "rgba(16, 185, 129, 0.05)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, width, height);

      const frame = keyframes[idx % keyframes.length];
      if (!frame) return;

      // Find bounds for normalization
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const pt of frame) {
        if (pt[2] < 0.1) continue;
        minX = Math.min(minX, pt[0]);
        maxX = Math.max(maxX, pt[0]);
        minY = Math.min(minY, pt[1]);
        maxY = Math.max(maxY, pt[1]);
      }

      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;
      const padding = 40;
      const scaleX = (width - padding * 2) / rangeX;
      const scaleY = (height - padding * 2) / rangeY;
      const scale = Math.min(scaleX, scaleY);
      const offsetX = (width - rangeX * scale) / 2 - minX * scale;
      const offsetY = (height - rangeY * scale) / 2 - minY * scale;

      const toScreen = (x: number, y: number): [number, number] => [
        x * scale + offsetX,
        y * scale + offsetY,
      ];

      // Draw connections with gradient
      ctx.lineWidth = 3;
      for (const [i, j] of SKELETON_CONNECTIONS) {
        if (i >= frame.length || j >= frame.length) continue;
        const [x1, y1, v1] = frame[i];
        const [x2, y2, v2] = frame[j];
        if (v1 < 0.1 || v2 < 0.1) continue;
        const [sx1, sy1] = toScreen(x1, y1);
        const [sx2, sy2] = toScreen(x2, y2);

        const lineGrad = ctx.createLinearGradient(sx1, sy1, sx2, sy2);
        lineGrad.addColorStop(0, "rgba(124, 58, 237, 0.85)");
        lineGrad.addColorStop(1, "rgba(16, 185, 129, 0.85)");
        ctx.strokeStyle = lineGrad;
        ctx.beginPath();
        ctx.moveTo(sx1, sy1);
        ctx.lineTo(sx2, sy2);
        ctx.stroke();
      }

      // Draw joints
      for (let i = 0; i < Math.min(frame.length, 33); i++) {
        const [x, y, vis] = frame[i];
        if (vis < 0.1) continue;
        const [sx, sy] = toScreen(x, y);

        // Glow effect
        ctx.shadowColor = "rgba(124, 58, 237, 0.6)";
        ctx.shadowBlur = 8;
        ctx.fillStyle = vis > 0.5 ? "rgba(124, 58, 237, 1.0)" : "rgba(200, 150, 255, 0.7)";
        ctx.beginPath();
        ctx.arc(sx, sy, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        // White outline
        ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Frame counter
      ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
      ctx.font = "11px monospace";
      ctx.fillText(`Frame ${idx + 1}/${keyframes.length}`, 8, height - 8);
    },
    [keyframes, width, height]
  );

  // Animation loop
  useEffect(() => {
    if (!playing || !keyframes || keyframes.length === 0) return;

    const fps = 30;
    const interval = 1000 / fps;

    const animate = (time: number) => {
      if (time - lastTimeRef.current >= interval) {
        lastTimeRef.current = time;
        setFrameIdx((prev) => {
          const next = (prev + 1) % keyframes.length;
          drawFrame(next);
          return next;
        });
      }
      animRef.current = requestAnimationFrame(animate);
    };

    animRef.current = requestAnimationFrame(animate);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [playing, keyframes, drawFrame]);

  // Draw first frame
  useEffect(() => {
    if (keyframes && keyframes.length > 0) {
      drawFrame(0);
    }
  }, [keyframes, drawFrame]);

  const sourceLabel =
    source === "modal_hymotion_a100"
      ? "HY-Motion on NVIDIA A100"
      : source === "ai_generated"
      ? "Canonical Template"
      : source === "dgx_motion_generation"
      ? "DGX Spark GPU"
      : source;

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative rounded-xl overflow-hidden border-2 border-purple-500/30 bg-black/40">
        <canvas ref={canvasRef} width={width} height={height} className="block" />
        {/* Overlay badge */}
        <div className="absolute top-2 left-2 flex items-center gap-1.5 px-2 py-1 rounded-md bg-purple-600/80 backdrop-blur-sm">
          <Sparkles className="h-3 w-3 text-white" />
          <span className="text-[10px] font-bold text-white uppercase tracking-wider">AI Expert</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => setPlaying(!playing)}
          className="p-1.5 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 transition-colors"
        >
          {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </button>
        <button
          onClick={() => { setFrameIdx(0); drawFrame(0); }}
          className="p-1.5 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 transition-colors"
        >
          <RotateCcw className="h-4 w-4" />
        </button>
        <span className="text-[11px] text-muted-foreground">{keyframes.length} frames</span>
      </div>

      {/* Info */}
      <div className="text-center space-y-0.5">
        <p className="text-xs font-semibold text-purple-400">{skill}</p>
        <p className="text-[10px] text-muted-foreground">{sourceLabel}</p>
        {generationMs > 0 && (
          <p className="text-[10px] text-muted-foreground">Generated in {(generationMs / 1000).toFixed(1)}s</p>
        )}
      </div>
    </div>
  );
}
