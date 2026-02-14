"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Camera,
  Mic,
  Volume2,
  Wifi,
  Server,
  Brain,
  Database,
  RefreshCw,
  CheckCircle,
  XCircle,
  Zap,
  Shield,
  Download,
} from "lucide-react";

interface SystemStatus {
  camera: boolean;
  microphone: boolean;
  backend: boolean;
  mlModel: boolean;
  voiceEngine: boolean;
}

export default function SettingsPage() {
  const [status, setStatus] = useState<SystemStatus>({
    camera: false,
    microphone: false,
    backend: false,
    mlModel: false,
    voiceEngine: false,
  });
  const [isTraining, setIsTraining] = useState(false);
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");

  const checkBackend = async () => {
    try {
      const res = await fetch(`${apiUrl}/api/state`);
      setStatus((s) => ({ ...s, backend: res.ok }));
    } catch {
      setStatus((s) => ({ ...s, backend: false }));
    }
  };

  const trainModel = async () => {
    setIsTraining(true);
    try {
      await fetch(`${apiUrl}/api/model/bootstrap?n_per_skill=30&epochs=50`, {
        method: "POST",
      });
      setStatus((s) => ({ ...s, mlModel: true }));
    } catch {
      // training failed
    }
    setIsTraining(false);
  };

  const StatusDot = ({ active }: { active: boolean }) => (
    <div className={`h-2.5 w-2.5 rounded-full ${active ? "bg-green-500" : "bg-red-500/60"}`} />
  );

  return (
    <div className="p-6 lg:p-8 space-y-6 max-w-3xl mx-auto">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground mt-1">
          Configure your Kinetic setup and check system status.
        </p>
      </div>

      {/* Connection */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Wifi className="h-4 w-4 text-primary" />
            Backend Connection
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="flex-1">
              <label className="text-xs text-muted-foreground">API URL</label>
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="w-full mt-1 px-3 py-2 bg-secondary/50 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
            <Button onClick={checkBackend} variant="outline" className="mt-5 gap-2">
              <RefreshCw className="h-3.5 w-3.5" />
              Test
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <StatusDot active={status.backend} />
            <span className="text-sm">
              {status.backend ? "Connected" : "Not connected"}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* System Status */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Server className="h-4 w-4 text-primary" />
            System Status
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {[
            { label: "Camera Feed", icon: Camera, active: status.camera, desc: "WebSocket /ws/video" },
            { label: "Microphone", icon: Mic, active: status.microphone, desc: "WebSocket /ws/audio" },
            { label: "Backend Server", icon: Server, active: status.backend, desc: `${apiUrl}` },
            { label: "ML Models", icon: Brain, active: status.mlModel, desc: "Skill scorer + Activity classifier" },
            { label: "Voice Engine", icon: Volume2, active: status.voiceEngine, desc: "Gemini Live bidirectional" },
          ].map((item) => (
            <div
              key={item.label}
              className="flex items-center justify-between p-3 rounded-lg bg-secondary/30"
            >
              <div className="flex items-center gap-3">
                <item.icon className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">{item.label}</p>
                  <p className="text-[10px] text-muted-foreground">{item.desc}</p>
                </div>
              </div>
              {item.active ? (
                <CheckCircle className="h-4 w-4 text-green-500" />
              ) : (
                <XCircle className="h-4 w-4 text-red-500/60" />
              )}
            </div>
          ))}
        </CardContent>
      </Card>

      {/* ML Model Training */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary" />
            ML Model
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-xs text-muted-foreground">
            Bootstrap the skill scoring model with synthetic training data. This trains a
            PyTorch 1D CNN that scores movement quality in real-time.
          </p>
          <div className="flex items-center gap-3">
            <Button
              onClick={trainModel}
              disabled={isTraining}
              className="gap-2"
            >
              {isTraining ? (
                <>
                  <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                  Training...
                </>
              ) : (
                <>
                  <Zap className="h-3.5 w-3.5" />
                  Bootstrap Model
                </>
              )}
            </Button>
            <Button variant="outline" className="gap-2" disabled={isTraining}>
              <Download className="h-3.5 w-3.5" />
              Export Data
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <StatusDot active={status.mlModel} />
            <span className="text-xs text-muted-foreground">
              {status.mlModel ? "Model trained and ready" : "No model trained yet"}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* About */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Shield className="h-4 w-4 text-primary" />
            About Kinetic
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div>
              <p className="text-muted-foreground">Version</p>
              <p className="font-medium">1.0.0-alpha</p>
            </div>
            <div>
              <p className="text-muted-foreground">Built at</p>
              <p className="font-medium">TreeHacks 2026</p>
            </div>
            <div>
              <p className="text-muted-foreground">Backend</p>
              <p className="font-medium">FastAPI + 50 endpoints</p>
            </div>
            <div>
              <p className="text-muted-foreground">AI Agent</p>
              <p className="font-medium">Claude SDK + 43 MCP tools</p>
            </div>
            <div>
              <p className="text-muted-foreground">CV Pipeline</p>
              <p className="font-medium">YOLO11n + MediaPipe + ByteTrack</p>
            </div>
            <div>
              <p className="text-muted-foreground">Voice</p>
              <p className="font-medium">Gemini Live bidirectional</p>
            </div>
          </div>
          <Separator />
          <p className="text-[10px] text-muted-foreground">
            Privacy-first: all video is processed locally. No frames are stored or transmitted.
            Only structured pose data is used for coaching.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
