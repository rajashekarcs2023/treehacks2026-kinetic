/**
 * AEGIS Phone Web App
 * 
 * Architecture:
 *   Camera → captures frames → WebSocket → AEGIS backend (CV pipeline)
 *   Backend → spatial state → WebSocket → overlay on phone screen
 *   Mic audio → Gemini Live (direct client-to-server) → speaker
 *   Spatial state injected into Gemini Live as text context
 * 
 * Clean separation:
 *   - VideoStream: handles camera + frame sending
 *   - SpatialOverlay: draws bounding boxes + labels on canvas
 *   - GeminiVoice: handles Gemini Live audio I/O
 *   - App: orchestrates everything
 */

// ═══════════════════════════════════════════════════════════════════════
// VideoStream — Camera capture + WebSocket frame streaming
// ═══════════════════════════════════════════════════════════════════════

class VideoStream {
    constructor() {
        this.video = document.getElementById('camera-video');
        this.ws = null;
        this.stream = null;
        this.sendCanvas = document.createElement('canvas');
        this.sendCtx = this.sendCanvas.getContext('2d');
        this.facingMode = 'environment'; // rear camera
        this.sending = false;
        this.frameInterval = null;
        this.targetFPS = 8; // frames sent to server per second
        this.onStateUpdate = null; // callback for spatial state
    }

    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: this.facingMode,
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                },
                audio: false, // audio handled separately by Gemini
            });
            this.video.srcObject = this.stream;
            await this.video.play();
            return true;
        } catch (err) {
            console.error('Camera error:', err);
            return false;
        }
    }

    async flipCamera() {
        this.facingMode = this.facingMode === 'environment' ? 'user' : 'environment';
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
        }
        return this.startCamera();
    }

    connectWebSocket() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${location.host}/ws/video`;
        
        this.ws = new WebSocket(wsUrl);
        this.ws.onopen = () => {
            console.log('[Video] WebSocket connected');
            this.sending = true;
            this.startSending();
        };
        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'state' && this.onStateUpdate) {
                this.onStateUpdate(msg.data);
            }
        };
        this.ws.onclose = () => {
            console.log('[Video] WebSocket disconnected');
            this.sending = false;
            this.stopSending();
        };
        this.ws.onerror = (err) => {
            console.error('[Video] WebSocket error:', err);
        };
    }

    startSending() {
        if (this.frameInterval) return;
        this.frameInterval = setInterval(() => this.sendFrame(), 1000 / this.targetFPS);
    }

    stopSending() {
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
    }

    sendFrame() {
        if (!this.sending || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (!this.video.videoWidth) return;

        // Scale down for bandwidth
        const scale = 0.5;
        this.sendCanvas.width = this.video.videoWidth * scale;
        this.sendCanvas.height = this.video.videoHeight * scale;
        this.sendCtx.drawImage(this.video, 0, 0, this.sendCanvas.width, this.sendCanvas.height);

        // Convert to JPEG base64
        const dataUrl = this.sendCanvas.toDataURL('image/jpeg', 0.6);
        const base64 = dataUrl.split(',')[1];

        this.ws.send(JSON.stringify({ type: 'frame', data: base64 }));
    }

    stop() {
        this.sending = false;
        this.stopSending();
        if (this.ws) this.ws.close();
        if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    }
}


// ═══════════════════════════════════════════════════════════════════════
// SpatialOverlay — Draws bounding boxes, labels, activities on canvas
// ═══════════════════════════════════════════════════════════════════════

class SpatialOverlay {
    constructor() {
        this.canvas = document.getElementById('overlay-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.lastState = null;
    }

    update(state) {
        this.lastState = state;
        this.draw();
    }

    draw() {
        const state = this.lastState;
        if (!state) return;

        const video = document.getElementById('camera-video');
        const rect = video.getBoundingClientRect();
        
        // Match canvas to video display size
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const frameW = state.frame_size?.width || 640;
        const frameH = state.frame_size?.height || 480;
        const scaleX = rect.width / frameW;
        const scaleY = rect.height / frameH;

        // ── Draw persons ────────────────────────────────
        for (const p of (state.persons || [])) {
            const b = p.bbox;
            const x1 = b.x1 * scaleX, y1 = b.y1 * scaleY;
            const w = (b.x2 - b.x1) * scaleX, h = (b.y2 - b.y1) * scaleY;
            const activity = p.activity || '';

            // Color based on activity
            let color = '#00ff88';
            if (activity === 'fallen' || activity === 'lying_down') color = '#ff4444';
            else if (activity === 'running') color = '#ffaa00';
            else if (activity === 'waving') color = '#00aaff';

            // Bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, w, h);

            // Label background
            const label = `ID:${p.track_id} ${activity}`;
            ctx.font = 'bold 12px -apple-system, sans-serif';
            const textW = ctx.measureText(label).width;
            ctx.fillStyle = color;
            ctx.fillRect(x1, y1 - 18, textW + 8, 18);
            ctx.fillStyle = '#000';
            ctx.fillText(label, x1 + 4, y1 - 4);
        }

        // ── Draw objects ────────────────────────────────
        for (const obj of (state.objects || [])) {
            const b = obj.bbox;
            const x1 = b.x1 * scaleX, y1 = b.y1 * scaleY;
            const w = (b.x2 - b.x1) * scaleX, h = (b.y2 - b.y1) * scaleY;

            ctx.strokeStyle = 'rgba(255,200,0,0.6)';
            ctx.lineWidth = 1;
            ctx.strokeRect(x1, y1, w, h);

            ctx.font = '11px -apple-system, sans-serif';
            ctx.fillStyle = 'rgba(255,200,0,0.8)';
            ctx.fillText(obj.class_name, x1 + 2, y1 - 3);
        }

        // ── Draw danger zones ───────────────────────────
        for (const z of (state.danger_zones || [])) {
            const b = z.bbox;
            const x1 = b.x1 * scaleX, y1 = b.y1 * scaleY;
            const w = (b.x2 - b.x1) * scaleX, h = (b.y2 - b.y1) * scaleY;

            ctx.strokeStyle = 'rgba(255,0,0,0.4)';
            ctx.lineWidth = 1;
            ctx.setLineDash([6, 4]);
            ctx.strokeRect(x1, y1, w, h);
            ctx.setLineDash([]);
        }

        // ── Risk warnings ───────────────────────────────
        const risks = state.risk_events || [];
        if (risks.length > 0) {
            ctx.font = 'bold 14px -apple-system, sans-serif';
            ctx.fillStyle = '#ff4444';
            risks.forEach((r, i) => {
                ctx.fillText(`⚠ ${r.description}`, 10, 60 + i * 20);
            });
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
// GeminiVoice — Gemini Live real-time voice I/O
// ═══════════════════════════════════════════════════════════════════════

class GeminiVoice {
    constructor() {
        this.ws = null;
        this.audioContext = null;
        this.micStream = null;
        this.micProcessor = null;
        this.isConnected = false;
        this.isListening = false;
        this.config = null;
        this.spatialContext = '';
        this.playbackQueue = [];
        this.isPlaying = false;
    }

    async init() {
        // Fetch Gemini config from backend
        const resp = await fetch('/api/config');
        this.config = await resp.json();
        if (!this.config.gemini_api_key) {
            console.warn('[Voice] No Gemini API key configured');
            return false;
        }
        return true;
    }

    async connect() {
        if (!this.config?.gemini_api_key) return false;

        const model = this.config.gemini_model;
        const apiKey = this.config.gemini_api_key;
        const wsUrl = `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=${apiKey}`;

        return new Promise((resolve) => {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('[Voice] Gemini Live WebSocket connected');
                // Send setup message
                const setup = {
                    setup: {
                        model: `models/${model}`,
                        generation_config: {
                            response_modalities: ["AUDIO"],
                            speech_config: {
                                voice_config: {
                                    prebuilt_voice_config: {
                                        voice_name: this.config.gemini_voice || "Kore"
                                    }
                                }
                            }
                        },
                        system_instruction: {
                            parts: [{
                                text: `You are AEGIS, a friendly spatial AI assistant. You can see and understand physical spaces in real-time through a camera.
                                
Your job is to narrate what you see when asked, and proactively alert if something important happens (like someone falling, or a new person entering).

You will receive periodic spatial state updates as text messages. These contain structured data about:
- People detected (positions, activities like standing/sitting/walking/running/fallen)
- Objects detected (chairs, laptops, cups, etc.)
- Risk events (someone approaching a monitored zone)

When describing the scene:
- Be natural and conversational, like a helpful friend
- Use spatial language: "to the left", "near the desk", "walking toward you"
- Mention activities: "sitting still", "just stood up", "walking to the right"
- Keep responses concise (1-3 sentences usually)
- If someone falls, alert IMMEDIATELY with urgency
- If asked "what do you see?", give a full scene description`
                            }]
                        }
                    }
                };
                this.ws.send(JSON.stringify(setup));
                this.isConnected = true;
                resolve(true);
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(JSON.parse(event.data));
            };

            this.ws.onclose = () => {
                console.log('[Voice] Gemini Live disconnected');
                this.isConnected = false;
                this.isListening = false;
            };

            this.ws.onerror = (err) => {
                console.error('[Voice] Gemini error:', err);
                resolve(false);
            };
        });
    }

    handleMessage(msg) {
        // Handle setup complete
        if (msg.setupComplete) {
            console.log('[Voice] Gemini Live setup complete');
            return;
        }

        // Handle audio response
        if (msg.serverContent?.modelTurn?.parts) {
            for (const part of msg.serverContent.modelTurn.parts) {
                if (part.inlineData?.data) {
                    // Queue audio for playback
                    const audioBytes = this.base64ToArrayBuffer(part.inlineData.data);
                    this.playbackQueue.push(audioBytes);
                    if (!this.isPlaying) {
                        this.playNextAudio();
                    }
                }
            }
        }

        // Handle interruption
        if (msg.serverContent?.interrupted) {
            this.playbackQueue = [];
            this.isPlaying = false;
        }
    }

    async startListening() {
        if (!this.isConnected) return;

        try {
            this.audioContext = new AudioContext({ sampleRate: 16000 });
            this.micStream = await navigator.mediaDevices.getUserMedia({
                audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true }
            });

            const source = this.audioContext.createMediaStreamSource(this.micStream);
            this.micProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);

            this.micProcessor.onaudioprocess = (e) => {
                if (!this.isListening || !this.isConnected) return;
                const float32 = e.inputBuffer.getChannelData(0);
                const int16 = this.float32ToInt16(float32);
                const base64 = this.arrayBufferToBase64(int16.buffer);

                this.ws.send(JSON.stringify({
                    realtime_input: {
                        media_chunks: [{
                            data: base64,
                            mime_type: "audio/pcm;rate=16000"
                        }]
                    }
                }));
            };

            source.connect(this.micProcessor);
            this.micProcessor.connect(this.audioContext.destination);
            this.isListening = true;
            console.log('[Voice] Listening started');
        } catch (err) {
            console.error('[Voice] Mic error:', err);
        }
    }

    stopListening() {
        this.isListening = false;
        if (this.micProcessor) {
            this.micProcessor.disconnect();
            this.micProcessor = null;
        }
        if (this.micStream) {
            this.micStream.getTracks().forEach(t => t.stop());
            this.micStream = null;
        }
        console.log('[Voice] Listening stopped');
    }

    updateSpatialContext(state) {
        if (!this.isConnected || !state) return;

        // Build a concise text description of the spatial state
        const parts = [];
        const persons = state.persons || [];
        const objects = state.objects || [];
        const risks = state.risk_events || [];

        if (persons.length === 0) {
            parts.push('No people detected.');
        } else {
            const pDescs = persons.map(p =>
                `Person ${p.track_id}: ${p.activity || 'detected'} at (${p.center.x},${p.center.y}), speed=${Math.round(p.speed_px_per_sec)}px/s`
            );
            parts.push(`${persons.length} person(s): ${pDescs.join('; ')}`);
        }

        if (objects.length > 0) {
            const objCounts = {};
            objects.forEach(o => { objCounts[o.class_name] = (objCounts[o.class_name] || 0) + 1; });
            const objStr = Object.entries(objCounts).map(([k,v]) => `${v}x ${k}`).join(', ');
            parts.push(`Objects: ${objStr}`);
        }

        if (risks.length > 0) {
            parts.push(`RISKS: ${risks.map(r => r.description).join('; ')}`);
        }

        const contextText = `[SPATIAL STATE UPDATE] ${parts.join(' | ')}`;

        // Only send if context actually changed
        if (contextText === this.spatialContext) return;
        this.spatialContext = contextText;

        // Send as client content (text alongside audio)
        this.ws.send(JSON.stringify({
            client_content: {
                turns: [{
                    role: "user",
                    parts: [{ text: contextText }]
                }],
                turn_complete: false
            }
        }));
    }

    async playNextAudio() {
        if (this.playbackQueue.length === 0) {
            this.isPlaying = false;
            return;
        }
        this.isPlaying = true;

        const audioData = this.playbackQueue.shift();
        try {
            // Gemini returns 24kHz 16-bit PCM
            const playbackCtx = new AudioContext({ sampleRate: 24000 });
            const int16 = new Int16Array(audioData);
            const float32 = new Float32Array(int16.length);
            for (let i = 0; i < int16.length; i++) {
                float32[i] = int16[i] / 32768.0;
            }

            const buffer = playbackCtx.createBuffer(1, float32.length, 24000);
            buffer.getChannelData(0).set(float32);

            const source = playbackCtx.createBufferSource();
            source.buffer = buffer;
            source.connect(playbackCtx.destination);
            source.onended = () => {
                playbackCtx.close();
                this.playNextAudio();
            };
            source.start();
        } catch (err) {
            console.error('[Voice] Playback error:', err);
            this.isPlaying = false;
        }
    }

    // ── Utility functions ───────────────────────────────
    float32ToInt16(float32) {
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
            const s = Math.max(-1, Math.min(1, float32[i]));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16;
    }

    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    base64ToArrayBuffer(base64) {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes.buffer;
    }

    disconnect() {
        this.stopListening();
        if (this.ws) this.ws.close();
        this.isConnected = false;
    }
}


// ═══════════════════════════════════════════════════════════════════════
// App — Main orchestrator
// ═══════════════════════════════════════════════════════════════════════

class App {
    constructor() {
        this.video = new VideoStream();
        this.overlay = new SpatialOverlay();
        this.voice = new GeminiVoice();
        this.voiceEnabled = false;
        this.spatialUpdateInterval = null;

        this.bindUI();
    }

    bindUI() {
        document.getElementById('start-btn').addEventListener('click', () => this.start());
        document.getElementById('btn-voice').addEventListener('click', () => this.toggleVoice());
        document.getElementById('btn-snapshot').addEventListener('click', () => this.takeSnapshot());
        document.getElementById('btn-flip').addEventListener('click', () => this.flipCamera());
    }

    async start() {
        // Start camera
        const cameraOk = await this.video.startCamera();
        if (!cameraOk) {
            this.toast('Camera access denied');
            return;
        }

        // Update UI
        document.getElementById('start-screen').classList.add('hidden');
        document.getElementById('camera-container').classList.remove('hidden');
        this.setDot('camera', 'active');

        // Connect video WebSocket
        this.video.onStateUpdate = (state) => {
            this.overlay.update(state);
            this.updateSummary(state);
            this.updateFPS(state);
        };
        this.video.connectWebSocket();
        this.setDot('server', 'pending');

        // Wait for server connection
        setTimeout(() => {
            if (this.video.ws?.readyState === WebSocket.OPEN) {
                this.setDot('server', 'active');
                this.toast('Connected to AEGIS');
            }
        }, 2000);

        // Init Gemini Voice (but don't start listening yet)
        const voiceOk = await this.voice.init();
        if (voiceOk) {
            this.setDot('voice', 'inactive');
        } else {
            this.setDot('voice', 'inactive');
            console.warn('Gemini voice not available');
        }
    }

    async toggleVoice() {
        const btn = document.getElementById('btn-voice');
        
        if (!this.voiceEnabled) {
            // Start voice
            this.toast('Connecting to voice...');
            const ok = await this.voice.connect();
            if (!ok) {
                this.toast('Voice connection failed');
                return;
            }

            await this.voice.startListening();
            this.voiceEnabled = true;
            btn.classList.add('recording');
            this.setDot('voice', 'active');
            this.toast('Voice active — speak to AEGIS');

            // Start sending spatial context to Gemini
            this.spatialUpdateInterval = setInterval(() => {
                if (this.overlay.lastState) {
                    this.voice.updateSpatialContext(this.overlay.lastState);
                }
            }, 3000);
        } else {
            // Stop voice
            this.voice.stopListening();
            this.voice.disconnect();
            this.voiceEnabled = false;
            btn.classList.remove('recording');
            this.setDot('voice', 'inactive');
            this.toast('Voice stopped');

            if (this.spatialUpdateInterval) {
                clearInterval(this.spatialUpdateInterval);
                this.spatialUpdateInterval = null;
            }
        }
    }

    takeSnapshot() {
        // Flash effect
        const canvas = document.getElementById('overlay-canvas');
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        setTimeout(() => this.overlay.draw(), 200);
        this.toast('Snapshot captured');
    }

    async flipCamera() {
        await this.video.flipCamera();
        this.toast('Camera flipped');
    }

    updateSummary(state) {
        const el = document.getElementById('spatial-summary');
        const parts = [];

        const persons = state.persons || [];
        if (persons.length === 0) {
            parts.push('No one in view');
        } else {
            const descs = persons.map(p => {
                const activity = p.activity || 'detected';
                return `Person ${p.track_id}: ${activity}`;
            });
            parts.push(descs.join(' · '));
        }

        const objects = state.objects || [];
        if (objects.length > 0) {
            const counts = {};
            objects.forEach(o => { counts[o.class_name] = (counts[o.class_name] || 0) + 1; });
            const objStr = Object.entries(counts).slice(0, 4).map(([k,v]) => `${v} ${k}`).join(', ');
            parts.push(objStr);
        }

        const risks = state.risk_events || [];
        if (risks.length > 0) {
            parts.push(`⚠ ${risks[0].description}`);
        }

        el.textContent = parts.join(' · ');
    }

    updateFPS(state) {
        const el = document.getElementById('fps-display');
        el.textContent = `${Math.round(state.fps || 0)} FPS`;
    }

    setDot(name, status) {
        const dot = document.getElementById(`dot-${name}`);
        dot.className = `status-dot ${status}`;
    }

    toast(msg) {
        const el = document.getElementById('toast');
        el.textContent = msg;
        el.classList.add('show');
        setTimeout(() => el.classList.remove('show'), 2500);
    }
}

// ── Initialize ──────────────────────────────────────────────────────
const app = new App();
