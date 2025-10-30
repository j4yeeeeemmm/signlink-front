import React, { useEffect, useRef, useState } from "react";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const holRef = useRef(null);
  const animationRef = useRef(null);
  const wsRef = useRef(null);

  const frameBufferRef = useRef([]);
  const smoothBufferRef = useRef([]);
  const capturingRef = useRef(false);
  const frameCountRef = useRef(0);

  const predictionHistoryRef = useRef([]);
  const MAX_HISTORY = 5;

  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState(null);
  const [cameraStarted, setCameraStarted] = useState(false);

  const [prediction, setPrediction] = useState("...");
  const [sentence, setSentence] = useState("");
  const [frameCountDisplay, setFrameCountDisplay] = useState(0);

  const T = 60;
  const INTERNAL_W = 320;
  const INTERNAL_H = 240;
  const SMOOTH_FRAMES = 5;

  const lmArr = (lm) => {
    if (!lm) return null;
    return Array.isArray(lm) ? lm : lm.landmark ? lm.landmark : null;
  };

  const enumerateCameras = async () => {
    try {
      const deviceInfos = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = deviceInfos.filter((d) => d.kind === "videoinput");
      setDevices(videoDevices);
      if (videoDevices.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevices[0].deviceId);
      }
    } catch (err) {
      console.error("Error enumerating devices:", err);
    }
  };
  useEffect(() => { enumerateCameras(); }, []);

  useEffect(() => {
    const socket = new WebSocket("wss://localhost:8000/ws");
    wsRef.current = socket;

    socket.onopen = () => console.log("WebSocket connected ✅");

    socket.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.prediction) {
          const word = msg.prediction;
          if (word !== "nothing") {
            setPrediction(word);

            predictionHistoryRef.current.push(word);
            if (predictionHistoryRef.current.length > MAX_HISTORY) {
              predictionHistoryRef.current.shift();
            }

            const recent = predictionHistoryRef.current;
            if (!recent.slice(0, -1).includes(word)) {
              setSentence((prev) => prev ? prev + " " + word : word);
            }
          }
        }
      } catch (err) {
        console.warn("WS parse error:", err);
      }
    };

    socket.onerror = (err) => console.error("WebSocket error:", err);

    return () => { try { socket.close(); } catch {} wsRef.current = null; };
  }, []);

  const requestCameraAccess = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: true });
      s.getTracks().forEach((t) => t.stop());
      await enumerateCameras();
      alert("✅ Camera access granted. Select a device and click Start.");
    } catch (err) {
      console.error("Camera access denied", err);
      alert("❌ Camera access denied.");
    }
  };

  const startCamera = async () => {
    try {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      }
      if (!selectedDeviceId) {
        alert("No camera selected");
        return;
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: selectedDeviceId }, width: INTERNAL_W, height: INTERNAL_H },
        audio: false,
      });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCameraStarted(true);
      capturingRef.current = true;
      frameBufferRef.current = [];
      smoothBufferRef.current = [];
      frameCountRef.current = 0;
      setFrameCountDisplay(0);
    } catch (err) {
      console.error("Failed to start camera:", err);
      alert("Could not start camera.");
    }
  };

  const stopCamera = () => {
    try { if (videoRef.current && videoRef.current.srcObject) videoRef.current.srcObject.getTracks().forEach((t) => t.stop()); } catch {}
    setCameraStarted(false);
    capturingRef.current = false;
  };

  const getSmoothed = (landmarks) => {
    if (!landmarks) return Array(33*4 + 21*3*2).fill(0);
    smoothBufferRef.current.push(landmarks);
    if (smoothBufferRef.current.length > SMOOTH_FRAMES) smoothBufferRef.current.shift();

    const N = landmarks.length;
    const avg = new Array(N).fill(0);
    smoothBufferRef.current.forEach((frame) => {
      for (let i = 0; i < N; i++) avg[i] += frame[i];
    });
    return avg.map((v) => v / smoothBufferRef.current.length);
  };

  useEffect(() => {
    if (!cameraStarted) return;
    if (!window.Holistic || !window.drawConnectors || !window.drawLandmarks) {
      alert("Mediapipe not loaded (Holistic / drawConnectors missing).");
      return;
    }

    const holistic = new window.Holistic({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`,
    });
    holRef.current = holistic;

    holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.3, minTrackingConfidence: 0.3 });

    holistic.onResults((results) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      try { ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height); } catch {}

      const pose = lmArr(results.poseLandmarks);
      const left = lmArr(results.leftHandLandmarks);
      const right = lmArr(results.rightHandLandmarks);

      if (pose) window.drawConnectors(ctx, pose, window.POSE_CONNECTIONS, { color: "#00FF00", lineWidth: 2 });
      if (left) window.drawConnectors(ctx, left, window.HAND_CONNECTIONS, { color: "#FF3030", lineWidth: 2 });
      if (right) window.drawConnectors(ctx, right, window.HAND_CONNECTIONS, { color: "#3030FF", lineWidth: 2 });

      if (capturingRef.current) {
        const flatten = (lm, fill, dim) => {
          if (!lm) return Array(fill*dim).fill(0);
          return lm.flatMap(p => dim===4 ? [p.x,p.y,p.z,p.visibility??0] : [p.x,p.y,p.z]);
        };

        const raw = [...flatten(pose,33,4), ...flatten(left,21,3), ...flatten(right,21,3)];
        const smoothed = getSmoothed(raw);

        frameBufferRef.current.push(smoothed);
        frameCountRef.current = frameBufferRef.current.length;
        setFrameCountDisplay(frameCountRef.current);

        if (frameBufferRef.current.length >= T) {
          try {
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ type: "frame_sequence", landmarks: frameBufferRef.current }));
            }
          } catch (e) { console.error("WS send error:", e); }
          frameBufferRef.current = [];
          frameCountRef.current = 0;
          setFrameCountDisplay(0);
        }
      }
    });

    const loop = async () => {
      try { if (videoRef.current && videoRef.current.readyState >= 2) await holistic.send({ image: videoRef.current }); } catch {}
      animationRef.current = requestAnimationFrame(loop);
    };
    animationRef.current = requestAnimationFrame(loop);

    return () => { try { if (animationRef.current) cancelAnimationFrame(animationRef.current); } catch {} try { holistic.close?.(); } catch {} holRef.current = null; };
  }, [cameraStarted]);

  // 🔊 Speak Filipino sentence
  const speakSentence = () => {
    if (!sentence) {
      alert("Walang pangungusap na mabibigkas!");
      return;
    }
    const utterance = new SpeechSynthesisUtterance(sentence);
    utterance.lang = "fil-PH"; // Filipino
    utterance.rate = 1; // speed
    utterance.pitch = 2; // tone
    speechSynthesis.speak(utterance);
  };

  return (
    <div style={{ display: "flex", justifyContent: "center", padding: 16 }}>
      <div className="container">
        <h2 className="title">Sign Language Translation</h2>
        <div className="controls">
          <button className="btn" onClick={requestCameraAccess}>Request Camera</button>
          <select className="select" value={selectedDeviceId || ""} onChange={(e) => setSelectedDeviceId(e.target.value)}>
            {devices.map(d => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${d.deviceId}`}</option>)}
          </select>
          <button className="btn" onClick={startCamera}>Start</button>
          <button className="btn danger" onClick={stopCamera}>Stop</button>
        </div>
        <div className="content">
          <div className="videoBox">
            <video ref={videoRef} style={{ display: "none" }} playsInline muted />
            <canvas ref={canvasRef} width={INTERNAL_W} height={INTERNAL_H} className="canvas"/>
            <div className="progressBar"><div className="progressFill" style={{ width: `${(frameCountDisplay/T)*100}%` }}/></div>
            <div className="info"><div><strong>Frame:</strong> {frameCountDisplay}/{T}</div><div><strong>Prediction:</strong> {prediction}</div></div>
          </div>
          <div className="textBox">
            <h4 className="subtitle">Formed Sentence</h4>
            <div className="sentenceBox">{sentence || "Wala pang pangungusap..."}</div>
            <div className="btnRow">
              <button className="btn secondary" onClick={() => { setSentence(""); setPrediction("..."); predictionHistoryRef.current = []; }}>Clear</button>
              <button className="btn success" onClick={speakSentence}>Speak</button>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .container { width: 100%; max-width: 700px; }
        .title { font-size: 1.4rem; margin-bottom: 12px; text-align: center; }
        .controls { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; justify-content: center; }
        .btn { padding: 10px 16px; font-size: 1rem; border: none; border-radius: 6px; background: #007bff; color: #fff; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .btn.danger { background: #dc3545; }
        .btn.secondary { background: #6c757d; }
        .btn.success { background: #28a745; }
        .select { padding: 10px; font-size: 1rem; border-radius: 6px; border: 1px solid #ccc; }
        .content { display: flex; flex-direction: row; gap: 16px; flex-wrap: wrap; }
        .videoBox, .textBox { flex: 1; min-width: 280px; }
        .canvas { width: 100%; height: auto; border: 1px solid #ccc; background: #000; border-radius: 6px; }
        .progressBar { width: 100%; background: #e6e6e6; height: 12px; margin-top: 8px; border-radius: 6px; overflow: hidden; }
        .progressFill { height: 100%; background: limegreen; transition: width 0.15s linear; }
        .info { margin-top: 6px; font-size: 0.95rem; }
        .subtitle { font-size: 1.1rem; margin-bottom: 8px; }
        .sentenceBox { border: 1px solid #aaa; padding: 12px; min-height: 120px; border-radius: 6px; background: #fff; font-size: 1rem; }
        .btnRow { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
        @media (max-width: 768px) { .content { flex-direction: column; } .btn, .select { width: 100%; font-size: 1.1rem; padding: 12px; } }
      `}</style>
    </div>
  );
}
