'use client';

import { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import RobotControlsPanel from '@/components/RobotControlsPanel';
import CodeExportModal from '@/components/CodeExportModal';
import { JointState } from '@/utils/robotKinematics';
import { Box, Code2, Sparkles, Maximize2, Minimize2, Video, Square, Play, Pause, RotateCcw } from 'lucide-react';
import confetti from 'canvas-confetti';

const ThreeRobotArmCanvas = dynamic(() => import('@/components/ThreeRobotArmCanvas'), { ssr: false });

export default function Home() {
  const [isCodeModalOpen, setIsCodeModalOpen] = useState(false);
  const [isControlsOpen, setIsControlsOpen] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const [joints, setJoints] = useState<JointState>({
    base: 0,
    shoulder: (28 * Math.PI) / 180,
    elbow: (-50 * Math.PI) / 180,
    wrist: (20 * Math.PI) / 180,
    gripper: 0.35,
  });

  const [isIkMode, setIsIkMode] = useState(false);
  const [isWebcamMode, setIsWebcamMode] = useState(false);
  const [isCarryingCargo, setIsCarryingCargo] = useState(false);

  // Recording & Playback States
  const [isRecording, setIsRecording] = useState(false);
  const [isPlayingBack, setIsPlayingBack] = useState(false);
  const [hasRecordedMotion, setHasRecordedMotion] = useState(false);

  const recordedJointSequence = useRef<JointState[]>([]);
  const isRecordingRef = useRef(false);
  isRecordingRef.current = isRecording;

  // Poll Python API via Next.js Proxy for Webcam Pose/Hand Tracking
  useEffect(() => {
    if (!isWebcamMode || isPlayingBack) return;
    
    let active = true;
    let prevJoints: JointState | null = null;

    const poll = async () => {
      try {
        const res = await fetch('/api/state');
        if (!res.ok) throw new Error('API error');
        const data = await res.json();
        
        if (active && data.angles && data.angles.length === 5) {
          const a = data.angles;
          
          const rawBase = ((a[0] - 100) / 100) * (90 * Math.PI / 180); 
          const rawShoulder = ((a[1] - 105) / 105) * (70 * Math.PI / 180);
          const rawElbow = ((a[2] - 75) / 75) * (70 * Math.PI / 180);
          const rawWrist = ((a[3] - 65) / 55) * (90 * Math.PI / 180);
          const rawGripper = Math.max(0, Math.min(1, a[4] / 60));
          
          let newJoints: JointState;
          if (prevJoints) {
            const alpha = 0.35;
            newJoints = {
              base: prevJoints.base + (rawBase - prevJoints.base) * alpha,
              shoulder: prevJoints.shoulder + (rawShoulder - prevJoints.shoulder) * alpha,
              elbow: prevJoints.elbow + (rawElbow - prevJoints.elbow) * alpha,
              wrist: prevJoints.wrist + (rawWrist - prevJoints.wrist) * alpha,
              gripper: prevJoints.gripper + (rawGripper - prevJoints.gripper) * alpha,
            };
          } else {
            newJoints = { base: rawBase, shoulder: rawShoulder, elbow: rawElbow, wrist: rawWrist, gripper: rawGripper };
          }

          prevJoints = newJoints;
          setJoints(newJoints);

          // If recording is active, capture joint state trajectory
          if (isRecordingRef.current) {
            recordedJointSequence.current.push(newJoints);
          }
        }
      } catch (err) {
        // Fail silently if backend is unavailable
      }
      
      if (active) setTimeout(poll, 30);
    };
    
    poll();
    return () => { active = false; };
  }, [isWebcamMode, isPlayingBack]);

  // Motion Playback Loop
  useEffect(() => {
    if (!isPlayingBack || recordedJointSequence.current.length === 0) return;

    let active = true;
    let frameIdx = 0;

    const playNextFrame = () => {
      if (!active) return;

      if (frameIdx < recordedJointSequence.current.length) {
        setJoints(recordedJointSequence.current[frameIdx]);
        frameIdx++;
        setTimeout(playNextFrame, 33); // ~30 FPS playback
      } else {
        // Replay completed
        setIsPlayingBack(false);
      }
    };

    playNextFrame();
    return () => { active = false; };
  }, [isPlayingBack]);

  // Handle Recording Toggle
  const handleToggleRecording = async () => {
    if (!isRecording) {
      // Start Recording
      try {
        await fetch('/api/recording/start', { method: 'POST' });
      } catch (e) {
        // ignore endpoint error if backend is local only
      }
      recordedJointSequence.current = [];
      setIsRecording(true);
      setIsPlayingBack(false);
    } else {
      // Stop Recording
      try {
        await fetch('/api/recording/stop', { method: 'POST' });
      } catch (e) {
        // ignore
      }
      setIsRecording(false);
      if (recordedJointSequence.current.length > 0) {
        setHasRecordedMotion(true);
      }
    }
  };

  // Handle Playback Toggle
  const handleTogglePlayback = () => {
    if (isPlayingBack) {
      setIsPlayingBack(false);
    } else if (recordedJointSequence.current.length > 0) {
      setIsPlayingBack(true);
    }
  };

  const handleToggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  const handleFxBurst = () => {
    confetti({ particleCount: 80, spread: 70, origin: { y: 0.3 }, colors: ['#94a3b8', '#cbd5e1', '#e2e8f0', '#06b6d4', '#a855f7'] });
  };

  return (
    <main className="fixed inset-0 overflow-hidden bg-[#030712] flex">
      {/* 
        ========================================================================
        Left Panel: Live Video Feed (Only visible when Webcam mode is ON) 
        ========================================================================
      */}
      {isWebcamMode && (
        <div className="w-[45%] lg:w-[40%] h-full relative bg-[#010308] border-r border-slate-800 flex-shrink-0 shadow-[4px_0_24px_rgba(0,0,0,0.8)] z-40 flex items-center justify-center">
          <div className="absolute top-4 left-4 z-30 flex items-center space-x-2.5 px-4 py-2.5 glass-hud rounded-2xl border border-pink-500/30">
            <span className="w-2.5 h-2.5 rounded-full bg-pink-500 animate-pulse shadow-[0_0_8px_#ec4899]" />
            <span className="font-bold text-xs text-pink-100 tracking-wider uppercase">Live Tracking</span>
          </div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img 
            src="/api/video_feed" 
            alt="Webcam Feed" 
            className="w-full h-full object-contain opacity-90 p-4"
          />
        </div>
      )}

      {/* 
        ========================================================================
        Right Panel: 3D Robot Arm Canvas
        ========================================================================
      */}
      <div className="flex-1 h-full relative">
        <ThreeRobotArmCanvas
          joints={joints}
          onJointsChange={setJoints}
          isIkMode={isIkMode}
          activeColorPreset="cyan"
          isCarryingCargo={isCarryingCargo}
          onCargoStateChange={setIsCarryingCargo}
        />

        {/* Top-Left Brand Badge */}
        <div className="absolute top-4 left-4 z-30 flex items-center space-x-2.5 px-4 py-2.5 glass-hud rounded-2xl border border-slate-700/60">
          <Box className="w-4 h-4 text-cyan-400" />
          <span className="font-bold text-sm text-slate-100 tracking-wider uppercase">
            Robot<span className="text-cyan-400">Arm</span>
          </span>
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        </div>

        {/* Top-Right Action Bar with Recording & Motion Playback */}
        <div className="absolute top-4 right-4 z-30 flex items-center space-x-2">
          {/* Start/Stop Recording Button */}
          <button
            onClick={handleToggleRecording}
            className={`flex items-center space-x-2 px-3.5 py-2.5 rounded-xl border text-xs font-mono font-bold tracking-wider uppercase transition-all cursor-pointer shadow-lg ${
              isRecording
                ? 'bg-red-500/20 border-red-500 text-red-300 shadow-[0_0_15px_rgba(239,68,68,0.4)] animate-pulse'
                : 'glass-hud border-slate-700/70 text-red-400 hover:bg-red-500/10 hover:border-red-500/40'
            }`}
            title={isRecording ? 'Stop Recording Video & Motion' : 'Start Recording Video & Motion'}
          >
            {isRecording ? (
              <>
                <Square className="w-3.5 h-3.5 fill-red-500 text-red-500" />
                <span>Stop REC</span>
              </>
            ) : (
              <>
                <Video className="w-4 h-4 text-red-400" />
                <span>Start REC</span>
              </>
            )}
          </button>

          {/* Replay Motion Button (Visible when motion sequence is available) */}
          {hasRecordedMotion && (
            <button
              onClick={handleTogglePlayback}
              className={`flex items-center space-x-2 px-3.5 py-2.5 rounded-xl border text-xs font-mono font-bold tracking-wider uppercase transition-all cursor-pointer shadow-lg ${
                isPlayingBack
                  ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-[0_0_15px_rgba(6,182,212,0.4)]'
                  : 'glass-hud border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/20'
              }`}
              title={isPlayingBack ? 'Pause Replay' : 'Replay Recorded Motion'}
            >
              {isPlayingBack ? (
                <>
                  <Pause className="w-3.5 h-3.5 text-cyan-300 fill-cyan-300" />
                  <span>Playing...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 text-cyan-300 fill-cyan-300" />
                  <span>Replay Motion</span>
                </>
              )}
            </button>
          )}

          <button
            onClick={handleFxBurst}
            className="p-2.5 glass-hud rounded-xl border border-purple-500/30 text-purple-300 hover:bg-purple-500/20 transition-all cursor-pointer"
            title="FX Burst"
          >
            <Sparkles className="w-4 h-4" />
          </button>
          <button
            onClick={() => setIsCodeModalOpen(true)}
            className="p-2.5 glass-hud rounded-xl border border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/20 transition-all cursor-pointer"
            title="Get Code"
          >
            <Code2 className="w-4 h-4" />
          </button>
          <button
            onClick={handleToggleFullscreen}
            className="p-2.5 glass-hud rounded-xl border border-slate-700/60 text-slate-300 hover:bg-slate-700/40 transition-all cursor-pointer"
            title="Fullscreen"
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>

        {/* Bottom Controls Toggle Button */}
        <div className="absolute bottom-5 right-5 z-30">
          <button
            onClick={() => setIsControlsOpen(!isControlsOpen)}
            className={`flex items-center space-x-2 px-4 py-2.5 rounded-2xl border text-xs font-mono font-semibold uppercase tracking-wider transition-all cursor-pointer shadow-xl ${
              isControlsOpen
                ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-[0_0_20px_rgba(6,182,212,0.3)]'
                : 'glass-hud border-slate-700/60 text-slate-300 hover:border-cyan-500/40'
            }`}
          >
            <span>{isControlsOpen ? '✕ Close Controls' : '⚙ Open Controls'}</span>
          </button>
        </div>

        {/* Floating Controls Panel (slides up from bottom) */}
        <div
          className={`absolute bottom-0 inset-x-0 z-20 px-4 pb-4 transition-all duration-500 ${
            isControlsOpen ? 'translate-y-0 opacity-100' : 'translate-y-full opacity-0 pointer-events-none'
          }`}
        >
          <RobotControlsPanel
            joints={joints}
            onJointsChange={setJoints}
            isIkMode={isIkMode}
            onToggleIkMode={setIsIkMode}
            isWebcamMode={isWebcamMode}
            onToggleWebcamMode={setIsWebcamMode}
            isCarryingCargo={isCarryingCargo}
            onToggleCargo={() => setIsCarryingCargo(!isCarryingCargo)}
          />
        </div>
      </div>

      {/* Code Export Modal */}
      <CodeExportModal isOpen={isCodeModalOpen} onClose={() => setIsCodeModalOpen(false)} />
    </main>
  );
}
