'use client';

import { useState } from 'react';
import { Box, Code2, Sparkles, Maximize2, Minimize2, ExternalLink, Activity } from 'lucide-react';
import confetti from 'canvas-confetti';

interface NavbarProps {
  onOpenCodeModal: () => void;
}

export default function Navbar({ onOpenCodeModal }: NavbarProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleToggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
      }
    }
  };

  const handleLaunchConfetti = () => {
    confetti({
      particleCount: 80,
      spread: 70,
      origin: { y: 0.2 },
      colors: ['#06b6d4', '#a855f7', '#10b981', '#ec4899']
    });
  };

  return (
    <header className="sticky top-4 z-40 w-full max-w-7xl mx-auto px-4">
      <div className="glass-hud rounded-2xl px-5 py-3 flex items-center justify-between border border-cyan-500/20 backdrop-blur-xl">
        {/* Left Brand Badge */}
        <div className="flex items-center space-x-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 to-purple-500/20 border border-cyan-400/30 text-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.3)]">
            <Box className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-extrabold text-base tracking-wider text-slate-100 uppercase">
                Spline<span className="text-cyan-400">3D</span> Studio
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-mono uppercase bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 rounded-full font-semibold">
                React v19
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>ThreeJS WebGL Engine</span>
            </p>
          </div>
        </div>

        {/* Middle Telemetry Badge */}
        <div className="hidden md:flex items-center space-x-3 px-4 py-1.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs font-mono text-slate-300">
          <Activity className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
          <span>SCENE: <span className="text-cyan-300">r5KAc7jXVA7ryXus</span></span>
          <span className="text-slate-600">|</span>
          <span className="text-emerald-400 flex items-center space-x-1">
            <span>● ACTIVE</span>
          </span>
        </div>

        {/* Right Actions */}
        <div className="flex items-center space-x-2">
          {/* Confetti Trigger */}
          <button
            onClick={handleLaunchConfetti}
            title="Launch Cyber FX"
            className="flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 text-purple-300 text-xs font-mono transition-all cursor-pointer hover:shadow-[0_0_12px_rgba(168,85,247,0.3)]"
          >
            <Sparkles className="w-4 h-4 text-purple-400" />
            <span className="hidden sm:inline">FX Burst</span>
          </button>

          {/* View React Code Button */}
          <button
            onClick={onOpenCodeModal}
            className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-gradient-to-r from-cyan-500/20 to-purple-500/20 hover:from-cyan-500/30 hover:to-purple-500/30 border border-cyan-400/40 text-cyan-300 text-xs font-mono font-medium transition-all shadow-[0_0_15px_rgba(6,182,212,0.2)] cursor-pointer"
          >
            <Code2 className="w-4 h-4 text-cyan-400" />
            <span>Get Code</span>
          </button>

          {/* Fullscreen Toggle */}
          <button
            onClick={handleToggleFullscreen}
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen View"}
            className="p-2 rounded-xl bg-slate-800/60 hover:bg-slate-700/80 border border-slate-700 text-slate-300 transition-all cursor-pointer"
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>

          {/* Official Spline Docs */}
          <a
            href="https://spline.design"
            target="_blank"
            rel="noopener noreferrer"
            title="Open Spline.design"
            className="p-2 rounded-xl bg-slate-800/60 hover:bg-slate-700/80 border border-slate-700 text-slate-300 transition-all hidden sm:block"
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>
      </div>
    </header>
  );
}
