'use client';

import { useState, useEffect } from 'react';
import { Cpu, Gauge, Radio, ShieldCheck } from 'lucide-react';

export default function TelemetryOverlay() {
  const [fps, setFps] = useState(60);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    // Dynamic slight FPS jitter simulation for hyper-realistic HUD feel
    const interval = setInterval(() => {
      setFps(Math.floor(58 + Math.random() * 3));
    }, 1200);

    const handleMouseMove = (e: MouseEvent) => {
      setMousePos({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      clearInterval(interval);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return (
    <div className="absolute top-4 right-4 z-30 pointer-events-none hidden lg:flex flex-col space-y-2">
      <div className="glass-hud rounded-xl p-3 border border-cyan-500/20 text-[11px] font-mono text-slate-300 w-52 backdrop-blur-md shadow-lg">
        <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800/80">
          <div className="flex items-center space-x-1.5 text-cyan-400 font-bold tracking-wider">
            <Radio className="w-3.5 h-3.5 animate-pulse text-cyan-400" />
            <span>TELEMETRY HUD</span>
          </div>
          <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[9px]">LIVE</span>
        </div>

        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <span className="text-slate-400 flex items-center space-x-1">
              <Gauge className="w-3 h-3 text-purple-400" />
              <span>FPS Counter:</span>
            </span>
            <span className="text-emerald-400 font-bold">{fps} FPS</span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-slate-400 flex items-center space-x-1">
              <Cpu className="w-3 h-3 text-cyan-400" />
              <span>Renderer:</span>
            </span>
            <span className="text-cyan-300">WebGL 2.0</span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-slate-400">Pointer (X, Y):</span>
            <span className="text-purple-300 font-mono">
              {mousePos.x.toString().padStart(4, '0')}, {mousePos.y.toString().padStart(4, '0')}
            </span>
          </div>

          <div className="flex justify-between items-center pt-1 border-t border-slate-800/50">
            <span className="text-slate-400 flex items-center space-x-1">
              <ShieldCheck className="w-3 h-3 text-emerald-400" />
              <span>Shadow Shader:</span>
            </span>
            <span className="text-slate-300">PBR Ultra</span>
          </div>
        </div>
      </div>
    </div>
  );
}
