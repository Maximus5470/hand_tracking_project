'use client';

import { Box, Heart, Sparkles } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="w-full border-t border-slate-800/80 bg-slate-950/60 backdrop-blur-lg py-8 mt-12">
      <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-slate-400 font-mono">
        <div className="flex items-center space-x-2">
          <Box className="w-4 h-4 text-cyan-400" />
          <span className="font-bold text-slate-200 uppercase tracking-wider">
            Spline<span className="text-cyan-400">3D</span> Engine
          </span>
          <span>• Interactive WebGL Experience</span>
        </div>

        <div className="flex items-center space-x-4">
          <span className="px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-slate-300">
            Next.js 15
          </span>
          <span className="px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-cyan-400">
            @splinetool/react-spline
          </span>
          <span className="px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-purple-400">
            Three.js Shaders
          </span>
        </div>

        <div className="flex items-center space-x-1">
          <span>Crafted with</span>
          <Heart className="w-3.5 h-3.5 text-rose-500 fill-rose-500" />
          <span>for 3D Web Apps</span>
        </div>
      </div>
    </footer>
  );
}
