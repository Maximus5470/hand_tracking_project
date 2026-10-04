'use client';

import { Box, Layers, Cpu, Code2, Zap, Sparkles } from 'lucide-react';

export default function FeatureSection() {
  const features = [
    {
      icon: Box,
      title: "Spline 3D Scene Integration",
      description: "Direct WebGL rendering of high-fidelity 3D assets loaded directly from Spline's cloud pipeline with real-time mouse orbit controls.",
      color: "cyan",
      badge: "Spline 3D"
    },
    {
      icon: Layers,
      title: "Next.js App Router Native",
      description: "Uses `@splinetool/react-spline/next` for optimized streaming, layout stability, and seamless React server/client hydration.",
      color: "purple",
      badge: "Next.js 14+"
    },
    {
      icon: Cpu,
      title: "ThreeJS WebGL Acceleration",
      description: "Hardware accelerated graphics pipeline rendering physical materials, dynamic lighting, and custom particle shaders smoothly at 60 FPS.",
      color: "emerald",
      badge: "WebGL 2.0"
    },
    {
      icon: Code2,
      title: "Event Listener & Controls API",
      description: "Hooks into Spline object events (`onLoad`, `onMouseDown`, `onFollow`) to enable custom HUD controls and camera positions.",
      color: "pink",
      badge: "Interactive"
    }
  ];

  return (
    <section className="w-full max-w-7xl mx-auto px-4 py-16">
      <div className="text-center max-w-3xl mx-auto mb-12">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-mono font-semibold uppercase mb-4">
          <Zap className="w-3.5 h-3.5 animate-pulse" />
          <span>Interactive 3D Capabilities</span>
        </div>
        <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight text-slate-100 mb-4">
          Powered by Next.js & <span className="text-gradient-cyan">Spline ThreeJS</span>
        </h2>
        <p className="text-slate-400 text-sm md:text-base font-light">
          Transform your web application with immersive 3D graphics, interactive UI controls, and reactive glassmorphism design.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {features.map((feature, idx) => {
          const Icon = feature.icon;
          return (
            <div
              key={idx}
              className="glass-panel glass-panel-hover p-6 rounded-2xl flex flex-col justify-between relative overflow-hidden group"
            >
              {/* Subtle Corner Ambient Glow */}
              <div className="absolute top-0 right-0 w-24 h-24 bg-cyan-500/5 rounded-full blur-2xl group-hover:bg-cyan-500/15 transition-all" />

              <div>
                <div className="flex items-center justify-between mb-4">
                  <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-cyan-400 group-hover:border-cyan-500/40 group-hover:text-cyan-300 transition-all">
                    <Icon className="w-6 h-6" />
                  </div>
                  <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                    {feature.badge}
                  </span>
                </div>

                <h3 className="text-lg font-bold text-slate-100 mb-2 group-hover:text-cyan-300 transition-colors">
                  {feature.title}
                </h3>
                
                <p className="text-xs text-slate-400 leading-relaxed font-light">
                  {feature.description}
                </p>
              </div>

              <div className="mt-6 pt-4 border-t border-slate-800/60 flex items-center text-[11px] font-mono text-cyan-400 font-medium">
                <span>View Documentation</span>
                <span className="ml-1 group-hover:translate-x-1 transition-transform">→</span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
