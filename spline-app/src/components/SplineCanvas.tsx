'use client';

import { useState, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Loader2, AlertCircle, RefreshCw, Sparkles } from 'lucide-react';
import type { Application } from '@splinetool/runtime';

// Dynamic import with SSR disabled to prevent WebGL hydration mismatches in React 19
const Spline = dynamic(() => import('@splinetool/react-spline'), {
  ssr: false,
});

interface SplineCanvasProps {
  sceneUrl?: string;
  onSplineLoad?: (splineApp: Application) => void;
  className?: string;
}

export default function SplineCanvas({
  sceneUrl = "https://prod.spline.design/r5KAc7jXVA7ryXus/scene.splinecode",
  onSplineLoad,
  className = ""
}: SplineCanvasProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const splineRef = useRef<Application | null>(null);

  const handleLoad = useCallback((splineApp: Application) => {
    splineRef.current = splineApp;
    setIsLoading(false);

    try {
      // Inspect scene objects for Robot Arm isolation
      const objects = (splineApp as any).getAllObjects ? (splineApp as any).getAllObjects() : [];
      console.log('📦 Spline Scene Hierarchy Objects:', objects.map((o: any) => ({ id: o.id, name: o.name, type: o.type })));
    } catch (e) {
      console.log('Spline object enumeration:', e);
    }

    if (onSplineLoad) {
      onSplineLoad(splineApp);
    }
  }, [onSplineLoad]);

  const handleError = useCallback(() => {
    setIsLoading(false);
    setHasError(true);
  }, []);

  const handleRetry = () => {
    setHasError(false);
    setIsLoading(true);
  };

  return (
    <div className={`relative w-full h-full min-h-[500px] md:min-h-[650px] rounded-2xl overflow-hidden glass-panel ${className}`}>
      {/* Scanline Effect Overlay */}
      <div className="scanline-effect" />

      {/* Cyber Grid Background */}
      <div className="absolute inset-0 cyber-grid-bg opacity-30 pointer-events-none" />

      {/* Radial Lights */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 radial-glow-cyan pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 radial-glow-purple pointer-events-none" />

      {/* Spline 3D Scene */}
      {!hasError && (
        <Spline
          scene={sceneUrl}
          onLoad={handleLoad}
          onError={handleError}
          className="w-full h-full object-cover transition-opacity duration-700 ease-out"
          style={{ opacity: isLoading ? 0 : 1 }}
        />
      )}

      {/* Loading Skeleton HUD */}
      {isLoading && !hasError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-950/80 backdrop-blur-md z-20 transition-all">
          <div className="relative mb-6">
            {/* Outer Pulsing Ring */}
            <div className="w-20 h-20 rounded-full border-2 border-cyan-500/20 border-t-cyan-400 animate-spin" />
            <div className="absolute inset-0 w-20 h-20 rounded-full border-2 border-purple-500/20 border-b-purple-400 animate-spin [animation-duration:1.8s]" />
            <Sparkles className="absolute inset-0 m-auto w-8 h-8 text-cyan-400 animate-pulse" />
          </div>
          
          <div className="flex items-center space-x-2 text-cyan-400 font-mono text-sm font-medium tracking-widest uppercase mb-2">
            <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
            <span>Initializing 3D Spline Scene</span>
          </div>
          
          <p className="text-xs text-slate-400 font-mono tracking-wider">
            Loading ThreeJS WebGL Engine & Shader Assets...
          </p>

          {/* Loading Progress bar animation */}
          <div className="w-48 h-1 bg-slate-800 rounded-full mt-4 overflow-hidden border border-cyan-500/20">
            <div className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 animate-[pulse_1.5s_infinite]" style={{ width: '80%' }} />
          </div>
        </div>
      )}

      {/* Error Fallback HUD */}
      {hasError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-950/90 backdrop-blur-md z-20 p-6 text-center">
          <div className="p-4 rounded-full bg-rose-500/10 border border-rose-500/30 text-rose-400 mb-4">
            <AlertCircle className="w-10 h-10" />
          </div>
          <h3 className="text-lg font-bold text-slate-100 mb-1">Failed to Load 3D Scene</h3>
          <p className="text-sm text-slate-400 max-w-md mb-6">
            Unable to connect to Spline CDN (`https://prod.spline.design/...`). Please check your network connection or verify scene permissions.
          </p>
          <button
            onClick={handleRetry}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-300 transition-all font-mono text-sm font-medium cursor-pointer"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Retry Connection</span>
          </button>
        </div>
      )}
    </div>
  );
}
