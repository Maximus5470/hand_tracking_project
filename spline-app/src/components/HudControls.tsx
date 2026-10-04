'use client';

import { useState } from 'react';
import { MousePointer, RotateCcw, Volume2, VolumeX, Layers, Eye, EyeOff, X, Crosshair } from 'lucide-react';
import type { Application } from '@splinetool/runtime';

interface HudControlsProps {
  splineApp: Application | null;
  onResetCamera?: () => void;
  activeColorPreset: 'cyan' | 'purple' | 'emerald';
  onChangeColorPreset: (preset: 'cyan' | 'purple' | 'emerald') => void;
}

interface SceneObjectItem {
  id: string;
  name: string;
  visible: boolean;
}

export default function HudControls({
  splineApp,
  onResetCamera,
  activeColorPreset,
  onChangeColorPreset
}: HudControlsProps) {
  const [isMuted, setIsMuted] = useState(true);
  const [isArmIsolated, setIsArmIsolated] = useState(false);
  const [showInspector, setShowInspector] = useState(false);
  const [sceneObjects, setSceneObjects] = useState<SceneObjectItem[]>([]);

  const handleToggleSound = () => {
    setIsMuted(!isMuted);
  };

  const handleResetView = () => {
    if (splineApp) {
      try {
        if (typeof (splineApp as any).setZoom === 'function') {
          (splineApp as any).setZoom(1);
        }
      } catch (e) {
        console.log('Spline camera reset:', e);
      }
    }
    if (onResetCamera) onResetCamera();
  };

  // Open the 3D Object Inspector to view and control individual scene objects
  const handleOpenInspector = () => {
    if (splineApp) {
      try {
        const app = splineApp as any;
        const objects = app.getAllObjects ? app.getAllObjects() : [];
        const objectItems: SceneObjectItem[] = objects.map((obj: any) => ({
          id: obj.id || obj.name,
          name: obj.name || 'Unnamed Object',
          visible: obj.visible !== false
        }));
        setSceneObjects(objectItems);
      } catch (e) {
        console.log('Failed to fetch scene objects:', e);
      }
    }
    setShowInspector(true);
  };

  // Toggle visibility of a specific object by ID/Name
  const handleToggleObjectVisibility = (name: string) => {
    if (!splineApp) return;
    try {
      const obj = splineApp.findObjectByName(name);
      if (obj) {
        obj.visible = !obj.visible;
        setSceneObjects(prev =>
          prev.map(item => item.name === name ? { ...item, visible: obj.visible } : item)
        );
      }
    } catch (e) {
      console.log('Error toggling object visibility:', e);
    }
  };

  // Robot Arm Isolation Mode Toggle
  const handleToggleArmIsolation = () => {
    if (!splineApp) return;
    const newIsolatedState = !isArmIsolated;
    setIsArmIsolated(newIsolatedState);

    try {
      const app = splineApp as any;
      const allObjects = app.getAllObjects ? app.getAllObjects() : [];
      
      allObjects.forEach((obj: any) => {
        const lowerName = (obj.name || '').toLowerCase();
        // Check if object is related to the robot/arm/hand or if it's camera/light
        const isRobotComponent = 
          lowerName.includes('arm') || 
          lowerName.includes('robot') || 
          lowerName.includes('hand') ||
          lowerName.includes('joint') ||
          lowerName.includes('finger') ||
          lowerName.includes('claw') ||
          obj.type === 'Camera' || 
          obj.type === 'Light';

        if (newIsolatedState) {
          // Hide non-robot background objects, grounds, walls, text logos
          if (!isRobotComponent && !lowerName.includes('camera') && !lowerName.includes('light')) {
            obj.visible = false;
          } else {
            obj.visible = true;
          }
        } else {
          // Restore all objects visibility
          obj.visible = true;
        }
      });
    } catch (e) {
      console.log('Error isolating robot arm:', e);
    }
  };

  return (
    <>
      <div className="absolute inset-x-4 bottom-4 z-30 flex flex-col md:flex-row items-center justify-between gap-3 pointer-events-none">
        {/* Interaction Hint Badge */}
        <div className="pointer-events-auto flex items-center space-x-2.5 px-4 py-2 rounded-xl glass-hud border border-slate-700/60 text-xs font-mono text-slate-300 shadow-lg">
          <MousePointer className="w-3.5 h-3.5 text-cyan-400 animate-bounce" />
          <span>Click & Drag to Rotate • Scroll to Zoom</span>
        </div>

        {/* Center Controls Bar */}
        <div className="pointer-events-auto flex items-center space-x-2 px-3 py-2 rounded-2xl glass-hud border border-cyan-500/20 shadow-xl">
          {/* Preset Glow Color Selectors */}
          <div className="flex items-center space-x-1.5 pr-2 border-r border-slate-800">
            <button
              onClick={() => onChangeColorPreset('cyan')}
              title="Cyan Glow Mode"
              className={`w-6 h-6 rounded-full border transition-all cursor-pointer ${
                activeColorPreset === 'cyan'
                  ? 'bg-cyan-500 border-cyan-300 ring-2 ring-cyan-500/40 shadow-[0_0_10px_rgba(6,182,212,0.8)]'
                  : 'bg-cyan-950/60 border-cyan-800 hover:border-cyan-500'
              }`}
            />
            <button
              onClick={() => onChangeColorPreset('purple')}
              title="Purple Cyber Mode"
              className={`w-6 h-6 rounded-full border transition-all cursor-pointer ${
                activeColorPreset === 'purple'
                  ? 'bg-purple-500 border-purple-300 ring-2 ring-purple-500/40 shadow-[0_0_10px_rgba(168,85,247,0.8)]'
                  : 'bg-purple-950/60 border-purple-800 hover:border-purple-500'
              }`}
            />
            <button
              onClick={() => onChangeColorPreset('emerald')}
              title="Emerald Matrix Mode"
              className={`w-6 h-6 rounded-full border transition-all cursor-pointer ${
                activeColorPreset === 'emerald'
                  ? 'bg-emerald-500 border-emerald-300 ring-2 ring-emerald-500/40 shadow-[0_0_10px_rgba(16,185,129,0.8)]'
                  : 'bg-emerald-950/60 border-emerald-800 hover:border-emerald-500'
              }`}
            />
          </div>

          {/* Robot Arm Isolation Mode Button */}
          <button
            onClick={handleToggleArmIsolation}
            title={isArmIsolated ? "Show Full Scene" : "Isolate Robot Arm Only"}
            className={`flex items-center space-x-1.5 px-3 py-2 rounded-xl border text-xs font-mono transition-all cursor-pointer ${
              isArmIsolated
                ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-[0_0_12px_rgba(6,182,212,0.4)]'
                : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800'
            }`}
          >
            <Crosshair className={`w-3.5 h-3.5 ${isArmIsolated ? 'text-cyan-400 animate-spin' : 'text-slate-400'}`} />
            <span>{isArmIsolated ? "Arm Isolated" : "Isolate Arm"}</span>
          </button>

          {/* Object Hierarchy Inspector */}
          <button
            onClick={handleOpenInspector}
            title="Inspect 3D Scene Objects Tree"
            className="flex items-center space-x-1 px-3 py-2 rounded-xl bg-slate-900/80 hover:bg-slate-800 border border-slate-700 text-xs font-mono text-slate-300 transition-all cursor-pointer"
          >
            <Layers className="w-3.5 h-3.5 text-purple-400" />
            <span className="hidden sm:inline">Scene Layers</span>
          </button>

          {/* Sound Toggle */}
          <button
            onClick={handleToggleSound}
            title={isMuted ? "Enable Ambient Audio" : "Mute Sound"}
            className={`p-2 rounded-xl border text-xs font-mono transition-all cursor-pointer flex items-center space-x-1 ${
              !isMuted
                ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            {!isMuted ? <Volume2 className="w-4 h-4 text-cyan-400" /> : <VolumeX className="w-4 h-4" />}
          </button>

          {/* Reset Camera Button */}
          <button
            onClick={handleResetView}
            title="Reset 3D Camera Position"
            className="flex items-center space-x-1 px-3 py-2 rounded-xl bg-slate-900/80 hover:bg-slate-800 border border-slate-700 text-xs font-mono text-slate-200 transition-all cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5 text-cyan-400" />
            <span className="hidden sm:inline">Reset Camera</span>
          </button>
        </div>
      </div>

      {/* 3D Scene Layers Inspector Drawer */}
      {showInspector && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-in fade-in duration-150">
          <div className="relative w-full max-w-lg rounded-2xl glass-hud border border-cyan-500/30 p-6 shadow-2xl overflow-hidden">
            <div className="flex items-center justify-between pb-4 mb-4 border-b border-slate-800">
              <div className="flex items-center space-x-2.5">
                <Layers className="w-5 h-5 text-cyan-400" />
                <div>
                  <h3 className="text-base font-bold text-slate-100">Spline Scene Objects Tree</h3>
                  <p className="text-xs text-slate-400 font-mono">Toggle visibility for individual 3D meshes</p>
                </div>
              </div>
              <button
                onClick={() => setShowInspector(false)}
                className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="max-h-80 overflow-y-auto space-y-2 pr-1">
              {sceneObjects.length > 0 ? (
                sceneObjects.map((item, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/70 border border-slate-800 text-xs font-mono"
                  >
                    <span className="text-slate-200 truncate max-w-[260px]">{item.name}</span>
                    <button
                      onClick={() => handleToggleObjectVisibility(item.name)}
                      className={`flex items-center space-x-1.5 px-3 py-1 rounded-lg border transition-all cursor-pointer ${
                        item.visible
                          ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300'
                          : 'bg-slate-800 border-slate-700 text-slate-500'
                      }`}
                    >
                      {item.visible ? <Eye className="w-3.5 h-3.5 text-cyan-400" /> : <EyeOff className="w-3.5 h-3.5" />}
                      <span>{item.visible ? 'Visible' : 'Hidden'}</span>
                    </button>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-xs font-mono text-slate-400">
                  Loading scene object hierarchy... If empty, interact with the scene first.
                </div>
              )}
            </div>

            <div className="mt-4 pt-3 border-t border-slate-800 flex justify-between items-center text-xs font-mono text-slate-400">
              <span>Total Objects: {sceneObjects.length}</span>
              <button
                onClick={() => setShowInspector(false)}
                className="px-4 py-1.5 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-400/30 text-cyan-300 transition-all cursor-pointer"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
