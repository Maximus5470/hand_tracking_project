'use client';

import { JointState } from '@/utils/robotKinematics';
import { Sliders, Crosshair, RotateCcw, Play, Box, Radio } from 'lucide-react';

interface Props {
  joints: JointState;
  onJointsChange: (j: JointState) => void;
  isIkMode: boolean;
  onToggleIkMode: (ik: boolean) => void;
  isWebcamMode: boolean;
  onToggleWebcamMode: (webcam: boolean) => void;
  isCarryingCargo: boolean;
  onToggleCargo: () => void;
}

const r2d = (r: number) => Math.round((r * 180) / Math.PI);
const d2r = (d: number) => (d * Math.PI) / 180;

interface SliderDef {
  key: keyof JointState;
  label: string;
  dof: string;
  min: number;
  max: number;
  color: string;
  isRaw?: boolean; // for gripper which is 0-1 already
}

const SLIDERS: SliderDef[] = [
  { key: 'base',     label: 'Base Rotation',  dof: 'DOF 1', min: -180, max: 180, color: 'accent-cyan-400' },
  { key: 'shoulder', label: 'Shoulder Pitch',  dof: 'DOF 2', min: -60,  max: 90,  color: 'accent-purple-400' },
  { key: 'elbow',    label: 'Elbow Pitch',     dof: 'DOF 3', min: -90,  max: 80,  color: 'accent-emerald-400' },
  { key: 'wrist',    label: 'Wrist Pitch',     dof: 'DOF 4', min: -90,  max: 90,  color: 'accent-pink-400' },
];

export default function RobotControlsPanel({
  joints, onJointsChange, isIkMode, onToggleIkMode, isWebcamMode, onToggleWebcamMode, isCarryingCargo, onToggleCargo,
}: Props) {

  const handleAngleChange = (key: keyof JointState, deg: number) => {
    onJointsChange({ ...joints, [key]: d2r(deg) });
  };
  const handleGripperChange = (val: number) => {
    onJointsChange({ ...joints, gripper: val });
  };

  const presetHome = () => onJointsChange({
    base: 0, shoulder: d2r(28), elbow: d2r(-50), wrist: d2r(20), gripper: 0.35,
  });
  const presetPickup = () => onJointsChange({
    base: d2r(38), shoulder: d2r(62), elbow: d2r(-72), wrist: d2r(30), gripper: 0.85,
  });
  const presetHighReach = () => onJointsChange({
    base: d2r(-25), shoulder: d2r(-15), elbow: d2r(35), wrist: d2r(-40), gripper: 1.0,
  });

  return (
    <div className="w-full glass-hud rounded-2xl p-5 border border-cyan-500/25 shadow-2xl backdrop-blur-xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 mb-4 border-b border-slate-800">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-xl bg-cyan-500/10 border border-cyan-500/30">
            <Radio className={`w-5 h-5 ${isWebcamMode ? 'text-pink-500 animate-pulse' : 'text-cyan-400'}`} />
          </div>
          <div>
            <h3 className="text-sm font-extrabold text-slate-100 uppercase tracking-widest">
              5-DOF Robotic Arm Control
            </h3>
            <p className="text-[11px] text-slate-400 font-mono">
              Base · Shoulder · Elbow · Wrist · Gripper
            </p>
          </div>
        </div>

        {/* Control Modes Toggle */}
        <div className="flex items-center bg-slate-950/80 p-1 rounded-xl border border-slate-800 text-xs font-mono">
          <button
            onClick={() => { onToggleWebcamMode(!isWebcamMode); onToggleIkMode(false); }}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
              isWebcamMode
                ? 'bg-pink-500/20 border border-pink-400/60 text-pink-300'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            <Radio className="w-3 h-3" />
            <span>Webcam</span>
          </button>
          <div className="w-px h-4 bg-slate-800 mx-1"></div>
          <button
            onClick={() => { onToggleIkMode(true); onToggleWebcamMode(false); }}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
              isIkMode && !isWebcamMode
                ? 'bg-cyan-500/20 border border-cyan-400/60 text-cyan-300'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            <Crosshair className="w-3 h-3" />
            <span>IK</span>
          </button>
          <button
            onClick={() => { onToggleIkMode(false); onToggleWebcamMode(false); }}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
              !isIkMode && !isWebcamMode
                ? 'bg-purple-500/20 border border-purple-400/60 text-purple-300'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            <Sliders className="w-3 h-3" />
            <span>FK</span>
          </button>
        </div>
      </div>

      {/* DOF 1-4 Sliders */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
        {SLIDERS.map(({ key, label, dof, min, max, color }) => (
          <div key={key} className="bg-slate-900/60 p-3 rounded-xl border border-slate-800/80">
            <div className="flex justify-between text-xs font-mono mb-1.5">
              <span className="text-slate-400">
                <span className="text-slate-600 mr-1">{dof}</span>{label}
              </span>
              <span className="text-slate-200 font-bold">{r2d(joints[key] as number)}°</span>
            </div>
            <input
              type="range"
              min={min}
              max={max}
              disabled={(isIkMode && key !== 'wrist') || isWebcamMode}
              value={r2d(joints[key] as number)}
              onChange={e => handleAngleChange(key, Number(e.target.value))}
              className={`w-full h-1.5 bg-slate-800 rounded-full appearance-none ${color} ${
                isWebcamMode ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'
              } disabled:opacity-35`}
            />
          </div>
        ))}
      </div>

      {/* DOF 5 – Gripper + Cargo + Presets */}
      <div className="flex flex-col sm:flex-row items-center gap-3 pt-3 border-t border-slate-800/60">
        {/* Gripper slider */}
        <div className="flex-1 bg-slate-900/60 p-3 rounded-xl border border-slate-800/80 w-full">
          <div className="flex justify-between text-xs font-mono mb-1.5">
            <span className="text-slate-400"><span className="text-slate-600 mr-1">DOF 5</span>Gripper Open</span>
            <span className="text-cyan-300 font-bold">{Math.round(joints.gripper * 100)}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.02"
            disabled={isWebcamMode}
            value={joints.gripper}
            onChange={e => handleGripperChange(Number(e.target.value))}
            className={`w-full h-1.5 bg-slate-800 rounded-full appearance-none accent-cyan-400 ${
              isWebcamMode ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'
            } disabled:opacity-35`}
          />
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            onClick={onToggleCargo}
            className={`flex items-center space-x-1.5 px-3.5 py-2 rounded-xl border text-xs font-mono font-medium transition-all cursor-pointer ${
              isCarryingCargo
                ? 'bg-emerald-500/20 border-emerald-400/60 text-emerald-300 shadow-[0_0_10px_rgba(16,185,129,0.3)]'
                : 'bg-slate-900 border-slate-700 text-slate-300 hover:border-slate-600'
            }`}
          >
            <Box className={`w-3.5 h-3.5 ${isCarryingCargo ? 'text-emerald-400' : 'text-slate-500'}`} />
            <span>{isCarryingCargo ? 'Release' : 'Pick Cargo'}</span>
          </button>

          {/* Hide Presets when Webcam overriding them */}
          {!isWebcamMode && (
            <>
              <button onClick={presetHome}    title="Home stance" className="p-2 rounded-xl bg-slate-900 border border-slate-700 text-slate-300 hover:border-slate-500 transition-all cursor-pointer">
                <RotateCcw className="w-3.5 h-3.5 text-cyan-400" />
              </button>
              <button onClick={presetPickup}  title="Pickup stance" className="px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-slate-300 hover:border-purple-500/60 transition-all cursor-pointer text-xs font-mono">
                Pick
              </button>
              <button onClick={presetHighReach} title="High reach" className="px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-slate-300 hover:border-cyan-500/60 transition-all cursor-pointer text-xs font-mono">
                Reach
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
