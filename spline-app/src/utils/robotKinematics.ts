// ─── 5-DOF Robot Arm Joint State ─────────────────────────────────────────────
// DOF 1: Base    – Yaw rotation (waist)
// DOF 2: Shoulder – Pitch (up/down from base)
// DOF 3: Elbow   – Pitch (bend mid-arm)
// DOF 4: Wrist   – Pitch (wrist flex up/down)
// DOF 5: Gripper – Open/Close distance (0 = closed, 1 = fully open)

export interface JointState {
  base: number;      // DOF1 – Base Yaw     (radians, -PI to PI)
  shoulder: number;  // DOF2 – Shoulder Pitch (radians, -60° to 90°)
  elbow: number;     // DOF3 – Elbow Pitch   (radians, -90° to 80°)
  wrist: number;     // DOF4 – Wrist Pitch   (radians, -90° to 90°)
  gripper: number;   // DOF5 – Gripper Open  (0 = closed, 1 = open)
}

export const DEFAULT_JOINT_STATE: JointState = {
  base: 0,
  shoulder: (28 * Math.PI) / 180,
  elbow: (-50 * Math.PI) / 180,
  wrist: (20 * Math.PI) / 180,
  gripper: 0.35,
};

export interface ArmLengths {
  baseH: number;    // Base pedestal height
  upperArm: number; // Shoulder → Elbow link length
  forearm: number;  // Elbow → Wrist link length
  wristLen: number; // Wrist → Gripper tip length
}

export const DEFAULT_ARM_LENGTHS: ArmLengths = {
  baseH: 1.2,
  upperArm: 2.2,
  forearm: 1.9,
  wristLen: 0.75,
};

// ─── Forward Kinematics ───────────────────────────────────────────────────────
// Returns 3D position of the gripper tip
export function forwardKinematics(
  j: JointState,
  L: ArmLengths = DEFAULT_ARM_LENGTHS
): THREE_Vec3 {
  const totalPitch = j.shoulder;
  const elbowPitch = totalPitch + j.elbow;
  const wristPitch = elbowPitch + j.wrist;

  const r1 = L.upperArm * Math.sin(totalPitch);
  const y1 = L.upperArm * Math.cos(totalPitch);

  const r2 = r1 + L.forearm * Math.sin(elbowPitch);
  const y2 = y1 + L.forearm * Math.cos(elbowPitch);

  const r3 = r2 + L.wristLen * Math.sin(wristPitch);
  const y3 = y2 + L.wristLen * Math.cos(wristPitch);

  return {
    x: r3 * Math.cos(j.base),
    y: L.baseH + y3,
    z: r3 * Math.sin(j.base),
  };
}

// ─── Analytical 2-Link IK (3-link approximated) ──────────────────────────────
export function inverseKinematics(
  tx: number,
  ty: number,
  tz: number,
  L: ArmLengths = DEFAULT_ARM_LENGTHS
): Pick<JointState, 'base' | 'shoulder' | 'elbow'> {
  const base = Math.atan2(tz, tx);

  const R = Math.sqrt(tx * tx + tz * tz);
  const dy = ty - L.baseH;

  // Treat wristLen as part of reach target
  const L1 = L.upperArm;
  const L2 = L.forearm + L.wristLen;

  const dist = Math.sqrt(R * R + dy * dy);
  const maxReach = (L1 + L2) * 0.99;
  const minReach = Math.abs(L1 - L2) * 1.01;
  const d = Math.max(minReach, Math.min(maxReach, dist));

  const cosElbow = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2);
  const elbow = -Math.acos(Math.max(-1, Math.min(1, cosElbow)));

  const alpha = Math.atan2(R, dy);
  const beta = Math.atan2(L2 * Math.sin(-elbow), L1 + L2 * Math.cos(-elbow));
  const shoulder = alpha - beta;

  return { base, shoulder, elbow };
}

// ─── Exponential Damped Lerp (game-physics spring) ───────────────────────────
export function damp(current: number, target: number, speed: number, dt: number): number {
  return current + (target - current) * (1 - Math.exp(-speed * dt));
}

// Type alias so kinematics.ts has no Three.js dep
interface THREE_Vec3 { x: number; y: number; z: number; }
