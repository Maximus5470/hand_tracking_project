'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three-stdlib';
import {
  JointState,
  DEFAULT_ARM_LENGTHS,
  inverseKinematics,
  damp,
} from '@/utils/robotKinematics';

interface Props {
  joints: JointState;
  onJointsChange: (j: JointState) => void;
  isIkMode: boolean;
  activeColorPreset: 'cyan' | 'purple' | 'emerald';
  isCarryingCargo: boolean;
  onCargoStateChange: (c: boolean) => void;
}

// ─── small helpers ─────────────────────────────────────────────────────────
function cyl(
  parent: THREE.Object3D,
  mat: THREE.Material,
  rT: number, rB: number, h: number,
  segs = 28,
  offsetY = 0
): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(rT, rB, h, segs), mat);
  m.position.y = offsetY;
  m.castShadow = true;
  m.receiveShadow = true;
  parent.add(m);
  return m;
}

function box(
  parent: THREE.Object3D,
  mat: THREE.Material,
  w: number, h: number, d: number,
  px = 0, py = 0, pz = 0
): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(px, py, pz);
  m.castShadow = true;
  m.receiveShadow = true;
  parent.add(m);
  return m;
}

function torus(
  parent: THREE.Object3D,
  mat: THREE.Material,
  r: number, tube: number,
  rx = 0, ry = 0, rz = 0,
  px = 0, py = 0, pz = 0
): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.TorusGeometry(r, tube, 16, 40), mat);
  m.rotation.set(rx, ry, rz);
  m.position.set(px, py, pz);
  m.castShadow = true;
  parent.add(m);
  return m;
}

function pivotDisc(
  parent: THREE.Object3D,
  jointMat: THREE.Material,
  chromeMat: THREE.Material,
  radius = 0.42,
  width = 0.70
): void {
  const d = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, width, 36),
    jointMat
  );
  d.rotation.z = Math.PI / 2;
  d.castShadow = true;
  parent.add(d);

  [-width / 2, width / 2].forEach((x) => {
    torus(parent, chromeMat, radius * 0.88, 0.028, 0, Math.PI / 2, 0, x, 0, 0);
  });

  cyl(parent, chromeMat, 0.10, 0.10, width + 0.02, 16);
}

interface PhysicsBody {
  mesh: THREE.Mesh;
  velocity: THREE.Vector3;
  halfHeight: number;
  isHeld: boolean;
  isMouseDragging?: boolean;
}

export default function ThreeRobotArmCanvas({
  joints, isIkMode, isCarryingCargo, onCargoStateChange,
}: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  const curRef = useRef<JointState>({ ...joints });
  const ikTarget = useRef(new THREE.Vector3(1.6, 2.6, 0));
  const dragging = useRef(false);

  const jointsRef = useRef<JointState>(joints);
  jointsRef.current = joints;

  const isCarryingCargoRef = useRef<boolean>(isCarryingCargo);
  isCarryingCargoRef.current = isCarryingCargo;

  const grabbedObjRef = useRef<number>(-1);

  const groupsRef = useRef<{
    J0: THREE.Group;
    J1: THREE.Group;
    J2: THREE.Group;
    J3: THREE.Group;
    J4L: THREE.Group;
    J4R: THREE.Group;
    ikSphere: THREE.Mesh;
    physicsBodies: PhysicsBody[];
    wristGroup: THREE.Group;
  } | null>(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // ── Scene & Renderer ─────────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060a12);
    scene.fog = new THREE.FogExp2(0x060a12, 0.022);

    const W = container.clientWidth, H = container.clientHeight;
    const camera = new THREE.PerspectiveCamera(44, W / H, 0.1, 80);
    camera.position.set(5.5, 4.0, 6.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.35;
    container.appendChild(renderer.domElement);

    // ── Orbit Controls ───────────────────────────────────────────────
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.target.set(0, 2.2, 0);
    controls.maxPolarAngle = Math.PI / 1.9;
    controls.minDistance = 2.5;
    controls.maxDistance = 20;

    // ── LIGHTING ─────────────────────────────────────────────────────
    const hemi = new THREE.HemisphereLight(0xf0f6ff, 0x1a2436, 1.6);
    scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 5.0);
    key.position.set(4, 7, 5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -8; key.shadow.camera.right = 8;
    key.shadow.camera.top = 12; key.shadow.camera.bottom = -3;
    key.shadow.bias = -0.0003;
    scene.add(key);

    const fill = new THREE.DirectionalLight(0xa5c4db, 2.8);
    fill.position.set(-4, 4, 5);
    scene.add(fill);

    const rim = new THREE.DirectionalLight(0x38bdf8, 2.2);
    rim.position.set(-2, 8, -5);
    scene.add(rim);

    const pt1 = new THREE.PointLight(0xffffff, 3.5, 9);
    pt1.position.set(1.5, 3.5, 2.5);
    scene.add(pt1);

    const pt2 = new THREE.PointLight(0xa855f7, 1.8, 6);
    pt2.position.set(-2, 2, 2);
    scene.add(pt2);

    // ── Floor ────────────────────────────────────────────────────────
    const floorMesh = new THREE.Mesh(
      new THREE.CircleGeometry(12, 60),
      new THREE.MeshStandardMaterial({ color: 0x0a101d, metalness: 0.5, roughness: 0.85 })
    );
    floorMesh.rotation.x = -Math.PI / 2;
    floorMesh.receiveShadow = true;
    scene.add(floorMesh);

    const grid = new THREE.GridHelper(22, 44, 0x1e3a5f, 0x0f1e2e);
    grid.position.y = 0.002;
    scene.add(grid);

    // ── Materials (Arctic White Theme) ───────────────────────────────
    const darkMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, metalness: 0.12, roughness: 0.18 });
    const midMat  = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.40, roughness: 0.22 });
    const chromeMat = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.95, roughness: 0.08 });
    const jointMat  = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.85, roughness: 0.20 });
    const accentMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.90, roughness: 0.15 });

    // ── ARM ROOT GROUP ───────────────────────────────────────────────
    const armRoot = new THREE.Group();
    scene.add(armRoot);

    // BASE
    cyl(armRoot, darkMat, 0.88, 1.65, 0.95, 40, 0.475);
    cyl(armRoot, midMat,  0.85, 0.90, 0.20, 40, 1.00);
    cyl(armRoot, chromeMat, 0.86, 0.86, 0.10, 40, 1.15);
    torus(armRoot, chromeMat, 0.88, 0.04, Math.PI / 2, 0, 0, 0, 0.04, 0);

    // DOF 1 – J0: BASE
    const J0 = new THREE.Group();
    J0.position.y = 1.20;
    armRoot.add(J0);
    cyl(J0, jointMat, 0.72, 0.72, 0.30, 36, 0.0);
    torus(J0, chromeMat, 0.74, 0.035, Math.PI / 2, 0, 0, 0, 0.0, 0);

    // DOF 2 – J1: SHOULDER
    const J1 = new THREE.Group();
    J1.position.y = 0.24;
    J0.add(J1);
    pivotDisc(J1, jointMat, chromeMat, 0.46, 0.74);

    const upperArmGroup = new THREE.Group();
    J1.add(upperArmGroup);

    const UA = DEFAULT_ARM_LENGTHS.upperArm; // 2.2
    box(upperArmGroup, darkMat, 0.22, UA, 0.36,  -0.27, UA / 2, 0);
    box(upperArmGroup, darkMat, 0.22, UA, 0.36,   0.27, UA / 2, 0);
    box(upperArmGroup, chromeMat, 0.10, UA * 0.88, 0.12, 0, UA / 2, 0);
    [0.38, 0.82, 1.26, 1.70, 2.06].forEach(y =>
      box(upperArmGroup, midMat, 0.54, 0.08, 0.28, 0, y, 0)
    );

    // DOF 3 – J2: ELBOW
    const J2 = new THREE.Group();
    J2.position.y = UA;
    upperArmGroup.add(J2);
    pivotDisc(J2, jointMat, chromeMat, 0.42, 0.70);

    const forearmGroup = new THREE.Group();
    J2.add(forearmGroup);

    const FA = DEFAULT_ARM_LENGTHS.forearm; // 1.9
    cyl(forearmGroup, darkMat, 0.22, 0.32, FA, 24, FA / 2);
    box(forearmGroup, chromeMat, 0.08, FA * 0.82, 0.08, 0.26, FA / 2, 0);
    cyl(forearmGroup, midMat,  0.068, 0.068, FA * 0.68, 14, FA * 0.38);
    cyl(forearmGroup, chromeMat, 0.044, 0.044, FA * 0.52, 12, FA * 0.72);

    // DOF 4 – J3: WRIST
    const J3 = new THREE.Group();
    J3.position.y = FA;
    forearmGroup.add(J3);
    pivotDisc(J3, jointMat, chromeMat, 0.30, 0.55);

    const wristGroup = new THREE.Group();
    wristGroup.position.y = 0.06;
    J3.add(wristGroup);

    cyl(wristGroup, jointMat, 0.24, 0.28, 0.32, 28, 0.22);
    torus(wristGroup, chromeMat, 0.265, 0.022, Math.PI / 2, 0, 0, 0, 0.38, 0);
    box(wristGroup, accentMat, 0.38, 0.22, 0.38, 0, 0.48, 0);

    // DOF 5 – GRIPPER
    box(wristGroup, jointMat, 0.44, 0.16, 0.20, 0, 0.67, 0);

    const J4L = new THREE.Group();
    J4L.position.set(-0.16, 0.75, 0);
    wristGroup.add(J4L);
    box(J4L, darkMat, 0.16, 0.08, 0.16, 0, 0, 0);
    box(J4L, darkMat, 0.06, 0.28, 0.14, -0.05, 0.18, 0);
    box(J4L, accentMat, 0.08, 0.22, 0.16, 0.02, 0.21, 0);

    const J4R = new THREE.Group();
    J4R.position.set(0.16, 0.75, 0);
    wristGroup.add(J4R);
    box(J4R, darkMat, 0.16, 0.08, 0.16, 0, 0, 0);
    box(J4R, darkMat, 0.06, 0.28, 0.14, 0.05, 0.18, 0);
    box(J4R, accentMat, 0.08, 0.22, 0.16, -0.02, 0.21, 0);

    // IK Target sphere
    const ikSphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, 20, 20),
      new THREE.MeshStandardMaterial({
        color: 0x06b6d4, emissive: 0x06b6d4, emissiveIntensity: 1.4, wireframe: true,
      })
    );
    scene.add(ikSphere);

    // ════════════════════════════════════════════════════════════════
    // FREEFORM SAMPLE OBJECTS WITH FULL RIGID-BODY PHYSICS & GRAVITY
    // ════════════════════════════════════════════════════════════════
    const physicsBodies: PhysicsBody[] = [];

    // Object 1: Glowing Purple Energy Cube
    const cargo1 = new THREE.Mesh(
      new THREE.BoxGeometry(0.38, 0.38, 0.38),
      new THREE.MeshStandardMaterial({
        color: 0xa855f7, metalness: 0.6, roughness: 0.2,
        emissive: 0x9333ea, emissiveIntensity: 0.3,
      })
    );
    cargo1.position.set(-2.4, 0.19, -2.5);
    cargo1.castShadow = true; cargo1.receiveShadow = true;
    scene.add(cargo1);
    physicsBodies.push({ mesh: cargo1, velocity: new THREE.Vector3(), halfHeight: 0.19, isHeld: false });

    // Object 2: Glowing Cyan Energy Sphere
    const cargo2 = new THREE.Mesh(
      new THREE.SphereGeometry(0.22, 24, 24),
      new THREE.MeshStandardMaterial({
        color: 0x06b6d4, metalness: 0.7, roughness: 0.15,
        emissive: 0x0891b2, emissiveIntensity: 0.4,
      })
    );
    cargo2.position.set(3.0, 0.22, -2.4);
    cargo2.castShadow = true; cargo2.receiveShadow = true;
    scene.add(cargo2);
    physicsBodies.push({ mesh: cargo2, velocity: new THREE.Vector3(), halfHeight: 0.22, isHeld: false });

    // Object 3: Gold Metal Cylinder
    const cargo3 = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.18, 0.40, 24),
      new THREE.MeshStandardMaterial({
        color: 0xeab308, metalness: 0.9, roughness: 0.15,
        emissive: 0xca8a04, emissiveIntensity: 0.2,
      })
    );
    cargo3.position.set(-0.4, 0.20, -3.0);
    cargo3.castShadow = true; cargo3.receiveShadow = true;
    scene.add(cargo3);
    physicsBodies.push({ mesh: cargo3, velocity: new THREE.Vector3(), halfHeight: 0.20, isHeld: false });

    // Object half-widths for collision (cube=0.19, sphere=0.22, cylinder=0.18)
    const objectRadii = [0.19, 0.22, 0.18];

    groupsRef.current = { J0, J1, J2, J3, J4L, J4R, ikSphere, physicsBodies, wristGroup };
    setIsLoaded(true);

    // ── IK Target Dragging Handler (Objects cannot be dragged by mouse) ─
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const dragPlane = new THREE.Plane();

    const onDown = (e: MouseEvent) => {
      if (!isIkMode) return;
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.set(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );
      raycaster.setFromCamera(mouse, camera);

      if (raycaster.intersectObject(ikSphere).length > 0) {
        dragging.current = true;
        controls.enabled = false;
      }
    };

    const onMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.set(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );
      raycaster.setFromCamera(mouse, camera);

      dragPlane.setFromNormalAndCoplanarPoint(
        camera.getWorldDirection(new THREE.Vector3()).negate(),
        ikTarget.current
      );
      const hit = new THREE.Vector3();
      if (raycaster.ray.intersectPlane(dragPlane, hit)) {
        hit.y = Math.max(0.4, hit.y);
        ikTarget.current.copy(hit);
      }
    };

    const onUp = () => {
      dragging.current = false;
      controls.enabled = true;
    };

    renderer.domElement.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);

    // ── Render / Physics animation loop ──────────────────────────────
    let rafId: number;
    let lastT = performance.now();

    const animate = (t: number) => {
      rafId = requestAnimationFrame(animate);
      const dt = Math.min((t - lastT) / 1000, 0.1);
      lastT = t;

      const g = groupsRef.current;
      if (g) {
        const c = curRef.current;
        const spd = 14;

        const target = jointsRef.current;

        if (isIkMode) {
          const ik = inverseKinematics(ikTarget.current.x, ikTarget.current.y, ikTarget.current.z);
          c.base     = damp(c.base,     ik.base,     spd, dt);
          c.shoulder = damp(c.shoulder, ik.shoulder, spd, dt);
          c.elbow    = damp(c.elbow,    ik.elbow,    spd, dt);
          c.wrist    = damp(c.wrist,    target.wrist, spd, dt);
          c.gripper  = damp(c.gripper,  target.gripper, spd, dt);
        } else {
          c.base     = damp(c.base,     target.base,     spd, dt);
          c.shoulder = damp(c.shoulder, target.shoulder, spd, dt);
          c.elbow    = damp(c.elbow,    target.elbow,    spd, dt);
          c.wrist    = damp(c.wrist,    target.wrist,    spd, dt);
          c.gripper  = damp(c.gripper,  target.gripper,  spd, dt);
        }

        // Apply joint transforms
        g.J0.rotation.y = c.base;
        g.J1.rotation.x = c.shoulder;
        g.J2.rotation.x = c.elbow;
        g.J3.rotation.x = c.wrist;

        // ── Gripper Collision-Aware Sliding ──────────────────────────
        // Compute desired spread from gripper value
        let desiredSpread = 0.08 + c.gripper * 0.12;

        // Get exact grip tip world position by transforming local tip coordinates (0, 0.80, 0)
        const gripTip = new THREE.Vector3(0, 0.80, 0);
        g.wristGroup.localToWorld(gripTip);

        // Check if an unheld object is between the fingers.
        // If so, clamp the spread so the fingers press against the object surface.
        g.physicsBodies.forEach((body, idx) => {
          if (body.isHeld) return;
          const dist = body.mesh.position.distanceTo(gripTip);
          if (dist < 0.75) {
            const objR = objectRadii[idx] || 0.20;
            const minSpread = objR + 0.02; // tiny pad so fingers rest on surface
            if (desiredSpread < minSpread) {
              desiredSpread = minSpread;
            }
          }
        });

        g.J4L.position.x = -desiredSpread;
        g.J4R.position.x =  desiredSpread;

        g.ikSphere.visible = isIkMode;
        if (isIkMode) g.ikSphere.position.copy(ikTarget.current);

        // ── 3D RIGID-BODY PHYSICS & FREE GRAVITY SIMULATION ────────────

        // Touch-Snap Grab logic:
        // When touching an object (< 0.90m), the object automatically snaps to the gripper UNLESS the gripper open angle is high (>= 0.60).
        if (target.gripper < 0.60) {
          if (grabbedObjRef.current === -1) {
            let closestIdx = -1;
            let minDist = 0.90;

            g.physicsBodies.forEach((body, idx) => {
              const dist = body.mesh.position.distanceTo(gripTip);
              if (dist < minDist) {
                minDist = dist;
                closestIdx = idx;
              }
            });

            if (closestIdx !== -1) {
              grabbedObjRef.current = closestIdx;
              g.physicsBodies[closestIdx].isHeld = true;
              onCargoStateChange(true);
            }
          }
        } else {
          // When gripper angle is high (>= 0.60), release the object!
          if (grabbedObjRef.current !== -1) {
            g.physicsBodies[grabbedObjRef.current].isHeld = false;
            grabbedObjRef.current = -1;
            onCargoStateChange(false);
          }
        }

        const GRAVITY = -11.8;

        g.physicsBodies.forEach((body) => {
          if (body.isHeld) {
            // Follow end-effector; track velocity for throw momentum on release
            const prevPos = body.mesh.position.clone();
            body.mesh.position.copy(gripTip);
            body.mesh.quaternion.copy(g.wristGroup.getWorldQuaternion(new THREE.Quaternion()));

            if (dt > 0) {
              body.velocity.copy(body.mesh.position).sub(prevPos).divideScalar(dt);
            }
          } else {
            // FREE GRAVITY & BOUNCE PHYSICS
            body.velocity.y += GRAVITY * dt;

            // Air resistance
            body.velocity.x *= 0.985;
            body.velocity.z *= 0.985;

            // Update position
            body.mesh.position.addScaledVector(body.velocity, dt);

            // Floor collision
            const floorY = body.halfHeight;
            if (body.mesh.position.y <= floorY) {
              body.mesh.position.y = floorY;

              if (Math.abs(body.velocity.y) > 0.6) {
                body.velocity.y = -body.velocity.y * 0.38;
              } else {
                body.velocity.y = 0;
              }

              body.velocity.x *= 0.86;
              body.velocity.z *= 0.86;
            }
          }
        });
      }

      controls.update();
      renderer.render(scene, camera);
    };
    rafId = requestAnimationFrame(animate);

    // ── Resize ───────────────────────────────────────────────────────
    const onResize = () => {
      if (!container) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener('resize', onResize);
      renderer.domElement.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      if (renderer.domElement.parentNode)
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      renderer.dispose();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isIkMode]);

  return (
    <div className="fixed inset-0 w-full h-full bg-[#060a12] cursor-grab active:cursor-grabbing">
      <div ref={mountRef} className="w-full h-full" />
      {!isLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#060a12] text-cyan-400 font-mono text-sm tracking-widest uppercase">
          <span className="animate-pulse">Initializing Arctic White 5-DOF Robot Physics...</span>
        </div>
      )}
    </div>
  );
}
