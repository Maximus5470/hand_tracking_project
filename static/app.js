import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const e = React.createElement;
const INITIAL_ANGLES = [75, 90, 55, 65, 60];

function makeMetalMaterial(color, emissive = 0x000000, emissiveIntensity = 0.0) {
  return new THREE.MeshStandardMaterial({
    color,
    metalness: 0.45,
    roughness: 0.32,
    emissive,
    emissiveIntensity,
  });
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function RobotScene({ angles }) {
  const canvasRef = React.useRef(null);
  const targetsRef = React.useRef(angles);

  React.useEffect(() => {
    targetsRef.current = angles;
  }, [angles]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return undefined;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07101d);
    scene.fog = new THREE.Fog(0x07101d, 18, 48);

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;

    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 120);
    camera.position.set(8, 8, 14);
    scene.add(camera);

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 4.2, 0);
    controls.minDistance = 9;
    controls.maxDistance = 24;
    controls.maxPolarAngle = Math.PI * 0.72;

    scene.add(new THREE.AmbientLight(0xb7d6ff, 0.72));

    const mainLight = new THREE.DirectionalLight(0xffffff, 2.0);
    mainLight.position.set(8, 14, 10);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.set(2048, 2048);
    scene.add(mainLight);

    const rimLight = new THREE.DirectionalLight(0x7af2c8, 0.95);
    rimLight.position.set(-10, 6, -4);
    scene.add(rimLight);

    const accentLight = new THREE.PointLight(0x7ad7ff, 1.4, 40);
    accentLight.position.set(-6, 8, 8);
    scene.add(accentLight);

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(16, 72),
      new THREE.MeshStandardMaterial({ color: 0x09121f, metalness: 0.15, roughness: 0.94 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    const grid = new THREE.GridHelper(32, 32, 0x36547a, 0x203144);
    grid.position.y = 0.01;
    scene.add(grid);

    const armRoot = new THREE.Group();
    armRoot.position.set(0, 0.5, 0);
    scene.add(armRoot);

    const baseGroup = new THREE.Group();
    armRoot.add(baseGroup);

    const base = new THREE.Mesh(
      new THREE.CylinderGeometry(1.35, 1.55, 0.8, 32),
      makeMetalMaterial(0x23364d)
    );
    base.castShadow = true;
    base.receiveShadow = true;
    baseGroup.add(base);

    const pedestal = new THREE.Mesh(
      new THREE.CylinderGeometry(0.52, 0.62, 1.4, 32),
      makeMetalMaterial(0x5fd3ff, 0x0c3046, 0.22)
    );
    pedestal.position.y = 1.05;
    pedestal.castShadow = true;
    baseGroup.add(pedestal);

    const shoulderPivot = new THREE.Group();
    shoulderPivot.position.y = 1.8;
    baseGroup.add(shoulderPivot);

    const shoulderJoint = new THREE.Group();
    shoulderPivot.add(shoulderJoint);

    const shoulderBall = new THREE.Mesh(
      new THREE.SphereGeometry(0.42, 32, 24),
      makeMetalMaterial(0x9cead1, 0x0d2e2a, 0.16)
    );
    shoulderBall.castShadow = true;
    shoulderJoint.add(shoulderBall);

    const upperArm = new THREE.Mesh(
      new THREE.BoxGeometry(1.05, 3.4, 1.05),
      makeMetalMaterial(0x77f2c5, 0x0d2e2a, 0.2)
    );
    upperArm.position.y = 1.9;
    upperArm.castShadow = true;
    shoulderJoint.add(upperArm);

    const elbowPivot = new THREE.Group();
    elbowPivot.position.y = 3.7;
    shoulderJoint.add(elbowPivot);

    const elbowJoint = new THREE.Group();
    elbowPivot.add(elbowJoint);

    const elbowCap = new THREE.Mesh(
      new THREE.SphereGeometry(0.36, 28, 20),
      makeMetalMaterial(0x5aa6ff, 0x0b2044, 0.18)
    );
    elbowCap.castShadow = true;
    elbowJoint.add(elbowCap);

    const forearm = new THREE.Mesh(
      new THREE.BoxGeometry(0.9, 3.0, 0.9),
      makeMetalMaterial(0x5aa6ff, 0x0b2044, 0.18)
    );
    forearm.position.y = 1.55;
    forearm.castShadow = true;
    elbowJoint.add(forearm);

    const wristPivot = new THREE.Group();
    wristPivot.position.y = 3.1;
    elbowJoint.add(wristPivot);

    const wristJoint = new THREE.Group();
    wristPivot.add(wristJoint);

    const wrist = new THREE.Mesh(
      new THREE.CylinderGeometry(0.34, 0.42, 1.0, 24),
      makeMetalMaterial(0xf2f7ff, 0x111827, 0.08)
    );
    wrist.rotation.z = Math.PI / 2;
    wrist.castShadow = true;
    wristJoint.add(wrist);

    const palm = new THREE.Mesh(
      new THREE.BoxGeometry(1.0, 0.7, 0.95),
      makeMetalMaterial(0x23364d)
    );
    palm.position.set(0, 0, 0.1);
    palm.castShadow = true;
    wristJoint.add(palm);

    function createFinger(color) {
      const finger = new THREE.Group();
      const upper = new THREE.Mesh(
        new THREE.BoxGeometry(0.18, 0.92, 0.16),
        makeMetalMaterial(color)
      );
      upper.position.y = 0.46;
      upper.castShadow = true;
      finger.add(upper);

      const tip = new THREE.Mesh(
        new THREE.BoxGeometry(0.16, 0.46, 0.14),
        makeMetalMaterial(0xf4f7ff)
      );
      tip.position.y = 1.02;
      tip.castShadow = true;
      finger.add(tip);
      return finger;
    }

    const leftFingerPivot = new THREE.Group();
    leftFingerPivot.position.set(-0.34, 0, 0.5);
    wristJoint.add(leftFingerPivot);

    const rightFingerPivot = new THREE.Group();
    rightFingerPivot.position.set(0.34, 0, 0.5);
    wristJoint.add(rightFingerPivot);

    const leftFinger = createFinger(0x77f2c5);
    leftFinger.position.y = 0.16;
    leftFingerPivot.add(leftFinger);

    const rightFinger = createFinger(0x5aa6ff);
    rightFinger.position.y = 0.16;
    rightFingerPivot.add(rightFinger);

    const smoothed = [...angles];
    let animationFrame = 0;

    function resize() {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) {
        return;
      }
      renderer.setSize(rect.width, rect.height, false);
      camera.aspect = rect.width / rect.height;
      camera.updateProjectionMatrix();
    }

    function applyPose() {
      const baseYaw = THREE.MathUtils.degToRad((smoothed[0] - 75) * 1.2);
      const shoulderPitch = THREE.MathUtils.degToRad((90 - smoothed[1]) * 0.95);
      const elbowPitch = THREE.MathUtils.degToRad((55 - smoothed[2]) * 0.9);
      const wristRoll = THREE.MathUtils.degToRad((smoothed[3] - 65) * 1.1);
      const fingerOpen = THREE.MathUtils.mapLinear(clamp(smoothed[4], 0, 60), 0, 60, 0.18, 0.82);

      baseGroup.rotation.y = baseYaw;
      shoulderJoint.rotation.z = shoulderPitch;
      elbowJoint.rotation.z = elbowPitch;
      wristJoint.rotation.y = wristRoll;
      leftFingerPivot.rotation.z = -fingerOpen;
      rightFingerPivot.rotation.z = fingerOpen;
    }

    function tick() {
      const target = targetsRef.current;
      smoothed[0] += (target[0] - smoothed[0]) * 0.12;
      smoothed[1] += (target[1] - smoothed[1]) * 0.12;
      smoothed[2] += (target[2] - smoothed[2]) * 0.12;
      smoothed[3] += (target[3] - smoothed[3]) * 0.12;
      smoothed[4] += (target[4] - smoothed[4]) * 0.12;

      applyPose();
      controls.update();
      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(tick);
    }

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    window.addEventListener('resize', resize);

    resize();
    tick();

    return () => {
      window.cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      window.removeEventListener('resize', resize);
      controls.dispose();
      renderer.dispose();
    };
  }, []);

  return e('canvas', {
    ref: canvasRef,
    id: 'sceneCanvas',
    'aria-label': '3D robot arm simulation',
  });
}

function MetricCard({ label, value, index }) {
  return e(
    'div',
    { className: 'metric' },
    e('span', null, label),
    e('strong', { 'data-angle': index }, value)
  );
}

function App() {
  const [angles, setAngles] = useState(INITIAL_ANGLES);
  const [cameraReady, setCameraReady] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Starting...');
  const [stateSummary, setStateSummary] = useState('Waiting for camera state.');

  useEffect(() => {
    let active = true;

    async function pollState() {
      try {
        const response = await fetch('/state', { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        if (!active) {
          return;
        }

        if (Array.isArray(data.angles) && data.angles.length === 5) {
          setAngles(data.angles);
        }

        const status = data.status || 'Tracking';
        const fps = Number(data.fps || 0);
        const live = Boolean(data.camera_ready);

        setCameraReady(live);
        setConnectionStatus(live ? 'Live' : 'Waiting');
        setStateSummary(`${status} | ${fps.toFixed(1)} FPS`);
      } catch (error) {
        if (!active) {
          return;
        }

        setCameraReady(false);
        setConnectionStatus('Disconnected');
        setStateSummary('Could not reach the Python capture loop.');
      }
    }

    pollState();
    const intervalId = window.setInterval(pollState, 60);

    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, []);

  return e(
    'div',
    { className: 'page-shell' },
    e(
      'header',
      { className: 'hero' },
      e(
        'div',
        null,
        e('p', { className: 'eyebrow' }, 'OpenCV to 3D sync'),
        e('h1', null, '5-DOF Robot Arm Mirror'),
        e(
          'p',
          { className: 'lede' },
          'The webcam is processed in Python with MediaPipe and OpenCV. The same live servo angles drive the browser arm.'
        )
      ),
      e(
        'div',
        { className: 'status-card' },
        e('span', { className: 'status-label' }, 'Connection'),
        e('strong', null, connectionStatus),
        e('span', null, stateSummary)
      )
    ),
    e(
      'main',
      { className: 'workspace' },
      e(
        'section',
        { className: 'panel camera-panel' },
        e(
          'div',
          { className: 'panel-header' },
          e('h2', null, 'OpenCV feed'),
          e('span', null, 'Annotated live input')
        ),
        e('img', { id: 'cameraFeed', src: '/video_feed', alt: 'Live OpenCV feed' })
      ),
      e(
        'section',
        { className: 'panel sim-panel' },
        e(
          'div',
          { className: 'panel-header' },
          e('h2', null, '3D robot arm simulation'),
          e('span', null, 'Base, shoulder, elbow, wrist, gripper')
        ),
        e(RobotScene, { angles })
      )
    ),
    e(
      'footer',
      { className: 'telemetry' },
      e(MetricCard, { label: 'Base', value: `${angles[0]} deg`, index: 0 }),
      e(MetricCard, { label: 'Shoulder', value: `${angles[1]} deg`, index: 1 }),
      e(MetricCard, { label: 'Elbow', value: `${angles[2]} deg`, index: 2 }),
      e(MetricCard, { label: 'Wrist', value: `${angles[3]} deg`, index: 3 }),
      e(MetricCard, { label: 'Gripper', value: `${angles[4]} deg`, index: 4 })
    ),
    e('div', { 'data-camera-ready': cameraReady ? 'true' : 'false' })
  );
}

createRoot(document.getElementById('app')).render(e(App));