"""
Wrist + Gripper Control — MediaPipe Tasks API
==============================================
Controls wrist and gripper servos.
  - Wrist  : right arm pose (elbow → wrist → index finger angle)
  - Gripper: pinch distance (thumb tip ↔ index tip)

Servo packet sent to Arduino: [base=90, shoulder=90, elbow=90, wrist, gripper]

Requires:
    pip install mediapipe opencv-python pyserial numpy

Download models:
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
import serial
import serial.tools.list_ports
import time
import math
import numpy as np

# =============================================================================
#  CONFIG
# =============================================================================
PORT                = "COM5"
DEAD_ZONE           = 3
PINCH_CLOSE         = 0.06      # normalised pinch threshold
GRIPPER_OPEN        = 10
GRIPPER_CLOSED      = 170


MIN_PRESENCE        = 0.50

POSE_MODEL_PATH = "pose_landmarker_heavy.task"
HAND_MODEL_PATH = "hand_landmarker.task"

# Pose landmark indices

RW_IDX = 16   # RIGHT_WRIST
RI_IDX = 20   # RIGHT_INDEX fingertip

# =============================================================================
#  KALMAN SMOOTHER
# =============================================================================
class KalmanServo:
    def __init__(self, initial=90.0, process_var=1.2, measure_var=22.0):
        self.x = float(initial)
        self.P = 1.0
        self.Q = process_var
        self.R = measure_var

    def update(self, z):
        P_ = self.P + self.Q
        K  = P_ / (P_ + self.R)
        self.x += K * (z - self.x)
        self.P  = (1 - K) * P_
        return self.x


class SmoothedServo:
    def __init__(self, initial=90, process_var=1.2, measure_var=22.0,
                 dead_zone=DEAD_ZONE, max_vel=15):
        self.kf        = KalmanServo(initial, process_var, measure_var)
        self.prev      = float(initial)
        self.dead_zone = dead_zone
        self.max_vel   = max_vel

    def update(self, raw, active=True):
        if not active:
            return int(self.prev)
        k = self.kf.update(float(raw))
        if abs(k - self.prev) < self.dead_zone:
            k = self.prev
        k = self.prev + float(np.clip(k - self.prev, -self.max_vel, self.max_vel))
        k = int(np.clip(k, 0, 180))
        self.prev = k
        return k


# =============================================================================
#  SERIAL
# =============================================================================
def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = p.description.lower()
        if "arduino" in desc or "usb serial" in desc or "ch340" in desc:
            return p.device
    for p in ports:
        if "bluetooth" not in p.description.lower():
            return p.device
    return None


auto_port = find_arduino_port()
if auto_port:
    PORT = auto_port

try:
    arduino = serial.Serial(PORT, 9600)
    time.sleep(2)
    print(f"Connected to Arduino on {PORT}")
except Exception as e:
    print(f"Serial failed ({e}) — dry-run mode")
    arduino = None


def send(wrist, gripper):
    # Arduino expects: [base, shoulder, elbow, wrist, gripper]
    msg = f"90,90,90,{wrist},{gripper}\n"
    if arduino:
        try:
            arduino.write(msg.encode())
        except Exception:
            pass


# =============================================================================
#  MEDIAPIPE SETUP
# =============================================================================
pose_options = mp_vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False,
)
pose_det = mp_vision.PoseLandmarker.create_from_options(pose_options)

hand_options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
hands_det = mp_vision.HandLandmarker.create_from_options(hand_options)

cap = cv2.VideoCapture(0)
time.sleep(2)

wrist_smoother   = SmoothedServo(initial=90,           measure_var=22.0, max_vel=10)
gripper_smoother = SmoothedServo(initial=GRIPPER_OPEN, measure_var=8.0,  max_vel=25)

frame_timestamp_ms = 0
WRIST_COLOR   = (255, 100, 200)
GRIPPER_COLOR = (100, 255, 255)


# =============================================================================
#  HELPERS
# =============================================================================
def lm_present(lm):
    score = lm.presence if hasattr(lm, 'presence') else lm.visibility
    return score >= MIN_PRESENCE




def draw_wrist_segment(frame, screen_lms, active):
    H, W = frame.shape[:2]
    pts = {i: (int(screen_lms[i].x * W), int(screen_lms[i].y * H))
           for i in [RW_IDX, RI_IDX]}
    color = WRIST_COLOR if active else (50, 50, 50)
    thickness = 3 if active else 1
    cv2.line(frame, pts[RW_IDX], pts[RI_IDX], color, thickness)
    for p in pts.values():
        cv2.circle(frame, p, 6, (255, 255, 255), -1)
        cv2.circle(frame, p, 6, (0,   0,   0),   1)


def draw_hand(frame, hand_lms):
    H, W = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17)
    ]
    pts = [(int(lm.x * W), int(lm.y * H)) for lm in hand_lms]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], GRIPPER_COLOR, 2)
    for p in pts:
        cv2.circle(frame, p, 3, (255, 255, 255), -1)
    cv2.circle(frame, pts[4], 8, (0,   0,   255), -1)   # thumb tip
    cv2.circle(frame, pts[8], 8, (255, 0,   255), -1)   # index tip
    cv2.line(frame,   pts[4], pts[8], (255, 255, 0), 2)


# =============================================================================
#  MAIN LOOP
# =============================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_timestamp_ms += 33

    wrist_raw     = 90
    gripper_raw   = GRIPPER_OPEN
    wrist_active  = False
    hand_detected = False
    pinch_dist    = None

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # -------------------------------------------------------------------------
    #  POSE — wrist angle only
    # -------------------------------------------------------------------------
    pose_result = pose_det.detect_for_video(mp_image, frame_timestamp_ms)

    pose_detected = (
        pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0 and
        pose_result.pose_world_landmarks and len(pose_result.pose_world_landmarks) > 0
    )

    if pose_detected:
        slm = pose_result.pose_landmarks[0]
        wlm = pose_result.pose_world_landmarks[0]

        if lm_present(slm[RW_IDX]) and lm_present(slm[RI_IDX]):
            # Vector from wrist → index fingertip in world space (no elbow needed)
            dx = wlm[RI_IDX].x - wlm[RW_IDX].x
            dy = wlm[RI_IDX].y - wlm[RW_IDX].y
            dz = wlm[RI_IDX].z - wlm[RW_IDX].z
            # Elevation angle: +90 = fingers pointing up, -90 = fingers pointing down
            elevation = math.degrees(math.atan2(-dy, math.sqrt(dx**2 + dz**2)))
            # Map -90°..+90° → 0..180 servo
            wrist_raw    = int(np.clip((elevation + 90), 0, 180))
            wrist_active = True
            draw_wrist_segment(frame, slm, active=True)
        else:
            if pose_detected:
                draw_wrist_segment(frame, pose_result.pose_landmarks[0], active=False)

    # -------------------------------------------------------------------------
    #  HAND — gripper only
    # -------------------------------------------------------------------------
    hand_result = hands_det.detect_for_video(mp_image, frame_timestamp_ms)

    if hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
        lm            = hand_result.hand_landmarks[0]
        hand_detected = True
        pinch_dist    = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        gripper_raw   = GRIPPER_CLOSED if pinch_dist < PINCH_CLOSE else GRIPPER_OPEN
        draw_hand(frame, lm)

    # -------------------------------------------------------------------------
    #  SMOOTH + SEND
    # -------------------------------------------------------------------------
    wrist_smooth   = wrist_smoother.update(wrist_raw,   active=wrist_active)
    gripper_smooth = gripper_smoother.update(gripper_raw)
    send(wrist_smooth, gripper_smooth)

    # -------------------------------------------------------------------------
    #  DEBUG OVERLAY
    # -------------------------------------------------------------------------
    # Wrist
    wrist_tag = "" if wrist_active else "  ~HOLD"
    cv2.putText(frame, f"Wrist:   {wrist_smooth:>3}°{wrist_tag}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WRIST_COLOR, 2)

    # Gripper
    state_label = "CLOSED" if gripper_smooth > 90 else "OPEN"
    cv2.putText(frame, f"Gripper: {gripper_smooth:>3}°  [{state_label}]",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GRIPPER_COLOR, 2)

    if pinch_dist is not None:
        cv2.putText(frame, f"Pinch: {pinch_dist:.3f}  (thresh {PINCH_CLOSE})",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)

    # Servo bars
    cv2.rectangle(frame, (10, 108), (10 + wrist_raw,    118), (80, 80, 80),   -1)
    cv2.rectangle(frame, (10, 120), (10 + wrist_smooth, 130), WRIST_COLOR,    -1)
    cv2.rectangle(frame, (10, 134), (10 + gripper_raw,    144), (80, 80, 80), -1)
    cv2.rectangle(frame, (10, 146), (10 + gripper_smooth, 156), GRIPPER_COLOR,-1)
    cv2.putText(frame, "W", (195, 117),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
    cv2.putText(frame, "W", (195, 129),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, WRIST_COLOR,   1)
    cv2.putText(frame, "G", (195, 143),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
    cv2.putText(frame, "G", (195, 155),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRIPPER_COLOR, 1)

    # Status
    pose_status = "Pose: tracking" if pose_detected else "Pose: searching..."
    cv2.putText(frame, pose_status,
                (10, frame.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 100) if pose_detected else (0, 100, 255), 2)
    hand_status = "Hand: tracking" if hand_detected else "Hand: searching..."
    cv2.putText(frame, hand_status,
                (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 100) if hand_detected else (0, 100, 255), 2)

    cv2.imshow("Wrist + Gripper Control  [ESC = quit]", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
pose_det.close()
hands_det.close()
if arduino:
    try:
        arduino.close()
    except Exception:
        pass
cv2.destroyAllWindows()