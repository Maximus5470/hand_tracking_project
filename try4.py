"""
Robot Arm Control — MediaPipe Tasks API (new, replaces mp.solutions)
====================================================================
Requires:
    pip install mediapipe opencv-python pyserial numpy
    + download model files (see comments below)

Servo order sent to Arduino: [base, shoulder, elbow, wrist, gripper]
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
PORT            = "COM5"
DEAD_ZONE       = 2
MAX_VEL         = 7
HAND_Z_NEAR     = -0.20
HAND_Z_FAR      = 0.12
ARM_Z_NEAR      = -0.25
ARM_Z_FAR       = 0.25
MIN_PRESENCE    = 0.50      # replaces "visibility" in new API
WRIST_NEUTRAL_ANGLE = 165.0
WRIST_GAIN      = 3

# Model file paths — download these once:
# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
POSE_MODEL_PATH = "pose_landmarker_heavy.task"
HAND_MODEL_PATH = "hand_landmarker.task"

# Pose landmark indices (new API uses plain integers, same values as before)
LS_IDX = 11   # LEFT_SHOULDER (Collarbone base)
RS_IDX = 12   # RIGHT_SHOULDER
RE_IDX = 14   # RIGHT_ELBOW
RW_IDX = 16   # RIGHT_WRIST
RI_IDX = 20   # RIGHT_INDEX (finger tip)

# =============================================================================
#  KALMAN SMOOTHER (unchanged)
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


class SmoothedServoArray:
    def __init__(self, n=5, initial=90, process_var=1.2, measure_var=22.0,
                 dead_zone=DEAD_ZONE, max_vel=MAX_VEL):
        self.kf        = [KalmanServo(initial, process_var, measure_var) for _ in range(n)]
        self.prev      = [float(initial)] * n
        self.dead_zone = dead_zone
        self.max_vel   = max_vel

    def update(self, raw, mask=None):
        if mask is None:
            mask = [True] * len(raw)
        out = []
        for i, (r, ok) in enumerate(zip(raw, mask)):
            if not ok:
                out.append(int(self.prev[i]))
                continue
            k = self.kf[i].update(float(r))
            if abs(k - self.prev[i]) < self.dead_zone:
                k = self.prev[i]
            k = self.prev[i] + float(
                np.clip(k - self.prev[i], -self.max_vel, self.max_vel))
            k = int(np.clip(k, 0, 180))
            self.prev[i] = k
            out.append(k)
        return out


# =============================================================================
#  SERIAL
# =============================================================================
def find_arduino_port():
    """Automatically find the Arduino port."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        # Check description for common Arduino strings
        desc = p.description.lower()
        if "arduino" in desc or "usb serial" in desc or "ch340" in desc:
            return p.device
    
    # If no obvious Arduino, return the first available COM port that isn't Bluetooth
    for p in ports:
        if "bluetooth" not in p.description.lower():
            return p.device
            
    return None


auto_port = find_arduino_port()
if auto_port:
    PORT = auto_port


# =============================================================================
#  MEDIAPIPE TASKS SETUP
# =============================================================================
# --- Pose Landmarker ---
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

# --- Hand Landmarker ---
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
hands_det = mp_vision.HandLandmarker.create_from_options(hand_options)

cap      = cv2.VideoCapture(0)
time.sleep(2)

try:
    arduino = serial.Serial(PORT, 9600)
    time.sleep(2)
    print(f"Connected to Arduino on {PORT}")
except Exception as e:
    print(f"Serial failed ({e}) — dry-run mode")
    arduino = None

smoother = SmoothedServoArray(n=5, process_var=1.2, measure_var=22.0)
frame_timestamp_ms = 0

# Drawing connections for hands (new API)
HAND_CONNECTIONS = mp_vision.HandLandmarker.HAND_CONNECTIONS if hasattr(
    mp_vision.HandLandmarker, 'HAND_CONNECTIONS') else []


# =============================================================================
#  HELPERS
# =============================================================================
def angle_3d(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z], dtype=float)
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z], dtype=float)
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0
    cos = np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos))


def map_range(val, in_min, in_max, out_min, out_max):
    val = max(in_min, min(in_max, val))
    return int((val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


def base_from_hand_depth(z):
    return map_range(z, HAND_Z_NEAR, HAND_Z_FAR, 0, 180)


def base_from_arm_depth(z):
    return map_range(z, ARM_Z_NEAR, ARM_Z_FAR, 0, 180)



def send(data):
    msg = ",".join(map(str, data)) + "\n"
    if arduino:
        try:
            arduino.write(msg.encode())
        except Exception:
            pass


def lm_present(lm, threshold=MIN_PRESENCE):
    """New API uses presence_score instead of visibility."""
    return (lm.presence if hasattr(lm, 'presence') else lm.visibility) >= threshold


def draw_pose_arm(frame, screen_lms, channel_colors):
    """Draw arm skeleton from normalized screen landmarks using channel colors."""
    H, W = frame.shape[:2]
    ids = [LS_IDX, RS_IDX, RE_IDX, RW_IDX, RI_IDX]
    pts = {}
    for i in ids:
        lm = screen_lms[i]
        pts[i] = (int(lm.x * W), int(lm.y * H), lm_present(lm))

    # Segment mapping to channels:
    # (Left Shoulder, Right Shoulder) -> Base channel color (Collarbone)
    # (Shoulder, Elbow) -> Shoulder channel color
    # (Elbow, Wrist)    -> Elbow channel color
    # (Wrist, Finger)   -> Wrist channel color
    segments = [
        (LS_IDX, RS_IDX, channel_colors[0]),
        (RS_IDX, RE_IDX, channel_colors[1]),
        (RE_IDX, RW_IDX, channel_colors[2]),
        (RW_IDX, RI_IDX, channel_colors[3])
    ]

    for (a, b, color) in segments:
        xa, ya, ca = pts[a]
        xb, yb, cb = pts[b]
        ok = ca and cb
        line_color = color if ok else (50, 50, 50)
        cv2.line(frame, (xa, ya), (xb, yb), line_color, 4 if ok else 1)

    for i, (x, y, ok) in pts.items():
        cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 6, (0, 0, 0), 1)


def draw_hand(frame, hand_screen_lms, color):
    """Draw hand landmarks from normalized screen coordinates using gripper color."""
    H, W = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17)
    ]
    pts = [(int(lm.x * W), int(lm.y * H)) for lm in hand_screen_lms]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame, p, 3, (255, 255, 255), -1)


# =============================================================================
#  MAIN LOOP
# =============================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_timestamp_ms += 33   # ~30fps; use actual elapsed ms for accuracy

    base_servo     = 90
    shoulder_servo = 90
    elbow_servo    = 90
    wrist_servo    = 90
    gripper        = 90
    arm_depth_z    = None
    hand_depth_z   = None
    mask = [False, False, False, False, True]

    # =========================================================================
    #  SETUP TOOLS
    # =========================================================================
    labels = ["Base", "Shoulder", "Elbow", "Wrist", "Gripper"]
    colors = [(0,200,255), (0,255,0), (255,200,0), (255,100,200), (100,255,255)]

    # Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # =========================================================================
    #  STEP 1 — Pose Landmarker (VIDEO mode uses timestamp)
    # =========================================================================
    pose_result = pose_det.detect_for_video(mp_image, frame_timestamp_ms)

    pose_detected = (
        pose_result.pose_landmarks and
        len(pose_result.pose_landmarks) > 0 and
        pose_result.pose_world_landmarks and
        len(pose_result.pose_world_landmarks) > 0
    )

    if pose_detected:
        slm = pose_result.pose_landmarks[0]       # screen (normalized x,y)
        wlm = pose_result.pose_world_landmarks[0] # world (metres x,y,z)

        s_ls = slm[LS_IDX]; s_sh = slm[RS_IDX]
        s_el  = slm[RE_IDX]; s_wr = slm[RW_IDX]; s_idx = slm[RI_IDX]

        w_ls = wlm[LS_IDX]; w_sh = wlm[RS_IDX]
        w_el  = wlm[RE_IDX]; w_wr = wlm[RW_IDX]; w_idx = wlm[RI_IDX]

        # BASE from arm depth in world space: closer to camera -> different z
        if lm_present(s_ls) and lm_present(s_sh) and lm_present(s_el):
            arm_depth_z = float(np.mean([w_el.z, w_wr.z]) - w_sh.z)
            base_servo  = base_from_arm_depth(arm_depth_z)
            mask[0]     = True

        # SHOULDER (calculated relative to the collarbone/shoulders)
        if lm_present(s_ls) and lm_present(s_sh) and lm_present(s_el):
            ang            = angle_3d(w_ls, w_sh, w_el)
            shoulder_servo = map_range(ang, 20, 160, 170, 10)
            mask[1]        = True

        # ELBOW
        if lm_present(s_sh) and lm_present(s_el) and lm_present(s_wr):
            ang         = angle_3d(w_sh, w_el, w_wr)
            elbow_servo = map_range(ang, 20, 160, 10, 170)
            mask[2]     = True

        # WRIST
        if lm_present(s_el) and lm_present(s_wr) and lm_present(s_idx):
            ang         = angle_3d(w_el, w_wr, w_idx)
            wrist_servo = int(np.clip(90 + (ang - WRIST_NEUTRAL_ANGLE) * WRIST_GAIN, 0, 180))
            mask[3]     = True

        draw_pose_arm(frame, slm, colors)

    # =========================================================================
    #  STEP 2 — Hand Landmarker (gripper only)
    # =========================================================================
    hand_result = hands_det.detect_for_video(mp_image, frame_timestamp_ms)

    if hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
        hlm = hand_result.hand_landmarks[0]   # screen landmarks
        lm  = hlm

        hand_depth_z = float(np.mean([lm[0].z, lm[5].z, lm[9].z, lm[13].z, lm[17].z]))

        dist    = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        gripper = 170 if dist < 0.06 else 10
        mask[4] = True

        draw_hand(frame, lm, colors[4])

    # =========================================================================
    #  SMOOTH + SEND
    # =========================================================================
    raw    = [base_servo, shoulder_servo, elbow_servo, wrist_servo, gripper]
    angles = smoother.update(raw, mask=mask)
    send(angles)

    # =========================================================================
    #  DEBUG OVERLAY
    # =========================================================================

    for i, (label, val) in enumerate(zip(labels, angles)):
        tag = "" if mask[i] else "  ~HOLD"
        cv2.putText(frame, f"{label}: {val:>3}°{tag}",
                    (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

    cv2.putText(frame, f"Raw Base: {raw[0]:>3}°",
                (10, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    cv2.rectangle(frame, (10, 188), (10 + raw[0],    198), (80,80,80),    -1)
    cv2.rectangle(frame, (10, 200), (10 + angles[0], 210), (0, 200, 255), -1)

    pose_status = "Pose: tracking" if pose_detected else "Pose: searching..."
    cv2.putText(frame, pose_status, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 100) if pose_detected else (0, 100, 255), 2)

    if arm_depth_z is not None:
        cv2.putText(frame, f"Arm Z: {arm_depth_z:+.3f}", (10, 232),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)

    if hand_depth_z is not None:
        cv2.putText(frame, f"Hand Z: {hand_depth_z:+.3f}", (10, 255),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)

    cv2.imshow("Robot Arm Control  [ESC = quit]", frame)
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