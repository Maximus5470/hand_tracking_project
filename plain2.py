import cv2
import mediapipe as mp
import serial
import time
import math

# ===== SERIAL SETUP =====
PORT = "COM4"   # CHANGE THIS
try:
    arduino = serial.Serial(PORT, 9600)
    time.sleep(2)
except Exception as e:
    print(f"Warning: Could not open {PORT}. Running in dry-run mode. Error: {e}")
    arduino = None

# ===== MEDIAPIPE =====
mp_pose  = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
)
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ===== ARM LINK LENGTHS (tune to your robot) =====
L1 = 13.0   # Upper arm
L2 = 12.0   # Forearm

# ===== WHITELISTED LANDMARK INDICES =====
# --- Pose: ONLY these 3 indices drive the arm motors ---
RIGHT_SHOULDER_IDX = mp_pose.PoseLandmark.RIGHT_SHOULDER.value   # 12
RIGHT_ELBOW_IDX    = mp_pose.PoseLandmark.RIGHT_ELBOW.value      # 14
RIGHT_WRIST_IDX    = mp_pose.PoseLandmark.RIGHT_WRIST.value      # 16

POSE_WHITELIST = {RIGHT_SHOULDER_IDX, RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX}

ALLOWED_POSE_CONNECTIONS = [
    (RIGHT_SHOULDER_IDX, RIGHT_ELBOW_IDX),
    (RIGHT_ELBOW_IDX,    RIGHT_WRIST_IDX),
    (RIGHT_WRIST_IDX,    RIGHT_SHOULDER_IDX), # Completes the visual triangle pattern
]

# --- Hand: ONLY these 2 indices drive the gripper ---
THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8

HAND_WHITELIST = {THUMB_TIP_IDX, INDEX_TIP_IDX}

ALLOWED_HAND_CONNECTIONS = [
    (THUMB_TIP_IDX, INDEX_TIP_IDX),
]

# ===== DRAWING STYLES =====
DOT_COLOR  = (0, 255, 0)   # Green dots for whitelisted points
LINE_COLOR = (0, 0, 255)   # Red lines for connections
DOT_RADIUS = 8
LINE_THICK = 2


def draw_whitelisted_pose(frame, landmarks, h, w):
    """Draw ONLY the 3 whitelisted pose landmarks and 2 connections. Nothing else."""
    pts = {}
    for idx in POSE_WHITELIST:
        lm = landmarks[idx]
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            pts[idx] = (cx, cy)
            cv2.circle(frame, (cx, cy), DOT_RADIUS, DOT_COLOR, -1)

    for (a, b) in ALLOWED_POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], LINE_COLOR, LINE_THICK)


def draw_whitelisted_hand(frame, landmarks, h, w):
    """Draw ONLY thumb tip (4) and index tip (8) and the line between them."""
    pts = {}
    for idx in HAND_WHITELIST:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        pts[idx] = (cx, cy)
        cv2.circle(frame, (cx, cy), DOT_RADIUS, DOT_COLOR, -1)

    for (a, b) in ALLOWED_HAND_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], LINE_COLOR, LINE_THICK)


# ===== SMOOTHING =====
prev_servo      = [90, 90, 90, 90]   # [wrist, shoulder, elbow, gripper]
ALPHA           = 0.80
MAX_DELTA       = 2

prev_base_speed = 0
BASE_DEAD_ZONE  = 0.008
BASE_SCALE      = 220
BASE_ALPHA      = 0.75
prev_wrist_x    = None


# ===== HELPERS =====
def ik_2d(dx, dy, l1, l2):
    dist = math.hypot(dx, dy)
    dist = min(dist, l1 + l2 - 0.01)

    cos_elbow = (dist**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow_deg = math.degrees(math.acos(cos_elbow))

    phi  = math.atan2(dy, dx)
    beta = math.acos(max(-1.0, min(1.0,
           (l1**2 + dist**2 - l2**2) / (2 * l1 * dist))))
    shoulder_deg = math.degrees(phi + beta)

    return shoulder_deg, elbow_deg


def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c.y-b.y, c.x-b.x) -
        math.atan2(a.y-b.y, a.x-b.x)
    )
    if angle < 0:
        angle += 360
    return angle

def map_range(val, in_min, in_max, out_min, out_max):
    val = max(in_min, min(in_max, val))
    return int((val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


def rate_limit(new_val, old_val, max_d):
    delta = max(-max_d, min(max_d, new_val - old_val))
    return old_val + delta


def smooth_servos(new_vals):
    global prev_servo
    result = []
    for i, nv in enumerate(new_vals):
        limited  = rate_limit(nv, prev_servo[i], MAX_DELTA)
        smoothed = int(ALPHA * prev_servo[i] + (1 - ALPHA) * limited)
        result.append(smoothed)
    prev_servo = result
    return result


def smooth_base(raw_speed):
    global prev_base_speed
    s = int(BASE_ALPHA * prev_base_speed + (1 - BASE_ALPHA) * raw_speed)
    prev_base_speed = s
    return s


def send(shoulder, gripper, elbow, wrist, base_spd):
    msg = f"{shoulder},{gripper},{elbow},{wrist},{base_spd}\n"
    if arduino:
        arduino.write(msg.encode())


# ===== MAIN LOOP =====
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_result = pose.process(rgb)
    hand_result = hands.process(rgb)

    # Safe defaults — hold last known position
    wrist_servo    = prev_servo[0]
    shoulder_servo = prev_servo[1]
    elbow_servo    = prev_servo[2]
    gripper        = prev_servo[3]
    raw_base_speed = 0

    # ===== POSE — ONLY whitelisted indices (12, 14, 16) used =====
    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks.landmark

        # Hard-coded index access — no other landmark can accidentally be read
        shoulder = lm[RIGHT_SHOULDER_IDX]   # 12 only
        elbow_lm = lm[RIGHT_ELBOW_IDX]      # 14 only
        wrist    = lm[RIGHT_WRIST_IDX]      # 16 only

        # Visibility gate — ignore if any whitelisted point is uncertain
        if (shoulder.visibility > 0.5 and
            elbow_lm.visibility > 0.5 and
            wrist.visibility    > 0.5):

            # Direct Angle Calculation (from movement patterns)
            # Calculates visual angles of the arm joints
            elbow_angle = calculate_angle(shoulder, elbow_lm, wrist)
            shoulder_angle = calculate_angle(elbow_lm, shoulder, wrist)

            # Map the visual angles to servo limits
            elbow_servo    = map_range(elbow_angle, 30, 160, 10, 170)
            shoulder_servo = map_range(shoulder_angle, 30, 160, 10, 170)
            
            # Wrist moves based on vertical screen space
            wrist_servo    = map_range(wrist.y, 0.2, 0.9, 10, 170)

            # Base motor — velocity-based, stops when wrist is still
            if prev_wrist_x is not None:
                delta_x = wrist.x - prev_wrist_x
                if abs(delta_x) > BASE_DEAD_ZONE:
                    raw_base_speed = max(-255, min(255, int(delta_x * BASE_SCALE)))
                else:
                    raw_base_speed = 0   # Stationary → full stop

            prev_wrist_x = wrist.x

        # Draw ONLY the 3 whitelisted pose points — full body skeleton suppressed
        draw_whitelisted_pose(frame, lm, h, w)

    # ===== HAND — ONLY thumb tip (4) and index tip (8) used =====
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            # Hard-coded index access — no other hand landmark is touched
            thumb = lm[THUMB_TIP_IDX]   # 4 only
            index = lm[INDEX_TIP_IDX]   # 8 only

            dist    = math.hypot(thumb.x - index.x, thumb.y - index.y)
            gripper = 180 if dist < 0.05 else 90   # 180 = closed, 90 = open

            # Draw ONLY thumb tip and index tip — full hand skeleton suppressed
            draw_whitelisted_hand(frame, lm, h, w)

    # ===== SMOOTH & SEND =====
    servo_vals = smooth_servos([wrist_servo, shoulder_servo, elbow_servo, gripper])
    base_cmd   = smooth_base(raw_base_speed)

    # map base_cmd (-255 to 255) to a continuous servo speed (neutral is 104, limit is 70 to 140)
    mapped_base_cmd = 104 + int(base_cmd * (54 / 255.0))
    mapped_base_cmd = max(70, min(140, mapped_base_cmd))

    # Send exactly in the order the Arduino expects (from plain.py):
    # Shoulder (Ch 0), Hand Speed / Gripper (Ch 4), Elbow (Ch 2), Wrist (Ch 1), Base (Ch 3)
    send(servo_vals[1], servo_vals[3], servo_vals[2], servo_vals[0], mapped_base_cmd)

    # ===== DEBUG OVERLAY =====
    labels = [
        "Shoulder (Ch 0)",
        "Hand Speed (Ch 4)",
        "Elbow (Ch 2)",
        "Wrist (Ch 1)",
        "Base (Ch 3)"
    ]
    
    display_vals = [
        servo_vals[1], # Shoulder
        servo_vals[3], # Hand Speed (Gripper)
        servo_vals[2], # Elbow
        servo_vals[0], # Wrist
        base_cmd       # Base
    ]

    for i, (lbl, val) in enumerate(zip(labels, display_vals)):
        text = f"{lbl}: {val}deg" if "Base" not in lbl and "Hand Speed" not in lbl else f"{lbl}: {val}"
        color = (0, 255, 0) if "Base" not in lbl else (0, 200, 255)
        
        cv2.putText(frame, text,
                    (10, 35 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    cv2.imshow("AI Arm Control v3", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        if arduino:
            arduino.write(b"90,90,90,90,104\n")   # safe-stop all motors on exit (104 is base neutral)
        break

cap.release()
if arduino:
    arduino.close()
cv2.destroyAllWindows()