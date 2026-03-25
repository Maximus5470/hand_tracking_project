import cv2
import mediapipe as mp
import serial
import time
import math

# ===== SERIAL SETUP =====
PORT = "COM4"   # CHANGE THIS
arduino = serial.Serial(PORT, 9600)
time.sleep(2)

# ===== MEDIAPIPE =====
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ===== FILTER =====
prev_angles = [90, 90, 90, 90, 90]
alpha = 0.2

# ===== HELPERS =====
def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]

    angle = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) -
        math.atan2(a[1]-b[1], a[0]-b[0])
    )

    if angle < 0:
        angle += 360

    return angle

def map_range(val, in_min, in_max, out_min, out_max):
    val = max(in_min, min(in_max, val))
    return int((val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def smooth(new_angles):
    global prev_angles
    smoothed = []
    for i in range(len(new_angles)):
        val = int(prev_angles[i]*alpha + new_angles[i]*(1-alpha))
        smoothed.append(val)
    prev_angles = smoothed
    return smoothed

def send(data):
    msg = ",".join(map(str, data)) + "\n"
    arduino.write(msg.encode())

# ===== MAIN LOOP =====
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_result = pose.process(rgb)
    hand_result = hands.process(rgb)

    wrist_servo = 90
    shoulder_servo = 90
    elbow_servo = 90
    base_servo = 90
    gripper = 90

    # ===== POSE TRACKING =====
    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks.landmark

        shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Angles
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        shoulder_angle = calculate_angle(elbow, shoulder, wrist)

        # Mapping
        elbow_servo = map_range(elbow_angle, 30, 160, 10, 170)
        shoulder_servo = map_range(shoulder_angle, 30, 160, 10, 170)

        base_servo = int(wrist.x * 180)
        wrist_servo = int(wrist.y * 180)

        mp_draw.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # ===== HAND TRACKING (GRIP CONTROL) =====
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            # distance between thumb tip & index tip
            thumb = lm[4]
            index = lm[8]

            dist = math.hypot(thumb.x - index.x, thumb.y - index.y)

            if dist < 0.05:
                gripper = 180   # CLOSE
            else:
                gripper = 90    # OPEN

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ===== COMBINE =====
    angles = [wrist_servo, shoulder_servo, elbow_servo, base_servo, gripper]

    # Smooth motion
    angles = smooth(angles)

    # Send to Arduino
    send(angles)

    # Debug
    cv2.putText(frame, f"{angles}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("AI Arm Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()