import serial
import serial.tools.list_ports
import time
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# config
write_video = True
debug = True
cam_source = 0  # use laptop's default webcam (device 0)

ser = None

def open_serial_interactive():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found. Running in debug mode (no serial).")
        return None
    print("Available serial ports:")
    for i, p in enumerate(ports):
        print(f"{i}: {p.device} - {p.description}")
    try:
        sel = input("Select port number (or press Enter to skip): ")
        if sel.strip() == "":
            print("No port selected. Running in debug mode.")
            return None
        idx = int(sel)
        port = ports[idx].device
        s = serial.Serial(port, 9600)
        time.sleep(2)
        print(f"Opened serial port {port} at 9600")
        return s
    except Exception as e:
        print("Failed to open serial port:", e)
        return None

# Always offer to open a serial port so motors can be connected without changing `debug`
ser = open_serial_interactive()

POSE_MODEL_PATH = "pose_landmarker_heavy.task"
HAND_MODEL_PATH = "hand_landmarker.task"
ARM_Z_NEAR = -0.25
ARM_Z_FAR = 0.25

x_min = 0
x_mid = 75
x_max = 150
# use angle between wrist and index finger to control x axis
palm_angle_min = -50
palm_angle_mid = 20

y_min = 0
y_mid = 90
y_max = 170
# use wrist y to control y axis
wrist_y_min = 0.3
wrist_y_max = 0.9

# Elbow (servo 2) range: 0-110
z_min = 0
z_mid = 55
z_max = 110
# use palm size to control z axis
plam_size_min = 0.1
plam_size_max = 0.3

claw_open_angle = 60
claw_close_angle = 0
wrist_min = 10
wrist_max = 120
pinch_close_ratio = 0.18
pinch_open_ratio = 0.55

# We'll produce five motor angles to match `base.py` motor order:
# [Base, Shoulder, Elbow, Wrist, Gripper]
# and map them to the same channel layout used by `base.py`:
# CH0 -> Base, CH4 -> Shoulder, CH2 -> Elbow, CH1 -> Wrist, CH3 -> Gripper
wrist_mid = 65
servo_angle = [x_mid, y_mid, z_mid, wrist_mid, claw_open_angle]
prev_servo_angle = None
channel_map = [0, 4, 2, 1, 3]

def build_channel_message(angles_list):
    channel_angles = [90] * len(angles_list)
    for motor_index, channel_index in enumerate(channel_map):
        channel_angles[channel_index] = angles_list[motor_index]
    return ",".join(str(angle) for angle in channel_angles) + "\n"
fist_threshold = 7


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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

cap = cv2.VideoCapture(cam_source)

# video writer
if write_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 60.0, (640, 480))

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
def map_range(x, in_min, in_max, out_min, out_max):
    # robust float mapping, returns int
    if in_max == in_min:
        return int(out_min)
    val = (float(x) - float(in_min)) * (float(out_max) - float(out_min)) / (float(in_max) - float(in_min)) + float(out_min)
    return int(round(val))


def base_from_arm_depth(z):
    return map_range(z, ARM_Z_NEAR, ARM_Z_FAR, z_min, z_max)


def neutral_servo_angle():
    return [x_mid, y_mid, z_mid, wrist_mid, claw_open_angle]

# Check if the hand is a fist
def is_fist(hand_landmarks, palm_size):
    # calculate the distance between the wrist and the each finger tip
    distance_sum = 0
    WRIST = hand_landmarks.landmark[0]
    for i in [7,8,11,12,15,16,19,20]:
        distance_sum += ((WRIST.x - hand_landmarks.landmark[i].x)**2 + \
                         (WRIST.y - hand_landmarks.landmark[i].y)**2 + \
                         (WRIST.z - hand_landmarks.landmark[i].z)**2)**0.5
    return distance_sum/palm_size < fist_threshold

def landmark_to_servo_angle(hand_landmarks):
    # Adapted to produce five motor angles: [Base, Shoulder, Elbow, Wrist, Gripper]
    servo_angle = [x_mid, y_mid, z_mid, wrist_mid, claw_open_angle]
    WRIST = hand_landmarks.landmark[0]
    INDEX_FINGER_MCP = hand_landmarks.landmark[5]
    THUMB_TIP = hand_landmarks.landmark[4]
    INDEX_FINGER_TIP = hand_landmarks.landmark[8]
    PINKY_MCP = hand_landmarks.landmark[17]
    # calculate the distance between the wrist and the index finger
    palm_size = ((WRIST.x - INDEX_FINGER_MCP.x)**2 + (WRIST.y - INDEX_FINGER_MCP.y)**2 + (WRIST.z - INDEX_FINGER_MCP.z)**2)**0.5

    # Gripper (motor 4) - pinch distance controls open/close
    pinch_distance = ((THUMB_TIP.x - INDEX_FINGER_TIP.x)**2 +
                      (THUMB_TIP.y - INDEX_FINGER_TIP.y)**2 +
                      (THUMB_TIP.z - INDEX_FINGER_TIP.z)**2)**0.5
    if palm_size > 0:
        pinch_ratio = pinch_distance / palm_size
    else:
        pinch_ratio = pinch_open_ratio
    servo_angle[4] = map_range(clamp(pinch_ratio, pinch_close_ratio, pinch_open_ratio),
                               pinch_close_ratio, pinch_open_ratio,
                               claw_close_angle, claw_open_angle)

    # For compatibility with `base.py` mapping, use the palm center to map X/Y to base/shoulder
    # Base: map landmark 9.x from [0,1] -> [x_min,x_max]
    # Shoulder: map wrist y from [wrist_y_min, wrist_y_max] -> [y_max, y_min]
    PALM_CENTER_X = sum(hand_landmarks.landmark[i].x for i in [0, 5, 9, 13, 17]) / 5.0
    WRIST_Y = hand_landmarks.landmark[0].y
    servo_angle[0] = map_range(PALM_CENTER_X, 0.0, 1.0, x_min, x_max)
    servo_angle[1] = map_range(clamp(WRIST_Y, wrist_y_min, wrist_y_max),
                               wrist_y_min, wrist_y_max,
                               y_max, y_min)

    # Wrist motor (motor 3) - palm roll from knuckle line
    knuckle_roll = math.degrees(math.atan2(INDEX_FINGER_MCP.y - PINKY_MCP.y,
                                           INDEX_FINGER_MCP.x - PINKY_MCP.x))
    servo_angle[3] = clamp(map_range(knuckle_roll, -90.0, 90.0, wrist_min, wrist_max),
                           wrist_min, wrist_max)

    servo_angle = [int(i) for i in servo_angle]
    return servo_angle


def hand_is_valid(hand_landmarks):
    # Treat out-of-frame or badly tracked hands as invalid so the arm can return home.
    for idx in [0, 5, 9, 13, 17]:
        lm = hand_landmarks.landmark[idx]
        if lm.x < 0.0 or lm.x > 1.0 or lm.y < 0.0 or lm.y > 1.0:
            return False
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    palm_size = ((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2 + (wrist.z - middle_mcp.z)**2)**0.5
    return 0.02 <= palm_size <= 0.4


def elbow_from_pose_depth(pose_world_landmarks):
    if not pose_world_landmarks:
        return z_mid

    pose = pose_world_landmarks[0]
    right_shoulder = pose[12]
    right_elbow = pose[14]
    right_wrist = pose[16]
    arm_depth = ((right_elbow.z + right_wrist.z) / 2.0) - right_shoulder.z
    return base_from_arm_depth(arm_depth)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_image)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        frame_timestamp_ms = int(time.time() * 1000)
        pose_results = pose_det.detect_for_video(mp_image, frame_timestamp_ms)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        current_servo_angle = neutral_servo_angle()
        if pose_results.pose_world_landmarks and len(pose_results.pose_world_landmarks) > 0:
            current_servo_angle[2] = elbow_from_pose_depth(pose_results.pose_world_landmarks)

        if hand_results.multi_hand_landmarks:
            if len(hand_results.multi_hand_landmarks) == 1:
                # print("One hand detected")
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                if hand_is_valid(hand_landmarks):
                    hand_angles = landmark_to_servo_angle(hand_landmarks)
                    current_servo_angle[0] = hand_angles[0]
                    current_servo_angle[1] = hand_angles[1]
                    current_servo_angle[3] = hand_angles[3]
                    current_servo_angle[4] = hand_angles[4]

                if current_servo_angle != prev_servo_angle:
                    msg = build_channel_message(current_servo_angle)
                    print("Servo angle: ", current_servo_angle)
                    print("Outgoing message:", msg.strip())
                    prev_servo_angle = list(current_servo_angle)
                    if ser is not None:
                        try:
                            ser.write(msg.encode())
                        except Exception as e:
                            print("Failed to write to serial:", e)
            else:
                print("More than one hand detected")
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        # show servo angle
        cv2.putText(image, str(current_servo_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('MediaPipe Hands', image)

        if write_video:
            out.write(image)
        if cv2.waitKey(5) & 0xFF == 27:
            if write_video:
                out.release()
            break
cap.release()