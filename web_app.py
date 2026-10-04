import math
from pathlib import Path
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, send_file
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode


app = Flask(__name__)
APP_JS_PATH = Path(app.root_path) / "static" / "app.js"

CAM_SOURCE = 0
POSE_MODEL_PATH = "pose_landmarker_heavy.task"
HAND_MODEL_PATH = "hand_landmarker.task"
ARM_Z_NEAR = -0.25
ARM_Z_FAR = 0.25

WRITE_VIDEO = False
VIDEO_OUTPUT_PATH = "output.mp4"
is_recording_active = False
video_writer = None
writer_lock = threading.Lock()

x_min = 0
x_mid = 100
x_max = 200
palm_angle_min = -50
palm_angle_mid = 20

y_min = 0
y_mid = 105
y_max = 210
wrist_y_min = 0.3
wrist_y_max = 0.9

z_min = 0
z_mid = 75
z_max = 150
plam_size_min = 0.1
plam_size_max = 0.3

claw_open_angle = 60
claw_close_angle = 0
wrist_min = 10
wrist_max = 120
pinch_close_ratio = 0.18
pinch_open_ratio = 0.55

wrist_mid = 65
channel_map = [0, 4, 2, 1, 3]
USER_MOTION_MULTIPLIER = 2


def build_channel_message(angles_list):
    channel_angles = [90] * len(angles_list)
    for motor_index, channel_index in enumerate(channel_map):
        channel_angles[channel_index] = angles_list[motor_index]
    return ",".join(str(angle) for angle in channel_angles) + "\n"


clamp = lambda n, minn, maxn: max(min(maxn, n), minn)


def map_range(x, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return int(out_min)
    val = (float(x) - float(in_min)) * (float(out_max) - float(out_min)) / (float(in_max) - float(in_min)) + float(out_min)
    return int(round(val))


def base_from_arm_depth(z):
    return map_range(z, ARM_Z_NEAR, ARM_Z_FAR, z_min, z_max)


def amplify_motion(angle, neutral_angle, minimum, maximum):
    amplified = neutral_angle + (float(angle) - float(neutral_angle)) * USER_MOTION_MULTIPLIER
    return clamp(int(round(amplified)), minimum, maximum)


def neutral_servo_angle():
    return [x_mid, y_mid, z_mid, wrist_mid, claw_open_angle]


def hand_is_valid(hand_landmarks):
    for idx in [0, 5, 9, 13, 17]:
        lm = hand_landmarks.landmark[idx]
        if lm.x < 0.0 or lm.x > 1.0 or lm.y < 0.0 or lm.y > 1.0:
            return False
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    palm_size = ((wrist.x - middle_mcp.x) ** 2 + (wrist.y - middle_mcp.y) ** 2 + (wrist.z - middle_mcp.z) ** 2) ** 0.5
    return 0.02 <= palm_size <= 0.4


def landmark_to_servo_angle(hand_landmarks):
    servo_angle = [x_mid, y_mid, z_mid, wrist_mid, claw_open_angle]
    wrist = hand_landmarks.landmark[0]
    index_finger_mcp = hand_landmarks.landmark[5]
    thumb_tip = hand_landmarks.landmark[4]
    index_finger_tip = hand_landmarks.landmark[8]
    pinky_mcp = hand_landmarks.landmark[17]

    palm_size = ((wrist.x - index_finger_mcp.x) ** 2 + (wrist.y - index_finger_mcp.y) ** 2 + (wrist.z - index_finger_mcp.z) ** 2) ** 0.5

    pinch_distance = ((thumb_tip.x - index_finger_tip.x) ** 2 +
                      (thumb_tip.y - index_finger_tip.y) ** 2 +
                      (thumb_tip.z - index_finger_tip.z) ** 2) ** 0.5
    if palm_size > 0:
        pinch_ratio = pinch_distance / palm_size
    else:
        pinch_ratio = pinch_open_ratio
    servo_angle[4] = amplify_motion(
        map_range(clamp(pinch_ratio, pinch_close_ratio, pinch_open_ratio),
                  pinch_close_ratio, pinch_open_ratio,
                  claw_close_angle, claw_open_angle),
        claw_open_angle,
        claw_close_angle,
        claw_open_angle,
    )

    palm_center_x = sum(hand_landmarks.landmark[i].x for i in [0, 5, 9, 13, 17]) / 5.0
    wrist_y = hand_landmarks.landmark[0].y
    servo_angle[0] = amplify_motion(
        map_range(palm_center_x, 0.0, 1.0, x_min, x_max),
        x_mid,
        x_min,
        x_max,
    )
    servo_angle[1] = amplify_motion(
        map_range(clamp(wrist_y, wrist_y_min, wrist_y_max),
                  wrist_y_min, wrist_y_max,
                  y_max, y_min),
        y_mid,
        y_min,
        y_max,
    )

    knuckle_roll = math.degrees(math.atan2(index_finger_mcp.y - pinky_mcp.y,
                                           index_finger_mcp.x - pinky_mcp.x))
    servo_angle[3] = amplify_motion(
        map_range(knuckle_roll, -90.0, 90.0, wrist_min, wrist_max),
        wrist_mid,
        wrist_min,
        wrist_max,
    )

    return [int(value) for value in servo_angle]


def elbow_from_pose_depth(pose_world_landmarks):
    if not pose_world_landmarks:
        return z_mid

    pose = pose_world_landmarks[0]
    right_shoulder = pose[12]
    right_elbow = pose[14]
    right_wrist = pose[16]
    arm_depth = ((right_elbow.z + right_wrist.z) / 2.0) - right_shoulder.z
    return amplify_motion(base_from_arm_depth(arm_depth), z_mid, z_min, z_max)


state_lock = threading.Lock()
shared_state = {
    "angles": neutral_servo_angle(),
    "message": build_channel_message(neutral_servo_angle()).strip(),
    "status": "Starting camera...",
    "fps": 0.0,
    "frame": None,
    "timestamp": time.time(),
    "camera_ready": False,
}
capture_thread_started = False


def update_state(**kwargs):
    with state_lock:
        shared_state.update(kwargs)


def get_state_snapshot():
    with state_lock:
        snapshot = dict(shared_state)
        if snapshot["frame"] is not None:
            snapshot["frame"] = snapshot["frame"].copy()
        return snapshot


def annotate_frame(frame, current_angles, fps, status_text):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90), (12, 18, 33), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.putText(frame, status_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Angles: {current_angles}", (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 255, 190), 1, cv2.LINE_AA)
    return frame


def capture_loop():
    global video_writer, is_recording_active
    cap = cv2.VideoCapture(CAM_SOURCE)
    if not cap.isOpened():
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        annotate = annotate_frame(blank, neutral_servo_angle(), 0.0, "Camera unavailable")
        update_state(status="Camera unavailable", camera_ready=False, frame=annotate)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
    prev_time = time.time()
    frame_count = 0

    try:
        with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while True:
                success, image = cap.read()
                if not success:
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    annotate = annotate_frame(blank, neutral_servo_angle(), 0.0, "Ignoring empty camera frame")
                    update_state(status="Ignoring empty camera frame", camera_ready=True, frame=annotate)
                    time.sleep(0.05)
                    continue

                image.flags.writeable = False
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                hand_results = hands.process(rgb_image)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                frame_timestamp_ms = int(time.time() * 1000)
                pose_results = pose_det.detect_for_video(mp_image, frame_timestamp_ms)

                image.flags.writeable = True
                image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                current_servo_angle = neutral_servo_angle()
                if pose_results.pose_world_landmarks and len(pose_results.pose_world_landmarks) > 0:
                    current_servo_angle[2] = elbow_from_pose_depth(pose_results.pose_world_landmarks)

                status_text = "Tracking hand and pose"
                if hand_results.multi_hand_landmarks:
                    if len(hand_results.multi_hand_landmarks) == 1:
                        hand_landmarks = hand_results.multi_hand_landmarks[0]
                        if hand_is_valid(hand_landmarks):
                            hand_angles = landmark_to_servo_angle(hand_landmarks)
                            current_servo_angle[0] = hand_angles[0]
                            current_servo_angle[1] = hand_angles[1]
                            current_servo_angle[3] = hand_angles[3]
                            current_servo_angle[4] = hand_angles[4]
                        else:
                            status_text = "Hand tracking unstable"
                    else:
                        status_text = "More than one hand detected"

                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                else:
                    status_text = "Waiting for hand"

                image = cv2.flip(image, 1)

                frame_count += 1
                now = time.time()
                elapsed = now - prev_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                if elapsed >= 1.0:
                    prev_time = now
                    frame_count = 0

                message = build_channel_message(current_servo_angle).strip()
                annotated = annotate_frame(image, current_servo_angle, fps, status_text)

                with writer_lock:
                    if is_recording_active and video_writer is not None:
                        video_writer.write(annotated)

                update_state(
                    angles=[int(value) for value in current_servo_angle],
                    message=message,
                    status=status_text,
                    fps=fps,
                    frame=annotated,
                    timestamp=now,
                    camera_ready=True,
                )

    except Exception as exc:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        annotate = annotate_frame(blank, neutral_servo_angle(), 0.0, f"Capture error: {exc}")
        update_state(status=f"Capture error: {exc}", camera_ready=False, frame=annotate)
    finally:
        cap.release()
        with writer_lock:
            if video_writer is not None:
                video_writer.release()
                video_writer = None
        pose_det.close()


def ensure_capture_thread():
    global capture_thread_started
    if capture_thread_started:
        return
    capture_thread_started = True
    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/app.js")
def react_app_js():
    return send_file(APP_JS_PATH, mimetype="application/javascript")


@app.route("/state")
def state():
    snapshot = get_state_snapshot()
    frame = snapshot.pop("frame", None)
    snapshot["frame_available"] = frame is not None
    snapshot["is_recording"] = is_recording_active
    return jsonify(snapshot)


@app.route("/recording/start", methods=["POST"])
def start_recording():
    global is_recording_active, video_writer
    with writer_lock:
        if not is_recording_active:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, 30.0, (640, 480))
            is_recording_active = True
    return jsonify({"status": "started", "is_recording": True})


@app.route("/recording/stop", methods=["POST"])
def stop_recording():
    global is_recording_active, video_writer
    with writer_lock:
        if is_recording_active:
            is_recording_active = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
    return jsonify({"status": "stopped", "is_recording": False, "file": VIDEO_OUTPUT_PATH})


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            snapshot = get_state_snapshot()
            frame = snapshot.get("frame")
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame = annotate_frame(frame, neutral_servo_angle(), 0.0, "Waiting for camera")

            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                time.sleep(0.05)
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n")
            time.sleep(0.016) # ~60 FPS limit

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    ensure_capture_thread()
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)