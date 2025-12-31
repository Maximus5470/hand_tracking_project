import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(
        self,
        mode=False,
        maxHands=2,
        detectionCon=0.5,
        trackCon=0.5,
        detectElbows=False,
        connectWrist=True
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.detectElbows = detectElbows
        self.connectWrist = connectWrist

        # Use the available MediaPipe API surface. Some installs expose
        # APIs under `mp.solutions`, others expose them directly on `mp`.
        if hasattr(mp, "solutions"):
            self.mpHands = mp.solutions.hands
            self.mpDraw = mp.solutions.drawing_utils
            # Holistic/pose (used when detectElbows=True)
            self.mpHolistic = mp.solutions.holistic if hasattr(mp.solutions, 'holistic') else None
        else:
            # Try top-level attributes (some installs expose these directly)
            if hasattr(mp, 'hands') and hasattr(mp, 'drawing_utils'):
                self.mpHands = mp.hands
                self.mpDraw = mp.drawing_utils
            else:
                # Clear, actionable error if nothing matches
                module_location = getattr(mp, '__file__', 'unknown')
                raise ImportError(
                    "MediaPipe API not found. Neither 'mp.solutions' nor top-level "
                    "'mp.hands'/'mp.drawing_utils' are available. Please install the "
                    "official 'mediapipe' package (e.g., `pip install mediapipe`) or fix your PYTHONPATH.\n"
                    f"Detected mediapipe module at: {module_location}"
                )

        # Initialize processors depending on whether elbow/pose detection is requested
        self.hands = None
        self.holistic = None

        if self.detectElbows:
            if not self.mpHolistic:
                raise ImportError(
                    "MediaPipe Holistic/Pose API not available. Please install the official 'mediapipe' package with Holistic support."
                )
            self.holistic = self.mpHolistic.Holistic(
                static_image_mode=self.mode,
                min_detection_confidence=self.detectionCon,
                min_tracking_confidence=self.trackCon
            )
        else:
            self.hands = self.mpHands.Hands(
                static_image_mode=self.mode,
                max_num_hands=self.maxHands,
                min_detection_confidence=self.detectionCon,
                min_tracking_confidence=self.trackCon
            )

        self.tipIds = [4, 8, 12, 16, 20]

        # ✅ Prevent AttributeError
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use Holistic when elbow/pose detection is enabled; otherwise use Hands API
        if self.detectElbows and self.holistic is not None:
            self.results = self.holistic.process(imgRGB)

            # Holistic exposes left/right hand landmarks separately; collect them into a list
            hands = []
            if getattr(self.results, 'multi_hand_landmarks', None):
                hands = self.results.multi_hand_landmarks
            else:
                if getattr(self.results, 'left_hand_landmarks', None):
                    hands.append(self.results.left_hand_landmarks)
                if getattr(self.results, 'right_hand_landmarks', None):
                    hands.append(self.results.right_hand_landmarks)

            if hands and draw:
                for handLms in hands:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS
                    )

        else:
            self.results = self.hands.process(imgRGB)

            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(
                            img,
                            handLms,
                            self.mpHands.HAND_CONNECTIONS
                        )

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if not self.results:
            return lmList

        # Collect hand landmark lists from either Hands or Holistic outputs
        hands = []
        if getattr(self.results, 'multi_hand_landmarks', None):
            hands = self.results.multi_hand_landmarks
        else:
            if getattr(self.results, 'left_hand_landmarks', None):
                hands.append(self.results.left_hand_landmarks)
            if getattr(self.results, 'right_hand_landmarks', None):
                hands.append(self.results.right_hand_landmarks)

        if not hands or handNo >= len(hands):
            return lmList

        myHand = hands[handNo]
        h, w, _ = img.shape

        for id, lm in enumerate(myHand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])

            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmList

    def fingersUp(self, lmList):
        # ✅ Prevent IndexError
        if not lmList or len(lmList) < 21:
            return [0, 0, 0, 0, 0]

        fingers = []

        # Thumb (works for right hand)
        if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other four fingers
        for id in range(1, 5):
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findElbows(self, img, draw=True):
        """Return list of elbow points in image coordinates.

        Uses MediaPipe Holistic/Pose landmarks when `detectElbows=True`.
        Returns list of tuples: (side, x, y) where side is 'left' or 'right'.
        """
        elbows = []

        if not self.detectElbows or not self.holistic:
            return elbows

        # If results are from a prior findHands call they should exist; otherwise process now
        if not self.results:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.holistic.process(imgRGB)

        pose = getattr(self.results, 'pose_landmarks', None)
        if not pose:
            return elbows

        h, w, _ = img.shape

        # MediaPipe Pose landmark indices: LEFT_ELBOW=13, RIGHT_ELBOW=14
        left_elbow = pose.landmark[13]
        right_elbow = pose.landmark[14]

        lx, ly = int(left_elbow.x * w), int(left_elbow.y * h)
        rx, ry = int(right_elbow.x * w), int(right_elbow.y * h)

        elbows.append(('left', lx, ly))
        elbows.append(('right', rx, ry))

        if draw:
            cv2.circle(img, (lx, ly), 7, (0, 255, 0), cv2.FILLED)
            cv2.putText(img, 'L-Elbow', (lx + 5, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.circle(img, (rx, ry), 7, (0, 0, 255), cv2.FILLED)
            cv2.putText(img, 'R-Elbow', (rx + 5, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # Draw connecting lines to pose wrists (indices 15,16) if enabled
            if self.connectWrist:
                left_wrist = pose.landmark[15]
                right_wrist = pose.landmark[16]
                lwx, lwy = int(left_wrist.x * w), int(left_wrist.y * h)
                rwx, rwy = int(right_wrist.x * w), int(right_wrist.y * h)
                cv2.line(img, (lx, ly), (lwx, lwy), (0,200,0), 3)
                cv2.line(img, (rx, ry), (rwx, rwy), (0,0,200), 3)

        return elbows
