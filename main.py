
import cv2
import time
import hand_tracking_module as htm

pTime = 0
cap = cv2.VideoCapture(0)

# Enable elbow detection by setting detectElbows=True
detector = htm.HandDetector(detectionCon=0.7, detectElbows=True)

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    # Get elbows when enabled
    elbows = detector.findElbows(img)

    if len(lmList) != 0:
        print("Index Finger Tip:", lmList[8])

    if elbows:
        for side, x, y in elbows:
            print(f"{side.capitalize()} Elbow: ({x}, {y})")

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}',
                (10, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Hand Tracking Test", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
