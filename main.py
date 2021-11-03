import cv2 as cv
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
hands_model = mp.solutions.hands

src = cv.VideoCapture(1)

with hands_model.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while src.isOpened():
        _, frame = src.read()
        frame = cv.flip(frame, 1)
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand, hands_model.HAND_CONNECTIONS)
                x = int(hand.landmark[hands_model.HandLandmark.WRIST].x * frame.shape[1])
                y = int(hand.landmark[hands_model.HandLandmark.WRIST].y * frame.shape[0])
                cv.putText(frame, results.multi_handedness[num].classification[0].label, (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv.LINE_AA)
                cv.imshow("Hand Pose", frame)
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

src.release()
cv.destroyAllWindows()