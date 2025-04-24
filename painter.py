import cv2
import numpy as np
import mediapipe as mp
from utils import draw_color_buttons, distance

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
colors = [(255, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
current_color = colors[0]
color_index = 0
brush_thickness = 5
xp, yp = 0, 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    draw_color_buttons(frame, colors, color_index)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm = handLms.landmark
            fingers = [(int(lm[8].x * w), int(lm[8].y * h)),
                       (int(lm[12].x * w), int(lm[12].y * h)),
                       (int(lm[4].x * w), int(lm[4].y * h))]
            index_finger = fingers[0]
            middle_finger = fingers[1]
            thumb = fingers[2]

            # Selection mode
            if distance(index_finger, middle_finger) < 40:
                xp, yp = 0, 0
                for i, color in enumerate(colors):
                    if 20 + i * 60 < index_finger[0] < 60 + i * 60 and 10 < index_finger[1] < 50:
                        color_index = i
                        current_color = colors[i]

            # Drawing mode
            elif distance(index_finger, thumb) > 50:
                if xp == 0 and yp == 0:
                    xp, yp = index_finger
                cv2.line(canvas, (xp, yp), index_finger, current_color, brush_thickness)
                xp, yp = index_finger

            # Clear gesture (fist)
            if distance(index_finger, middle_finger) < 20 and distance(index_finger, thumb) < 20:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    frame = cv2.add(frame, canvas)
    cv2.imshow("Virtual Painter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
