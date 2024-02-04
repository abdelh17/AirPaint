import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox

def ask_hand_preference():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    hand_preference = messagebox.askquestion('Hand Preference', 'Are you right-handed?')
    root.destroy()
    return 'right' if hand_preference == 'yes' else 'left'

def calculate_distance(point1, point2):
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

hand_preference = ask_hand_preference()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1)

canvas = None
prev_tip_position = None
is_drawing_mode = False
is_erase_mode = False
current_color = (10, 10, 247)  # Default color: Blue

toggle_gesture_threshold = 0.05
gesture_toggle_state = False
gesture_toggle_counter = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    if canvas is None:
        canvas = np.zeros_like(frame)

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = handedness.classification[0].label.lower()
            if hand_type != hand_preference:
                continue

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            tip_position = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
            distance = calculate_distance(thumb_tip, index_tip)

            if distance < toggle_gesture_threshold:
                if not gesture_toggle_state:
                    is_drawing_mode = not is_drawing_mode
                    is_erase_mode = False
                    gesture_toggle_state = True
                    gesture_toggle_counter = 10
            elif gesture_toggle_state and gesture_toggle_counter > 0:
                gesture_toggle_counter -= 1
            else:
                gesture_toggle_state = False

            if prev_tip_position is not None:
                if is_drawing_mode:
                    cv2.line(canvas, prev_tip_position, tip_position, current_color, thickness=5)
                elif is_erase_mode:
                    cv2.line(canvas, prev_tip_position, tip_position, (0, 0, 0), thickness=20)

            prev_tip_position = tip_position

    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    modes_text = f'Drawing mode: {"ON" if is_drawing_mode else "OFF"}  Erase mode: {"ON" if is_erase_mode else "OFF"}'
    cv2.putText(frame, modes_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Hand Landmarks with Drawing', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        canvas = np.zeros_like(frame)
        is_drawing_mode = False
        is_erase_mode = False
    elif key == ord('e'):
        is_erase_mode = not is_erase_mode
        is_drawing_mode = False if is_erase_mode else is_drawing_mode
    elif key == ord('r'):
        current_color = (0, 0, 255)
    elif key == ord('g'):
        current_color = (0, 255, 0)
    elif key == ord('b'):
        current_color = (255, 0, 0)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
