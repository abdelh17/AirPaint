import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Function to create pop-up for hand preference
def ask_hand_preference():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    hand_preference = messagebox.askquestion('Hand Preference', 'Are you right-handed?')
    root.destroy()
    return 'right' if hand_preference == 'yes' else 'left'

# Get hand preference from user
hand_preference = ask_hand_preference()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1)  # Only detect one hand
mp_draw = mp.solutions.drawing_utils

#
# # Prompt for hand preference
# hand_preference = input("Are you lefty or righty? Enter 'left' or 'right': ").strip().lower()
# if hand_preference not in ['left', 'right']:
#     print("Invalid input. Defaulting to 'right'.")
#     hand_preference = 'right'
#
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     model_complexity=0,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)
# mp_draw = mp.solutions.drawing_utils

canvas = None
prev_tip_position = None
is_drawing_mode = False
is_erase_mode = False
current_color = (10, 10, 247)  # Default color: Blue

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Check hand type (left or right)
            hand_type = handedness.classification[0].label.lower()
            if hand_type != hand_preference:
                continue

            # Get the index finger tip position
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            tip_position = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))

            if prev_tip_position is not None:
                if is_drawing_mode:
                    cv2.line(canvas, prev_tip_position, tip_position, current_color, thickness=5)
                elif is_erase_mode:
                    cv2.line(canvas, prev_tip_position, tip_position, (0, 0, 0), thickness=20)  # Erase with black color

            prev_tip_position = tip_position

    # Overlay the canvas on the frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display current mode
    modes_text = f'Drawing mode: {"ON" if is_drawing_mode else "OFF"}  Erase mode: {"ON" if is_erase_mode else "OFF"}'
    cv2.putText(frame, modes_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Hand Landmarks with Drawing', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):  # Wipe canvas
        canvas = np.zeros_like(frame)
        is_drawing_mode = False
        is_erase_mode = False
    elif key == ord('d'):  # Toggle drawing mode
        is_drawing_mode = not is_drawing_mode
        is_erase_mode = False if is_drawing_mode else is_erase_mode
    elif key == ord('e'):  # Toggle erase mode
        is_erase_mode = not is_erase_mode
        is_drawing_mode = False if is_erase_mode else is_drawing_mode
    elif key == ord('r'):  # Red color
        current_color = (0, 0, 255)
    elif key == ord('g'):  # Green color
        current_color = (0, 255, 0)
    elif key == ord('b'):  # Blue color
        current_color = (255, 0, 0)
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
