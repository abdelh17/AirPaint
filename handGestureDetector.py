import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

canvas = None
# Colors for each finger
colors = {
    "thumb": (0, 255, 0),     # Green
    "index": (255, 0, 0),     # Blue
    "middle": (0, 0, 255),    # Red
    "ring": (255, 255, 0),    # Cyan
    "pinky": (255, 0, 255)    # Magenta
}

# Define finger landmarks
finger_connections = {
    "wrist_to_index": [(0, 5)],       # Connection from wrist to thumb base
    "wrist_to_pinky": [(0, 17)],
    "wrist_to_thumb": [(0, 1)],
    "thumb": [(1, 2), (2, 3), (3, 4)],
    "index": [(5, 6), (6, 7), (7, 8)],
    "middle": [(9, 10), (10, 11), (11, 12)],
    "ring": [(13, 14), (14, 15), (15, 16)],
    "pinky": [(17, 18), (18, 19), (19, 20)]
}

index_finger_tip_positions = []

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

    neutral_color = (255, 255, 255)  # White color for wrist connections

    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         # Drawing wrist connections
    #         for connection in finger_connections["wrist_to_index"]:
    #             mp_draw.draw_landmarks(
    #                 frame,
    #                 hand_landmarks,
    #                 [connection],
    #                 mp_draw.DrawingSpec(color=neutral_color, thickness=2, circle_radius=5),
    #                 mp_draw.DrawingSpec(color=neutral_color, thickness=2, circle_radius=2)
    #             )
    #         for connection in finger_connections["wrist_to_pinky"]:
    #             mp_draw.draw_landmarks(
    #                 frame,
    #                 hand_landmarks,
    #                 [connection],
    #                 mp_draw.DrawingSpec(color=neutral_color, thickness=2, circle_radius=5),
    #                 mp_draw.DrawingSpec(color=neutral_color, thickness=2, circle_radius=2)
    #             )
    #         for connection in finger_connections["wrist_to_thumb"]:
    #             mp_draw.draw_landmarks(
    #                 frame,
    #                 hand_landmarks,
    #                 [connection],
    #                 mp_draw.DrawingSpec(color=neutral_color, thickness=2, circle_radius=5),
    #                 mp_draw.DrawingSpec(color=neutral_color, thickness=2, circle_radius=2)
    #             )
    #
    #         # Drawing finger connections
    #         for finger, connections in finger_connections.items():
    #             if finger not in ["wrist_to_index", "wrist_to_pinky", "wrist_to_thumb"]:  # Skip wrist connections here
    #                 for connection in connections:
    #                     mp_draw.draw_landmarks(
    #                         frame,
    #                         hand_landmarks,
    #                         [connection],
    #                         mp_draw.DrawingSpec(color=colors[finger], thickness=2, circle_radius=5),
    #                         mp_draw.DrawingSpec(color=colors[finger], thickness=2, circle_radius=2)
    #                     )

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]  # Index finger tip landmark
            # Convert normalized coordinates to frame coordinates
            h, w, c = frame.shape
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            # Draw on the canvas
            cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)  # Red color, filled circle

    # Overlay the canvas on the frame
    frame = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

    # Display the resulting frame
    cv2.imshow('Hand Landmarks with Drawing', frame)

    # Break the loop with the 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    # # File handling
    # with open("hand_landmarks.txt", "a") as file:
    #     if results.multi_hand_landmarks:
    #         index_tip = results.multi_hand_landmarks[0].landmark[8]
    #         file.write(f"Landmark 8: x={index_tip.x}, y={index_tip.y}, z={index_tip.z}\n")
    #         file.write("\n")
    #
    #         # for hand_landmarks in results.multi_hand_landmarks:
    #         #     for i, lm in enumerate(hand_landmarks.landmark):
    #         #         file.write(f"Landmark {i}: x={lm.x}, y={lm.y}, z={lm.z}\n")
    #         #     file.write("\n")
    #
    # # Display the resulting frame
    # cv2.imshow('Hand Landmarks', frame)
    #
    # # Break the loop with the 'q' key
    # if cv2.waitKey(5) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()


