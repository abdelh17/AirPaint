# import cv2
# import mediapipe as mp
# import numpy as np
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult
#
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     model_complexity=0,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)
# mp_draw = mp.solutions.drawing_utils
#
# canvas = None
# # Colors for each finger
# colors = {
#     "thumb": (0, 255, 0),  # Green
#     "index": (255, 0, 0),  # Blue
#     "middle": (0, 0, 255),  # Red
#     "ring": (255, 255, 0),  # Cyan
#     "pinky": (255, 0, 255)  # Magenta
# }
#
# # Define finger landmarks
# finger_connections = {
#     "wrist_to_index": [(0, 5)],  # Connection from wrist to thumb base
#     "wrist_to_pinky": [(0, 17)],
#     "wrist_to_thumb": [(0, 1)],
#     "thumb": [(1, 2), (2, 3), (3, 4)],
#     "index": [(5, 6), (6, 7), (7, 8)],
#     "middle": [(9, 10), (10, 11), (11, 12)],
#     "ring": [(13, 14), (14, 15), (15, 16)],
#     "pinky": [(17, 18), (18, 19), (19, 20)]
# }
#
# index_finger_tip_positions = []
#
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         continue
#
#     if canvas is None:
#         canvas = np.zeros_like(frame)
#
#     # Flip the image horizontally for a later selfie-view display
#     frame = cv2.flip(frame, 1)
#
#     # Convert the BGR image to RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process the image and detect hands
#     results = hands.process(frame_rgb)
#
#     neutral_color = (255, 255, 255)  # White color for wrist connections
#
#
#     # if results.multi_hand_landmarks:
#     #     for hand_landmarks in results.multi_hand_landmarks:
#     #         index_tip = hand_landmarks.landmark[8]  # Index finger tip landmark
#     #         # Convert normalized coordinates to frame coordinates
#     #         h, w, c = frame.shape
#     #         x, y = int(index_tip.x * w), int(index_tip.y * h)
#     #
#     #         # Draw on the canvas
#     #         cv2.circle(canvas, center=(x, y), radius=5, color=(0, 0, 255), thickness=3)  # Red color, filled circle
#
#     # if results.multi_hand_landmarks:
#     #     for hand_landmarks in results.multi_hand_landmarks:
#     #         # Iterate over finger connections
#     #         for finger_name, connections in finger_connections.items():
#     #             for connection in connections:
#     #                 start_landmark = connection[0]
#     #                 end_landmark = connection[1]
#     #
#     #                 # Convert normalized coordinates to frame coordinates
#     #                 h, w, c = frame.shape
#     #                 start_x, start_y = int(hand_landmarks.landmark[start_landmark].x * w), int(
#     #                     hand_landmarks.landmark[start_landmark].y * h)
#     #                 end_x, end_y = int(hand_landmarks.landmark[end_landmark].x * w), int(
#     #                     hand_landmarks.landmark[end_landmark].y * h)
#     #
#     #                 # Draw lines on the canvas
#     #                 cv2.line(canvas, (start_x, start_y), (end_x, end_y), colors.get(finger_name, neutral_color),
#     #                          thickness=3)
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Get the index finger tip position
#             index_tip = hand_landmarks.landmark[8]  # Index finger tip landmark
#             tip_position = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
#
#             # Draw a line connecting the current tip to the previous tip
#             if prev_tip_position is not None:
#                 cv2.line(canvas, prev_tip_position, tip_position, colors.get("index", (255, 255, 255)), thickness=3)
#
#             # Store the current tip position for the next iteration
#             prev_tip_position = tip_position
#
#
#     # Overlay the canvas on the frame
#     frame = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
#
#     # Display the resulting frame
#     cv2.imshow('Hand Landmarks with Drawing', frame)
#
#     # Break the loop with the 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#




# ----------------------------------------
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

canvas = None
prev_tip_position = None
# Colors for each finger
colors = {
    "index": (10, 10, 247),  # Blue
}

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
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the index finger tip position
            index_tip = hand_landmarks.landmark[8]  # Index finger tip landmark
            tip_position = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))

            # Draw a line connecting the current tip to the previous tip
            if prev_tip_position is not None:
                cv2.line(canvas, prev_tip_position, tip_position, colors.get("index"), thickness=5)

            # Store the current tip position for the next iteration
            prev_tip_position = tip_position


    # Overlay the canvas on the frame
    frame = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

    # Display the resulting frame
    cv2.imshow('Hand Landmarks with Drawing', frame)

    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
