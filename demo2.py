from keras.models import model_from_json # type: ignore
import cv2 # type: ignore
import mediapipe as mp # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import json
from collections import deque

pred_queue = deque(maxlen=10)
stable_prediction = ""

# Load trained model
model = load_model("pro_cnn_model.keras")

# Load class labels
with open("class_indices.json") as f:
    class_indices = json.load(f)

# Reverse mapping (index → label)
labels = {v: k for k, v in class_indices.items()}


mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0)

cap.set(3, 1920)
cap.set(4, 1080)

window_name = "Hand Tracking"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1920, 1080)

connections = mp_hands.HAND_CONNECTIONS

# Example sentence text (replace later with real word formation)
sentence = "HELLO I AM RAJA"

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # TOP CENTER WHITE SENTENCE BOX
    box_width = 1920
    box_height = 40

    x1 = (w // 2) - (box_width // 2)
    y1 = 20
    x2 = x1 + box_width
    y2 = y1 + box_height

    # Draw white filled rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)


    # Center sentence text
    text_size = cv2.getTextSize(sentence,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, 3)[0]

    text_x = x1 + (box_width - text_size[0]) // 2
    text_y = y1 + (box_height + text_size[1]) // 2

    cv2.putText(frame,
                sentence,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                2)

    # HAND DETECTION
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            points = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))

            # Skeleton
            for connection in connections:
                cv2.line(frame,
                         points[connection[0]],
                         points[connection[1]],
                         (0, 255, 0),
                         2)

            # Landmarks
            for point in points:
                cv2.circle(frame,
                           point,
                           3,
                           (255, 0, 0),
                           -1)

            # Bounding box
            x_min = min([p[0] for p in points])
            y_min = min([p[1] for p in points])
            x_max = max([p[0] for p in points])
            y_max = max([p[1] for p in points])

            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            cv2.rectangle(frame,
                          (x_min, y_min),
                          (x_max, y_max),
                          (0, 255, 0),
                          2)

            # Prediction box
            box_height = 60
            pred_y_min = max(0, y_min - box_height - 10)
            pred_y_max = y_min - 10

            cv2.rectangle(frame,
                          (x_min, pred_y_min),
                          (x_max, pred_y_max),
                          (0, 255, 0),
                          -1)

            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:

                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                
                # Resize to 128x128 (same as training)
                hand_img = cv2.resize(hand_img, (128, 128))

                # Convert to float32
                hand_img = hand_img.astype("float32")

                # Rescale (same as ImageDataGenerator)
                hand_img = hand_img / 255.0

                # Add batch dimension
                hand_img = np.expand_dims(hand_img, axis=0)

                # Predict
                prediction = model.predict(hand_img, verbose=0)
                predicted_class_index = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # GEOMETRY CORRECTION BLOCK
                # if confidence < 0.93:  # Only override if not extremely confident

                #     wrist = points[0]
                #     thumb_tip = points[4]
                #     index_tip = points[8]
                #     middle_tip = points[12]
                #     ring_tip = points[16]
                #     pinky_tip = points[20]

                #     # Condition A vs S 
                #     if predicted_text in ["a", "s"]:
                #         index_mcp = points[5]
                #         if thumb_tip[0] < index_mcp[0]:
                #             predicted_text = "s"
                #         else:
                #             predicted_text = "a"

                #     # Condition D vs R/U/V
                #     if predicted_text in ["d", "r", "u", "v"]:
                #         extended = 0
                #         if index_tip[1] < wrist[1]: extended += 1
                #         if middle_tip[1] < wrist[1]: extended += 1
                #         if ring_tip[1] < wrist[1]: extended += 1
                #         if pinky_tip[1] < wrist[1]: extended += 1

                #         if extended == 1:
                #             predicted_text = "d"

                #     # ---------------- F vs B ----------------
                #     if predicted_text in ["f", "b"]:

                #         thumb_tip = points[4]
                #         index_tip = points[8]
                #         middle_tip = points[12]
                #         ring_tip = points[16]
                #         pinky_tip = points[20]
                #         wrist = points[0]

                #         # Distance between thumb & index
                #         thumb_index_dist = np.linalg.norm(
                #             np.array(thumb_tip) - np.array(index_tip)
                #         )

                #         # Count extended fingers
                #         extended = 0
                #         if middle_tip[1] < wrist[1]: extended += 1
                #         if ring_tip[1] < wrist[1]: extended += 1
                #         if pinky_tip[1] < wrist[1]: extended += 1

                #         # F → circle + fingers extended
                #         if thumb_index_dist < 40 and extended >= 2:
                #             predicted_text = "f"

                #         # B → no circle + fingers extended
                #         elif thumb_index_dist >= 40 and extended >= 3:
                #             predicted_text = "b"

                #     # ---------------- G vs H ----------------
                #     if predicted_text in ["g", "h"]:

                #         index_tip = points[8]
                #         middle_tip = points[12]
                #         wrist = points[0]

                #         # Check if fingers are extended (above wrist)
                #         index_extended = index_tip[1] < wrist[1]
                #         middle_extended = middle_tip[1] < wrist[1]

                #         # Check vertical alignment
                #         vertical_diff = abs(index_tip[1] - middle_tip[1])

                #         # G: only index extended
                #         if index_extended and not middle_extended:
                #             predicted_text = "g"

                #         # H: both extended & roughly aligned
                #         elif index_extended and middle_extended and vertical_diff < 25:
                #             predicted_text = "h"

                #     # ---------------- T vs N ----------------
                #     if predicted_text in ["t", "n"]:

                #         thumb_tip = np.array(points[4])
                #         index_mcp = np.array(points[5])
                #         middle_mcp = np.array(points[9])
                #         ring_mcp = np.array(points[13])

                #         # Compute horizontal midpoints
                #         midpoint_left = (index_mcp[0] + middle_mcp[0]) / 2
                #         midpoint_right = (middle_mcp[0] + ring_mcp[0]) / 2

                #         # If thumb is closer to index-middle region → T
                #         if thumb_tip[0] < midpoint_left:
                #             predicted_text = "t"

                #         # If thumb closer to middle-ring region → N
                #         elif thumb_tip[0] > midpoint_right:
                #             predicted_text = "n"

                #         # Fallback
                #         else:
                #             predicted_text = "n"

                #     # ---------------- W vs V/K ----------------
                #     if predicted_text in ["w", "v", "k"]:
                #         extended = 0
                #         if index_tip[1] < wrist[1]: extended += 1
                #         if middle_tip[1] < wrist[1]: extended += 1
                #         if ring_tip[1] < wrist[1]: extended += 1

                #         if extended == 3:
                #             predicted_text = "w"

                #     # ---------------- Z / X / P ----------------
                #     if predicted_text in ["z", "x", "p"]:

                #         wrist = np.array(points[0])
                #         index_tip = np.array(points[8])
                #         middle_tip = np.array(points[12])
                #         index_pip = np.array(points[6])
                #         index_mcp = np.array(points[5])

                #         # Check extension direction
                #         index_up = index_tip[1] < wrist[1]
                #         index_down = index_tip[1] > wrist[1]
                #         middle_down = middle_tip[1] > wrist[1]

                #         # Compute index finger bend angle
                #         v1 = index_pip - index_mcp
                #         v2 = index_tip - index_pip
                #         cos_angle = np.dot(v1, v2) / (
                #             np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
                #         )
                #         angle = np.degrees(np.arccos(cos_angle))

                #         # P → index + middle downward
                #         if index_down and middle_down:
                #             predicted_text = "p"

                #         # Z → index straight upward
                #         elif index_up and angle > 150:
                #             predicted_text = "z"

                #         # X → index bent
                #         else:
                #             predicted_text = "x"

                #     # ---------------- Q / P / SPACE ----------------
                #     if predicted_text in [ "p", "space"]:

                #         wrist = np.array(points[0])
                #         index_tip = np.array(points[8])
                #         middle_tip = np.array(points[12])
                #         ring_tip = np.array(points[16])
                #         pinky_tip = np.array(points[20])

                #         # Check downward fingers
                #         index_down = index_tip[1] > wrist[1]
                #         middle_down = middle_tip[1] > wrist[1]

                #         # Count upward extended fingers
                #         extended_up = 0
                #         if index_tip[1] < wrist[1]: extended_up += 1
                #         if middle_tip[1] < wrist[1]: extended_up += 1
                #         if ring_tip[1] < wrist[1]: extended_up += 1
                #         if pinky_tip[1] < wrist[1]: extended_up += 1

                #         # P → two downward fingers
                #         if index_down and middle_down:
                #             predicted_text = "p"

                #         # Q → only index downward
                #         # elif index_down and not middle_down:
                #         #     predicted_text = "q"

                #         # Space → open palm (3+ upward fingers)
                #         elif extended_up >= 3:
                #             predicted_text = "space"
                    
                #     # ---------------- OK / Minimize / G ----------------
                #     if predicted_text in ["ok", "minimize", "g"]:

                #         thumb_tip = np.array(points[4])
                #         index_tip = np.array(points[8])
                #         middle_tip = np.array(points[12])
                #         ring_tip = np.array(points[16])
                #         pinky_tip = np.array(points[20])
                #         wrist = np.array(points[0])

                #         thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)

                #         # Count extended fingers
                #         extended = 0
                #         if index_tip[1] < wrist[1]: extended += 1
                #         if middle_tip[1] < wrist[1]: extended += 1
                #         if ring_tip[1] < wrist[1]: extended += 1
                #         if pinky_tip[1] < wrist[1]: extended += 1

                #         # OK → circle + multiple extended fingers
                #         if thumb_index_dist < 35 and extended >= 2:
                #             predicted_text = "ok"

                #         # G → only index extended
                #         elif extended == 1:
                #             predicted_text = "g"

                #         # Otherwise → Minimize
                #         else:
                #             predicted_text = "minimize"
                

                # -------------------------------
                # STABILIZATION (Queue Smoothing)
                # -------------------------------
                
                if confidence > 0.7:
                    pred_queue.append(predicted_class_index)

                if len(pred_queue) == 10:
                    most_common = max(set(pred_queue), key=pred_queue.count)
                    stable_prediction = labels[most_common]

                predicted_text = stable_prediction

            cv2.putText(frame,
                        predicted_text,
                        (x_min + 20, pred_y_max - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 0),
                        3)

    cv2.imshow(window_name, frame)

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
