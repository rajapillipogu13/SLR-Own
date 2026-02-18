import cv2
import mediapipe as mp
import numpy as np
import os

# ==============================
# SETTINGS
# ==============================
SAVE_PATH = "dataset"
CURRENT_LABEL = "M"        # 🔥 Change this when collecting
IMG_SIZE = 128              # 🔥 Must match training size
MAX_IMAGES = 500           # Number per class

# ==============================
# CREATE FOLDER
# ==============================
os.makedirs(os.path.join(SAVE_PATH, CURRENT_LABEL), exist_ok=True)

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

window_name = "Dataset Collection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

connections = mp_hands.HAND_CONNECTIONS

count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            points = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))

            # Draw skeleton
            for connection in connections:
                cv2.line(frame,
                         points[connection[0]],
                         points[connection[1]],
                         (0, 255, 0),
                         2)

            # Draw landmarks
            for point in points:
                cv2.circle(frame,
                           point,
                           3,
                           (255, 0, 0),
                           -1)

            # ==============================
            # BOUNDING BOX (SAME AS APP)
            # ==============================
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

            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size != 0:

                roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

                # 🔥 If training in grayscale use this:
                # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # ==============================
                # SAVE IMAGE WHEN 'S' PRESSED
                # ==============================
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s') and count < MAX_IMAGES:
                    filename = os.path.join(
                        SAVE_PATH,
                        CURRENT_LABEL,
                        f"{count}.jpg"
                    )
                    cv2.imwrite(filename, roi)
                    count += 1
                    print(f"Saved {count}")

    # Info Text
    cv2.putText(frame,
                f"Label: {CURRENT_LABEL} | Count: {count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow(window_name, frame)

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
