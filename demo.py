from tensorflow.keras.models import model_from_json # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import mediapipe as mp # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from collections import deque
import json # type: ignore
import cv2 # type: ignore


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Input(shape=(128,128,3)),

#     tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
#     tf.keras.layers.MaxPool2D(2,2),

#     tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
#     tf.keras.layers.MaxPool2D(2,2),

#     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
#     tf.keras.layers.MaxPool2D(2,2),

#     tf.keras.layers.GlobalAveragePooling2D(),

#     tf.keras.layers.Dense(256,activation='relu'),
#     tf.keras.layers.Dropout(0.5),

#     tf.keras.layers.Dense(32,activation='softmax')
# ])


model = tf.keras.models.Sequential([
    #
    tf.keras.layers.Input(shape=(128,128,3)),
    tf.keras.layers.Conv2D(16,(3,3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(16,(3,3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(32,(3,3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(32,(3,3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(64,(3,3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(64,(3,3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(128,(3,3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(128,(3,3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.GlobalAveragePooling2D(),
    #
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    #
    tf.keras.layers.Dense(32,activation='softmax')
])


pred_queue = deque(maxlen=10)
stable_prediction = ""


# model = load_model("fixed_model.h5", compile=False)

# Load trained model
#model = load_model("pro_cnn_model.keras")
model.load_weights("cnn_upgrade_model.h5")

# Load class labels
with open("models/class_indices.json") as f:
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

    # ==============================
    # TOP CENTER WHITE SENTENCE BOX
    # ==============================
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

    # ==============================
    # HAND DETECTION
    # ==============================
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

            # Small green prediction box
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
                predicted_text = labels[predicted_class_index]
                confidence = np.max(prediction)

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