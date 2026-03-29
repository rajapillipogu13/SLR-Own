from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import tensorflow as tf
import mediapipe as mp
import numpy as np
from collections import deque
import json
import cv2
from tkinter import *
from PIL import Image, ImageTk
import time

class HandTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Sign Language Recognition")
        self.root.state("zoomed")  

        # Title
        self.title_label = Label(root)

        # Video display
        self.video_label = Label(root, bg="black")
        self.video_label.pack(fill=BOTH, expand=True)

        # Status bar
        self.status_label = Label(
            root,
            text="Status: Initializing...",
            bd=1,
            relief=SUNKEN,
            anchor=W,
            bg="#2e2e2e",
            fg="white"
        )
        self.status_label.pack(side=BOTTOM, fill=X)

        # Model definition
        self.model = tf.keras.models.Sequential([
            # block 1
            tf.keras.layers.Input(shape=(128,128,3)),
            tf.keras.layers.Conv2D(16,(3,3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(16,(3,3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(2,2),
            # block 2
            tf.keras.layers.Conv2D(32,(3,3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(32,(3,3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(2,2),
            # block 3
            tf.keras.layers.Conv2D(64,(3,3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(64,(3,3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(2,2),
            # block 4
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
        self.model.load_weights("models/model_upgrade/cnn_upgrade_model.h5")
        

        # Labels
        with open("models/class_indices.json") as f:
            class_indices = json.load(f)
        self.labels = {v: k for k, v in class_indices.items()}

        # State
        self.pred_queue = deque(maxlen=10)
        self.stable_prediction = ""
        self.sentence = "HELLO I AM RAJA"

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.connections = self.mp_hands.HAND_CONNECTIONS

        # Camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.pTime = 0
        self.prev_box = None
        self.smooth_factor = 0.7

        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def smooth_box(self, new_box):
        if self.prev_box is None:
            self.prev_box = new_box
            return new_box
        smoothed = []
        for prev, new in zip(self.prev_box, new_box):
            smoothed.append(int(prev * self.smooth_factor + new * (1 - self.smooth_factor)))
        self.prev_box = smoothed
        return smoothed

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Status: Camera Error")
            self.video_label.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)

        h, w, c = frame.shape

        # Top center white sentence box (from demo4)
        box_width = min(1920, w)
        box_height = 40
        x1 = (w // 2) - (box_width // 2)
        y1 = 20
        x2 = x1 + box_width
        y2 = y1 + box_height
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        text_size = cv2.getTextSize(self.sentence, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = x1 + (box_width - text_size[0]) // 2
        text_y = y1 + (box_height + text_size[1]) // 2
        cv2.putText(frame, self.sentence, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )
                # Points for bbox and circles
                x_list = []
                y_list = []
                for lm in hand_landmarks.landmark:
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))

                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)

                # Smooth bbox
                xmin_s, ymin_s, xmax_s, ymax_s = self.smooth_box([xmin, ymin, xmax, ymax])

                # Green hand box
                padding = 20
                cv2.rectangle(frame, (xmin_s - padding, ymin_s - padding),
                              (xmax_s + padding, ymax_s + padding), (0, 255, 0), 2)

                # Blue landmark circles
                points = list(zip(x_list, y_list))
                for point in points:
                    cv2.circle(frame, point, 3, (255, 0, 0), -1)

                # Prediction crop (from bbox smoothed + padding)
                crop_ymin = max(0, ymin_s - padding)
                crop_xmin = max(0, xmin_s - padding)
                crop_ymax = min(h, ymax_s + padding)
                crop_xmax = min(w, xmax_s + padding)
                hand_img = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

                if hand_img.size > 0:
                    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                    hand_img = cv2.resize(hand_img, (128, 128))
                    hand_img = hand_img.astype("float32") / 255.0
                    hand_img = np.expand_dims(hand_img, axis=0)
                    prediction = self.model.predict(hand_img, verbose=0)
                    predicted_class_index = np.argmax(prediction)
                    confidence = np.max(prediction)

                    if confidence > 0.7:
                        self.pred_queue.append(predicted_class_index)

                    if len(self.pred_queue) == 10:
                        most_common = max(set(self.pred_queue), key = self.pred_queue.count)
                        self.stable_prediction = self.labels[most_common]

                # Green prediction box above hand
                pred_height = 60
                pred_ymin = max(0, ymin_s - pred_height - 10)
                pred_ymax = ymin_s - 10
                pred_width = xmax_s - xmin_s + 2 * padding
                cv2.rectangle(frame, (xmin_s - padding, pred_ymin),
                              (xmin_s - padding + pred_width, pred_ymax), (0, 255, 0), -1)
                text_size_pred = cv2.getTextSize(self.stable_prediction, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                scale = min(1.5, pred_width / text_size_pred[0] * 1.5)
                cv2.putText(frame, self.stable_prediction, (int(xmin_s - padding + 20), int(pred_ymax - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)

            status_text = f"Status: Predicting {self.stable_prediction}" if self.stable_prediction else "Status: Detecting Hands..."
        else:
            status_text = "Status: Detecting Hands..."

        self.status_label.config(text=status_text)

        
        # Resize to fit label
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        if label_width > 1 and label_height > 1:
            frame = cv2.resize(frame, (label_width, label_height), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.video_label.after(10, self.update_frame)

    def on_close(self):
        self.cap.release()
        self.hands.close()
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = HandTrackingApp(root)
    root.mainloop()
