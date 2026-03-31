from tensorflow.keras.models import model_from_json # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from collections import deque
import tensorflow as tf
import mediapipe as mp
from tkinter import *
import numpy as np
import time
import json
import cv2
from PIL import Image, ImageTk
from tkinter import messagebox

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Sign Language Recognition")
        self.root.state("zoomed")  

        # Title
        self.title_label = Label(root)

        # Video display
        self.video_label = Label(root, bg="black")
        self.video_label.pack(fill=BOTH, expand=True)
        
        # Clear button
        self.clear_btn = Button(
            root,
            text="CLEAR",
            font=("Arial", 14, "bold"),
            bg="red",
            fg="white",
            command=self.clear_sentence
        )

        self.clear_btn.place(relx=0.95, rely=0.05, anchor="ne")

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
        
        self.label_to_index = {v: k for k, v in self.labels.items()}


        # State
        self.pred_queue = deque(maxlen=5)
        self.stable_prediction = ""
        
        
        # sentence 
        self.sentence = ""
        self.last_action_time = time.time()
        self.prev_gesture = ""
        self.pending_action = None 
        self.last_char = ""

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
    

    def clear_sentence(self):
        if messagebox.askyesno("Clear", "Clear the sentence?"):
            self.sentence = ""
            self.pending_action = None
            self.last_char = ""
            self.prev_gesture = ""
        
    def handle_gesture(self, char):

        # STORE INPUT
        if char not in ["NEXT"]:

            if char == "BACKSPACE":
                self.pending_action = "BACKSPACE"

            elif char == "OK":
                self.pending_action = "SPACE"

            else:
                self.pending_action = "CHAR"
                self.last_char = char

        # NEXT → EXECUTE
        elif char == "NEXT":

            if self.prev_gesture != "NEXT":

                # ADD LETTER
                if self.pending_action == "CHAR":
                    if self.last_char not in ["", "NEXT", "BACKSPACE"]:
                        self.sentence += self.last_char

                # DELETE
                elif self.pending_action == "BACKSPACE":
                    if len(self.sentence) > 0:
                        self.sentence = self.sentence[:-1]

                # SPACE
                elif self.pending_action == "SPACE":
                    if not self.sentence.endswith(" "):
                        self.sentence += " "

                # reset
                self.pending_action = None

        # UPDATE STATE
        self.prev_gesture = char
    
    
    
    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def apply_rules(self, ch1, ch2, ch3, landmarks):        

        # C & O
        dist = np.linalg.norm(np.array(landmarks[4]) - np.array(landmarks[8]))

        if ch1 == self.label_to_index['C'] or ch1 == self.label_to_index['O']:
            if dist > 40:
                ch1 = self.label_to_index['C']
            else:
                ch1 = self.label_to_index['O']
                                
        # A & S thumb position
        if ch1 in [self.label_to_index['A'], self.label_to_index['S']]:
            if landmarks[4][0] < landmarks[6][0]:
                ch1 = self.label_to_index['A']
            else:
                ch1 = self.label_to_index['S']
                
        # A vs BACKSPACE
        if ch1 in [self.label_to_index['A'], self.label_to_index['BACKSPACE']]:
            if (landmarks[8][1] > landmarks[6][1] and   # index down
                landmarks[12][1] > landmarks[10][1] and # middle down
                landmarks[16][1] > landmarks[14][1] and # ring down
                landmarks[20][1] > landmarks[18][1]):   # pinky down

                dist_thumb_index = np.linalg.norm(
                    np.array(landmarks[4]) - np.array(landmarks[8])
                )

                # BACKSPACE  thumb far
                if dist_thumb_index > 100:
                    return self.label_to_index['BACKSPACE']

                # A thumb close
                else:
                    return self.label_to_index['A']
                
        # I
        if (landmarks[20][1] < landmarks[18][1] and   
                landmarks[8][1] > landmarks[6][1] and   
                landmarks[12][1] > landmarks[10][1] and  
                landmarks[16][1] > landmarks[14][1]):  
                
                ch1 = self.label_to_index['I']
        
        # I / J / Y pinky-up group
        if (landmarks[20][1] < landmarks[18][1] and   
            landmarks[8][1] > landmarks[6][1] and
            landmarks[12][1] > landmarks[10][1] and
            landmarks[16][1] > landmarks[14][1]):    

            # distance between thumb tip and ring tip
            dist_thumb_ring = np.linalg.norm(
                np.array(landmarks[4]) - np.array(landmarks[16])
            )

            # Y 
            if dist_thumb_ring > 70:
                return self.label_to_index['Y']

            # I vs J
            dx = landmarks[20][0] - landmarks[18][0]

            if dx > 15:
                return self.label_to_index['J']
            else:
                return self.label_to_index['I']
            
            
        # D vs Z 
        if ch1 in [self.label_to_index['D'], self.label_to_index['Z']]:

            dx = abs(landmarks[8][0] - landmarks[6][0])  
            dy = abs(landmarks[8][1] - landmarks[6][1]) 

            # Z angled
            if dx > dy * 0.7:
                return self.label_to_index['Z']

            # D vertical
            else:
                return self.label_to_index['D']
    
    
        # D
        if (landmarks[8][1] < landmarks[6][1] and     # index up
            landmarks[12][1] > landmarks[10][1] and   # middle down
            landmarks[16][1] > landmarks[14][1] and   # ring down
            landmarks[20][1] > landmarks[18][1]):     # pinky down
            
            ch1 = self.label_to_index['D']
            
        # D vs L
        if ch1 in [self.label_to_index['D'], self.label_to_index['L']]:

            # index up
            if landmarks[8][1] < landmarks[6][1]:

                dist_thumb_index = np.linalg.norm(
                    np.array(landmarks[4]) - np.array(landmarks[8])
                )

                # thumb horizontal extension (x difference)
                thumb_index_x_diff = abs(landmarks[4][0] - landmarks[8][0])

                # L (L shape)
                if dist_thumb_index > 50 and thumb_index_x_diff > 30:
                    ch1 = self.label_to_index['L']

                # D thumb close to index
                else:
                    ch1 = self.label_to_index['D']
            
            
        # G vs L
        if ch1 in [self.label_to_index['G'], self.label_to_index['L']]:
            
            # index direction 
            dx = abs(landmarks[8][0] - landmarks[6][0])
            dy = abs(landmarks[8][1] - landmarks[6][1])

            if dy > dx:
                ch1 = self.label_to_index['L']   
            else:
                ch1 = self.label_to_index['G'] 
        
        # U index & middle up
        if (landmarks[8][1] < landmarks[6][1] and
            landmarks[12][1] < landmarks[10][1] and
            landmarks[16][1] > landmarks[14][1] and
            landmarks[20][1] > landmarks[18][1]):

            # check closeness
            dist = np.linalg.norm(np.array(landmarks[8]) - np.array(landmarks[12]))

            if dist < 30:
                ch1 = self.label_to_index['U']
                
        # Next / B
        if (landmarks[8][1] < landmarks[6][1] and
            landmarks[12][1] < landmarks[10][1] and
            landmarks[16][1] < landmarks[14][1] and
            landmarks[20][1] < landmarks[18][1]):

            thumb_x = landmarks[4][0]
            index_x = landmarks[8][0]
            pinky_x = landmarks[20][0]

            dist_thumb_index = np.linalg.norm(
                np.array(landmarks[4]) - np.array(landmarks[8])
            )

            # NEXT 
            if (thumb_x < index_x or thumb_x > pinky_x) and dist_thumb_index > 50:
                ch1 = self.label_to_index['NEXT']

            # B
            else:
                ch1 = self.label_to_index['B']
        
        # P vs G 
        if ch1 in [self.label_to_index['P'], self.label_to_index['G']]:

            # Check index horizontal
            dx = abs(landmarks[8][0] - landmarks[6][0])
            dy = abs(landmarks[8][1] - landmarks[6][1])

            if dx > dy:

                # Check thumb position
                thumb_y = landmarks[4][1]
                index_y = landmarks[8][1]

                # G → thumb above index
                if thumb_y < index_y:
                    return self.label_to_index['G']

                # P → thumb not above
                else:
                    return self.label_to_index['P']

        # NEXT vs F vs B
        if (landmarks[8][1] < landmarks[6][1] and
            landmarks[12][1] < landmarks[10][1] and
            landmarks[16][1] < landmarks[14][1] and
            landmarks[20][1] < landmarks[18][1]):

            # distance thumb ↔ index
            dist_thumb_index = np.linalg.norm(
                np.array(landmarks[4]) - np.array(landmarks[8])
            )

            # finger straightness (C is curved)
            curvature = abs(landmarks[8][0] - landmarks[6][0]) + \
                        abs(landmarks[12][0] - landmarks[10][0])

            thumb_x = landmarks[4][0]
            index_x = landmarks[8][0]
            pinky_x = landmarks[20][0]

            # F → thumb touching index
            if dist_thumb_index < 30:
                return self.label_to_index['F']

            # NEXT → straight + thumb outside
            elif curvature < 25 and (thumb_x < index_x or thumb_x > pinky_x):
                return self.label_to_index['NEXT']

            # else B
            else:
                return self.label_to_index['B']
            
        # F vs B
        if ch1 in [self.label_to_index['F'], self.label_to_index['B']]:

            dist_thumb_index = np.linalg.norm(
                np.array(landmarks[4]) - np.array(landmarks[8])
            )

            # F  thumb touching/near index
            if dist_thumb_index < 35:
                return self.label_to_index['F']

            # B  thumb far (folded or inside)
            else:
                return self.label_to_index['B']
    
        # K 
        if ch1 in [self.label_to_index['U'], self.label_to_index['K']]:
            
            # index and middle fingers up
            if (landmarks[8][1] < landmarks[6][1] and
                landmarks[12][1] < landmarks[10][1]):
                
                # distance between index & middle (spread)
                dist = np.linalg.norm(np.array(landmarks[8]) - np.array(landmarks[12]))
                
                # thumb (between index and middle)
                thumb_x = landmarks[4][0]
                index_x = landmarks[8][0]
                middle_x = landmarks[12][0]

                if (min(index_x, middle_x) < thumb_x < max(index_x, middle_x)) and dist > 40:
                    ch1 = self.label_to_index['K']
                    
        # NEXT 
        if (landmarks[8][1] < landmarks[6][1] and
            landmarks[12][1] < landmarks[10][1] and
            landmarks[16][1] < landmarks[14][1] and
            landmarks[20][1] < landmarks[18][1]):

            # check finger straightness (C is curved)
            straight = (
                abs(landmarks[8][0] - landmarks[6][0]) < 20 and
                abs(landmarks[12][0] - landmarks[10][0]) < 20
            )

            thumb_x = landmarks[4][0]
            index_x = landmarks[8][0]
            pinky_x = landmarks[20][0]

            if straight and (thumb_x < index_x or thumb_x > pinky_x):
                ch1 = self.label_to_index['NEXT']        
        
        # NEXT vs F
        if ch1 in [self.label_to_index['NEXT'], self.label_to_index['F']]:
            index_up  = landmarks[8][1] < landmarks[6][1]
            middle_up = landmarks[12][1] < landmarks[10][1]
            ring_up   = landmarks[16][1] < landmarks[14][1]
            pinky_up  = landmarks[20][1] < landmarks[18][1]

            # -------------------------
            # F → index DOWN + others UP
            # -------------------------
            if (not index_up and middle_up and ring_up and pinky_up):
                return self.label_to_index['F']

            # -------------------------
            # NEXT → all fingers UP
            # -------------------------
            elif (index_up and middle_up and ring_up and pinky_up):

                thumb_x = landmarks[4][0]
                index_x = landmarks[8][0]
                pinky_x = landmarks[20][0]

                dist_thumb_index = np.linalg.norm(
                    np.array(landmarks[4]) - np.array(landmarks[8])
                )

                # thumb clearly outside → NEXT
                if (thumb_x < index_x or thumb_x > pinky_x) and dist_thumb_index > 50:
                    return self.label_to_index['NEXT']
        # R vs U
        if ch1 in [self.label_to_index['R'], self.label_to_index['U']]:
            
            if landmarks[8][0] > landmarks[12][0]:
                ch1 = self.label_to_index['R']
            else:
                ch1 = self.label_to_index['U']
        
        
        # T vs N/S
        if ch1 in [self.label_to_index['T'], self.label_to_index['N'], self.label_to_index['S']]:
            
            thumb_x = landmarks[4][0]
            index_x = landmarks[8][0]
            middle_x = landmarks[12][0]

            if min(index_x, middle_x) < thumb_x < max(index_x, middle_x):
                ch1 = self.label_to_index['T']
                
                
        # W vs V
        if ch1 in [self.label_to_index['W'], self.label_to_index['V']]:
            
            ring_up = landmarks[16][1] < landmarks[14][1]

            if ring_up:
                ch1 = self.label_to_index['W']
            else:
                ch1 = self.label_to_index['V']
        

        return ch1

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Status: Camera Error")
            self.video_label.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)

        h, w, c = frame.shape

        
        ###########################################################
        
        # AR TEXT 
        display_text = self.sentence if self.sentence else "..."

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(display_text, font, scale, thickness)

        # exact center
        text_x = (w - text_w) // 2
        text_y = 60 

        # shadow
        cv2.putText(frame, display_text, (text_x + 2, text_y + 2),
                    font, scale, (0, 0, 0), 4, cv2.LINE_AA)

        # main text
        cv2.putText(frame, display_text, (text_x, text_y),
                    font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # landmarks and connections
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
                    confidence = np.max(prediction)
                    
                    prob = prediction[0]
                    top3 = np.argsort(prob)[-3:][::-1]

                    ch1, ch2, ch3 = top3
                    
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                    corrected_index = self.apply_rules(ch1, ch2, ch3, landmarks)
                    
                    
                    if confidence > 0.85:
                        self.pred_queue.append(corrected_index)

                    if len(self.pred_queue) == 5:
                        most_common = max(set(self.pred_queue), key = self.pred_queue.count)
                        self.stable_prediction = self.labels[most_common]
                        self.handle_gesture(self.stable_prediction)
                        
                        
                # prediction box
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

        
        # Resize
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
    app = Application(root)
    root.mainloop()
