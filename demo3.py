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

# Sentence building variables (like final_pred.py)
sentence = ""
prev_ch = ""
ten_prev_chars = [" "] * 10
char_count = 0
next_detected = False

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
                
                pts = points
                
                prob = np.array(model.predict(hand_img, verbose=0)[0], dtype='float32')
                ch1 = np.argmax(prob, axis=0)
                prob[ch1] = 0
                ch2 = np.argmax(prob, axis=0)
                prob[ch2] = 0
                ch3 = np.argmax(prob, axis=0)

                pl = [ch1, ch2]  # Confusion pair
                
                def distance(x, y):
                    return np.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))
                
                
                # condition for [Aemnst]
                l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
                    [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
                    [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
                if pl in l:
                    if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][
                        1]):
                        ch1 = 0

                # condition for [o][s]
                l = [[2, 2], [2, 1]]
                if pl in l:
                    if (pts[5][0] < pts[4][0]):
                        ch1 = 0
                        print("++++++++++++++++++")
                        # print("00000")

                # condition for [c0][aemnst]
                l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][
                        0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
                        ch1 = 2

                # condition for [c0][aemnst]
                l = [[6, 0], [6, 6], [6, 2]]
                pl = [ch1, ch2]
                if pl in l:
                    if distance(pts[8], pts[16]) < 52:
                        ch1 = 2


                # condition for [gh][bdfikruvw]
                l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
                pl = [ch1, ch2]

                if pl in l:
                    if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][
                        0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
                        ch1 = 3



                # con for [gh][l]
                l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[4][0] > pts[0][0]:
                        ch1 = 3

                # con for [gh][pqz]
                l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[2][1] + 15 < pts[16][1]:
                        ch1 = 3

                # con for [l][x]
                l = [[6, 4], [6, 1], [6, 2]]
                pl = [ch1, ch2]
                if pl in l:
                    if distance(pts[4], pts[11]) > 55:
                        ch1 = 4

                # con for [l][d]
                l = [[1, 4], [1, 6], [1, 1]]
                pl = [ch1, ch2]
                if pl in l:
                    if (distance(pts[4], pts[11]) > 50) and (
                            pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                            pts[20][1]):
                        ch1 = 4

                # con for [l][gh]
                l = [[3, 6], [3, 4]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[4][0] < pts[0][0]):
                        ch1 = 4

                # con for [l][c0]
                l = [[2, 2], [2, 5], [2, 4]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[1][0] < pts[12][0]):
                        ch1 = 4

                # con for [l][c0]
                l = [[2, 2], [2, 5], [2, 4]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[1][0] < pts[12][0]):
                        ch1 = 4

                # con for [gh][z]
                l = [[3, 6], [3, 5], [3, 4]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][
                        1]) and pts[4][1] > pts[10][1]:
                        ch1 = 5

                # con for [gh][pq]
                l = [[3, 2], [3, 1], [3, 6]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][
                        1] + 17 > pts[20][1]:
                        ch1 = 5

                # con for [l][pqz]
                l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[4][0] > pts[0][0]:
                        ch1 = 5

                # con for [pqz][aemnst]
                l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
                        ch1 = 5

                # con for [pqz][yj]
                l = [[5, 7], [5, 2], [5, 6]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[3][0] < pts[0][0]:
                        ch1 = 7

                # con for [l][yj]
                l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[6][1] < pts[8][1]:
                        ch1 = 7

                # con for [x][yj]
                l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[18][1] > pts[20][1]:
                        ch1 = 7

                # condition for [x][aemnst]
                l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[5][0] > pts[16][0]:
                        ch1 = 6


                # condition for [yj][x]
                print("2222  ch1=+++++++++++++++++", ch1, ",", ch2)
                l = [[7, 2]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
                        ch1 = 6

                # condition for [c0][x]
                l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
                pl = [ch1, ch2]
                if pl in l:
                    if distance(pts[8], pts[16]) > 50:
                        ch1 = 6

                # con for [l][x]

                l = [[4, 6], [4, 2], [4, 1], [4, 4]]
                pl = [ch1, ch2]
                if pl in l:
                    if distance(pts[4], pts[11]) < 60:
                        ch1 = 6

                # con for [x][d]
                l = [[1, 4], [1, 6], [1, 0], [1, 2]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[5][0] - pts[4][0] - 15 > 0:
                        ch1 = 6

                # con for [b][pqz]
                l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
                    [6, 3], [6, 4], [7, 5], [7, 2]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][
                        1]):
                        ch1 = 1

                # con for [f][pqz]
                l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
                    [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and
                            pts[18][1] > pts[20][1]):
                        ch1 = 1

                l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and
                            pts[18][1] > pts[20][1]):
                        ch1 = 1

                # con for [d][pqz]
                fg = 19
                # print("_________________ch1=",ch1," ch2=",ch2)
                l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
                pl = [ch1, ch2]
                if pl in l:
                    if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                        pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[4][1] > pts[14][1]):
                        ch1 = 1

                l = [[4, 1], [4, 2], [4, 4]]
                pl = [ch1, ch2]
                if pl in l:
                    if (distance(pts[4], pts[11]) < 50) and (
                            pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                            pts[20][1]):
                        ch1 = 1

                l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
                pl = [ch1, ch2]
                if pl in l:
                    if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                        pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1] < pts[4][1]):
                        ch1 = 1

                l = [[6, 6], [6, 4], [6, 1], [6, 2]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[5][0] - pts[4][0] - 15 < 0:
                        ch1 = 1

                # con for [i][pqz]
                l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
                pl = [ch1, ch2]
                if pl in l:
                    if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                        pts[18][1] > pts[20][1])):
                        ch1 = 1

                # con for [yj][bfdi]
                l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[4][0] < pts[5][0] + 15) and (
                    (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                    pts[18][1] > pts[20][1])):
                        ch1 = 7

                # con for [uvr]
                l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
                pl = [ch1, ch2]
                if pl in l:
                    if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and
                        pts[18][1] < pts[20][1])) and pts[4][1] > pts[14][1]:
                        ch1 = 1

                # con for [w]
                fg = 13
                l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
                pl = [ch1, ch2]
                if pl in l:
                    if not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0] and pts[0][0] + fg < pts[16][0] and
                            pts[0][0] + fg < pts[20][0]) and not (
                            pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][
                        0]) and distance(pts[4], pts[11]) < 50:
                        ch1 = 1

                # con for [w]

                l = [[5, 0], [5, 5], [0, 1]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
                        ch1 = 1
                        
                        
                        
                #######################
                
                if ch1 == 0:
                    ch1 = 'A'
                    if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
                        ch1 = 'A'
                    if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][
                        0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
                        ch1 = 'T'
                    if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
                        ch1 = 'E'
                    if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
                        ch1 = 'M'
                    if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
                        ch1 = 'N'

                if ch1 == 2:
                    if distance(pts[12], pts[4]) > 42:
                        ch1 = 'C'
                    else:
                        ch1 = 'O'

                if ch1 == 3:
                    if (distance(pts[8], pts[12])) > 72:
                        ch1 = 'G'
                    else:
                        ch1 = 'H'

                if ch1 == 7:
                    if distance(pts[8], pts[4]) > 42:
                        ch1 = 'Y'
                    else:
                        ch1 = 'J'

                if ch1 == 4:
                    ch1 = 'L'

                if ch1 == 6:
                    ch1 = 'X'

                if ch1 == 5:
                    if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                        if pts[8][1] < pts[5][1]:
                            ch1 = 'Z'
                        else:
                            ch1 = 'Q'
                    else:
                        ch1 = 'P'

                if ch1 == 1:
                    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][
                        1]):
                        ch1 = 'B'
                    if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][
                        1]):
                        ch1 = 'D'
                    if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][
                        1]):
                        ch1 = 'F'
                    if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][
                        1]):
                        ch1 = 'I'
                    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][
                        1]):
                        ch1 = 'W'
                    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][
                        1]) and pts[4][1] < pts[9][1]:
                        ch1 = 'K'
                    if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (
                            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                            pts[20][1]):
                        ch1 = 'U'
                    if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (
                            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                            pts[20][1]) and (pts[4][1] > pts[9][1]):
                        ch1 = 'V'

                    if (pts[8][0] > pts[12][0]) and (
                            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                            pts[20][1]):
                        ch1 = 'R'

                if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
                    if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1=" "



                print(pts[4][0] < pts[5][0])
                if ch1 == 'E' or ch1=='Y' or ch1=='B':
                    if (pts[4][0] < pts[5][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1="next"


                if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
                    if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and (pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and pts[4][1] < pts[16][1] and pts[4][1] < pts[20][1]) and (pts[4][1] < pts[6][1] and pts[4][1] < pts[10][1] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]):
                        ch1 = 'Backspace'



                 
                # -------------------------------
                # STABILIZATION (Queue stores INT indices only)
                # -------------------------------
                if confidence > 0.7:
                    # Convert string letters → int indices
                    if isinstance(ch1, str):
                        try:
                            ch1_index = list(class_indices.keys())[list(class_indices.values()).index(ch1.upper())]
                        except ValueError:
                            ch1_index = predicted_class_index  # Fallback to original
                    else:
                        ch1_index = ch1
                    
                    pred_queue.append(ch1_index)  # Always INT

                if len(pred_queue) == 10:
                    most_common = max(set(pred_queue), key=pred_queue.count)  # INT
                    stable_prediction = labels[most_common]  # Safe!

                predicted_text = stable_prediction if stable_prediction else labels[predicted_class_index]

                # NEXT GESTURE HANDLING (exact final_pred.py logic)
                if ch1 == "next" and prev_ch != "next":
                    if ten_prev_chars[(char_count-2) % 10] != "next":
                        prev_char = ten_prev_chars[(char_count-2) % 10]
                        if prev_char != "Backspace":
                            sentence += prev_char
                    prev_ch = "next"
                    next_detected = True
                elif ch1 == "Backspace":
                    if len(sentence) > 0:
                        sentence = sentence[:-1]
                elif ch1 == " ":
                    sentence += " "
                else:
                    prev_ch = ch1
                
                # Update history (only letters/space, skip controls)
                if ch1 not in ["next", "Backspace"] and isinstance(ch1, str) and ch1.isalpha():
                    ten_prev_chars[char_count % 10] = ch1.upper()
                    char_count += 1


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
