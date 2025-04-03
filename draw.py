import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
color_index = 0
draw_color = colors[color_index]
canvas = np.zeros((480, 640, 3), np.uint8)
drawing = False  

def change_color(index):
    global draw_color, color_index
    color_index = index
    draw_color = colors[color_index]

def clear_canvas():
    global canvas
    canvas = np.zeros((480, 640, 3), np.uint8)

def draw_buttons(frame):
    button_size = 40
    spacing = 10
    for i, color in enumerate(colors):
        x_start = 10 + i * (button_size + spacing)
        y_start = 10
        cv2.rectangle(frame, (x_start, y_start), (x_start + button_size, y_start + button_size), color, -1)
        if i == color_index:
            cv2.rectangle(frame, (x_start, y_start), (x_start + button_size, y_start + button_size), (255, 255, 255), 2)

    # Add Clear Button
    clear_x, clear_y = 10 + len(colors) * (button_size + spacing), 10
    cv2.rectangle(frame, (clear_x, clear_y), (clear_x + button_size, clear_y + button_size), (0, 0, 0), -1)
    cv2.putText(frame, "X", (clear_x + 10, clear_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return (clear_x, clear_y, clear_x + button_size, clear_y + button_size)

def is_index_finger_up(landmarks):
    tip = landmarks[8].y  
    dip = landmarks[7].y  
    pip = landmarks[6].y  
    return tip < dip < pip  

def write_on_air(frame):
    global draw_color, canvas, drawing

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            if is_index_finger_up(hand_landmarks.landmark):
                drawing = True
                cv2.circle(frame, (x, y), 10, draw_color, -1)
                cv2.circle(canvas, (x, y), 5, draw_color, -1)
            else:
                drawing = False

    combined = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
    clear_button_coords = draw_buttons(combined)
    return combined, clear_button_coords
