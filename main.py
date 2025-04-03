import cv2
import time
import draw
import face  # Import face.py to use its functionality

user_name = input("Enter your name: ")
cap = cv2.VideoCapture(0)
writing_mode = False
face_mode = False  # Add a flag for face filter mode

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        button_size = 40
        spacing = 10
        for i in range(len(draw.colors)):
            x_start = 10 + i * (button_size + spacing)
            y_start = 10
            x_end = x_start + button_size
            y_end = y_start + button_size
            if x_start <= x <= x_end and y_start <= y <= y_end:
                draw.change_color(i)
        
        clear_x, clear_y, clear_x_end, clear_y_end = param
        if clear_x <= x <= clear_x_end and clear_y <= y <= clear_y_end:
            draw.clear_canvas()

cv2.namedWindow("Air Writing")
frame_count = 0
MAX_FRAMES = 10000
TIMEOUT = 10 * 60
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break  

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "W: Write | Click Colors | X: Clear | F: Face Filter | Q: Quit",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Apply face filter only when face_mode is True
    if face_mode:
        frame = face.apply_face_filter(frame, user_name)  # Use face.py's function here

    if writing_mode:
        frame, clear_button_coords = draw.write_on_air(frame)  
    else:
        clear_button_coords = draw.draw_buttons(frame)

    cv2.setMouseCallback("Air Writing", mouse_callback, clear_button_coords)
    cv2.imshow("Air Writing", frame)

    key = cv2.waitKey(1) & 0xFF  
    if key == ord('w'):
        writing_mode = not writing_mode
        print(f"Writing Mode: {'ON' if writing_mode else 'OFF'}")
    elif key == ord('f'):  # Press 'F' to toggle face filter mode
        face_mode = not face_mode
        print(f"Face Filter Mode: {'ON' if face_mode else 'OFF'}")
    elif key == ord('q'):
        print("Exiting program...")
        break  

cap.release()
cv2.destroyAllWindows()
