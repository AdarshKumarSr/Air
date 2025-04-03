import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load assets
script_dir = os.path.dirname(os.path.abspath(__file__))
glasses_path = os.path.join(script_dir, "assests/glasses.png")

glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
if glasses is None:
    raise ValueError("Error: Glasses image failed to load! Check the path.")

# def apply_face_filter(frame, user_name):
#     h, w, _ = frame.shape
#     with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_detection.process(rgb_frame)

#         if results.detections:
#             for detection in results.detections:
#                 bbox = detection.location_data.relative_bounding_box
#                 x, y, w_rel, h_rel = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)

#                 # Overlay glasses properly
#                 if glasses is not None:
#                     glasses_resized = cv2.resize(glasses, (w_rel, int(h_rel * 0.3)))

#                     # Split the glasses image into color and alpha channels
#                     glasses_rgb = glasses_resized[:, :, :3]  # Take only the RGB channels
#                     glasses_alpha = glasses_resized[:, :, 3]  # Take the alpha (transparency) channel

#                     # Ensure the glasses overlay fits inside the frame bounds
#                     y1, y2 = y, y + glasses_resized.shape[0]
#                     x1, x2 = x, x + glasses_resized.shape[1]

#                     # Create a mask from the alpha channel
#                     alpha_mask = glasses_alpha / 255.0

#                     # Blend the glasses onto the frame using the alpha mask
#                     for c in range(0, 3):  # Iterate over the 3 color channels (BGR)
#                         frame[y1:y2, x1:x2, c] = (1.0 - alpha_mask) * frame[y1:y2, x1:x2, c] + \
#                             alpha_mask * glasses_rgb[:, :, c]

#                 # Display name
#                 cv2.putText(frame, f"Hey {user_name}!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     return frame

def apply_face_filter(frame, user_name):
    h, w, _ = frame.shape
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y, w_rel, h_rel = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)

                # Overlay glasses properly
                if glasses is not None:
                    # Scale factor to increase the size of the glasses
                    scale_factor = 1.1  # Change this to increase/decrease size
                    glasses_resized = cv2.resize(glasses, 
                                                 (int(w_rel * scale_factor), int(h_rel * scale_factor * 0.3)))

                    # Split the glasses image into color and alpha channels
                    glasses_rgb = glasses_resized[:, :, :3]  # Take only the RGB channels
                    glasses_alpha = glasses_resized[:, :, 3]  # Take the alpha (transparency) channel

                    # Ensure the glasses overlay fits inside the frame bounds
                    y1, y2 = y, y + glasses_resized.shape[0]
                    x1, x2 = x, x + glasses_resized.shape[1]

                    # Create a mask from the alpha channel
                    alpha_mask = glasses_alpha / 255.0

                    # Blend the glasses onto the frame using the alpha mask
                    for c in range(0, 3):  # Iterate over the 3 color channels (BGR)
                        frame[y1:y2, x1:x2, c] = (1.0 - alpha_mask) * frame[y1:y2, x1:x2, c] + \
                            alpha_mask * glasses_rgb[:, :, c]

                # Display name
                cv2.putText(frame, f"Hey {user_name}!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame
