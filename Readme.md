# Air Writing & Face Filter Application

## Overview
This project is a computer vision-based application that enables air writing using hand gestures and applies a face filter overlay. It uses OpenCV and MediaPipe for hand and face tracking.

## Features
- **Air Writing Mode**: Users can write in the air using their index finger.
- **Face Filter Mode**: Applies a fun overlay (e.g., glasses) on detected faces.
- **Color Selection**: Change the writing color dynamically.
- **Clear Canvas**: Clear the written strokes.

## File Structure
- `main.py` - The entry point of the application.
- `draw.py` - Handles air writing with hand tracking.
- `face.py` - Detects faces and overlays a glasses filter.
- `assests/` - Contains images (like glasses) used in the face filter.

## Dependencies
Make sure you have the following installed:
```bash
pip install opencv-python mediapipe numpy
```

## Usage
Run the application:
```bash
python main.py
```
### Controls
- Press `W` to toggle air writing mode.
- Click on color buttons to change writing color.
- Press `X` to clear the canvas.
- Press `F` to toggle the face filter.
- Press `Q` to quit.

## Future Improvements
- Enhance hand tracking for smoother drawing.
- Add more face filters.
- Save drawings as images.

Enjoy air writing and face filtering! 🚀

