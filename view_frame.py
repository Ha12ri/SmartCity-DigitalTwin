
import cv2
from pathlib import Path

video_path = Path("videos") / "2024-11-04 12-00-31.mp4"

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Video could not be opened")
else:
    ret, frame = cap.read()

    if ret:
        cv2.imwrite("first_frame.jpg", frame)
        print("First frame saved successfully")
        print("Frame shape:", frame.shape)
    else:
        print("Could not read frame")

cap.release()

