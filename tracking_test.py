
from ultralytics import YOLO
import cv2
from pathlib import Path

video_path = Path("videos") / "2024-11-04 12-00-31.mp4"
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print("Loading model...", flush=True)
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Could not open video")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(output_dir / "tracking_test.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

seen_ids = set()
frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )[0]

    if result.boxes.id is not None:
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for track_id, class_id in zip(ids, classes):
            if class_id in (2, 3, 5, 7):
                seen_ids.add(track_id)

    writer.write(result.plot())
    frame_number += 1

    if frame_number % 30 == 0:
        print(
            f"Processed {frame_number}/{total_frames} frames",
            flush=True
        )

cap.release()
writer.release()

print("Tracking completed", flush=True)
print("Frames processed:", frame_number)
print("Distinct track IDs observed:", len(seen_ids))
print("Output: outputs/tracking_test.mp4")
print("Note: Track IDs are not guaranteed to represent unique vehicles.")
