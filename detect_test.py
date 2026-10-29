
from ultralytics import YOLO
import cv2
import csv
from pathlib import Path

video_path = Path("videos") / "2024-11-04 12-00-31.mp4"
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Could not open video:", video_path)
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0
sample_interval = max(1, round(fps))

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

csv_path = output_dir / "traffic_counts.csv"
frame_number = 0
rows = []
latest_annotated_frame = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % sample_interval == 0:
        timestamp = frame_number / fps if fps > 0 else 0

        result = model(
            frame,
            conf=0.25,
            iou=0.45,
            imgsz=960,
            verbose=False
        )[0]

        counts = {name: 0 for name in vehicle_classes.values()}

        for box in result.boxes:
            class_id = int(box.cls[0])

            if class_id in vehicle_classes:
                counts[vehicle_classes[class_id]] += 1

        rows.append({
            "timestamp_seconds": round(timestamp, 2),
            **counts,
            "total_vehicles_detected": sum(counts.values())
        })

        latest_annotated_frame = result.plot()

        print(
            f"Time: {timestamp:.1f}s | "
            f"Vehicles: {sum(counts.values())}"
        )

    frame_number += 1

cap.release()

if latest_annotated_frame is not None:
    cv2.imwrite(
        str(output_dir / "detection_sample.jpg"),
        latest_annotated_frame
    )

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    fieldnames = [
        "timestamp_seconds",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "total_vehicles_detected"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\nVideo processing completed")
print(f"Video duration: {duration:.2f} seconds")
print(f"Frames processed: {len(rows)}")
print(f"CSV saved: {csv_path}")
print(f"Sample image saved: {output_dir / 'detection_sample.jpg'}")
