"""
line_crossing.py  -  Line-crossing vehicle counting for recorded traffic video
(Digital Twin for Smart City Infrastructure Monitoring - traffic prototype)

What it does
------------
1. Shows a preview of a REAL frame from the video.
2. You click two points per counting line (default: one line).
   Mouse positions are converted back to ORIGINAL video-frame pixels.
3. Tracks vehicles with YOLO + ByteTrack (persistent IDs).
4. Counts a vehicle when the bottom-centre of its box moves from one side of the
   FINITE line segment to the other (each track ID is counted at most once per line).
5. Writes, into a new timestamped folder under outputs/line_crossing/:
      line_crossing_result.mp4   annotated video (lines drawn from the same coordinates)
      vehicle_crossings.csv      one row per counted crossing
      vehicle_count_summary.csv  counts per line / vehicle type / direction
      selected_lines.json        the exact line coordinates (re-usable with --load-lines)
      line_preview.png           full-resolution frame with the lines drawn
      run_info.json              settings used (for your report)

Direction convention
--------------------
Looking at the screen, stand on P1 (first click) and face P2 (second click).
    S1 = the side on your LEFT,  S2 = the side on your RIGHT.
    S1_to_S2 = vehicle moved from the left side to the right side.
The overlay draws an arrow and the labels S1 / S2 on the line, so you can see this.

Limitations (disclose in report): see the notes printed at the end of the run.
"""

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
DEFAULT_VIDEO = Path("videos") / "2024-11-04 12-00-31.mp4"
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "line_crossing"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DIRECTIONS = ("S1_to_S2", "S2_to_S1")

# BGR colours, one per line
LINE_COLORS = [
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 165, 255),    # orange
    (0, 255, 0),      # green
    (255, 128, 128),  # light blue
]
MIN_LINE_LENGTH_PX = 30   # reject accidental tiny lines
NEAR_LINE_PX = 40         # "vehicle came near the line" diagnostic radius
MAX_LINES = 6


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thickness=1):
    """Text with a black outline so it is readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def format_time(seconds):
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:05.2f}"


def display_to_original(x, y, frame_w, frame_h, disp_w, disp_h):
    """Map a mouse position in the displayed preview to ORIGINAL frame pixels."""
    ox = int(round(x * frame_w / disp_w))
    oy = int(round(y * frame_h / disp_h))
    return min(max(ox, 0), frame_w - 1), min(max(oy, 0), frame_h - 1)


def get_screen_size():
    """
    Real screen size in pixels. Marks the process DPI-aware first so Windows
    display scaling (125 %, 150 %...) does not silently stretch the OpenCV window.
    """
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1366, 768


def point_segment_distance(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    t = 0.0 if denom == 0 else ((p[0] - ax) * dx + (p[1] - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def draw_lines(img, lines):
    """
    Draw counting lines on a full-resolution frame. The SAME function is used for
    the preview, the saved PNG and every frame of the output video, so the line
    is always drawn from the same original-frame coordinates.
    """
    for i, (p1, p2) in enumerate(lines):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        cv2.line(img, p1, p2, color, 3, cv2.LINE_AA)
        cv2.circle(img, p1, 7, color, -1, cv2.LINE_AA)
        cv2.circle(img, p2, 7, color, -1, cv2.LINE_AA)

        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if length == 0:
            continue
        mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
        # unit normal pointing toward S2 (right-hand side of P1 -> P2 on screen)
        nx = -(p2[1] - p1[1]) / length
        ny = (p2[0] - p1[0]) / length

        tail = (int(mx - nx * 25), int(my - ny * 25))
        head = (int(mx + nx * 25), int(my + ny * 25))
        cv2.arrowedLine(img, tail, head, color, 2, cv2.LINE_AA, tipLength=0.35)
        put_text(img, "S1", (int(mx - nx * 50) - 10, int(my - ny * 50) + 5), 0.55, color, 2)
        put_text(img, "S2", (int(mx + nx * 50) - 10, int(my + ny * 50) + 5), 0.55, color, 2)
        put_text(img, f"L{i + 1}", (p1[0] + 10, p1[1] - 12), 0.65, color, 2)


# ----------------------------------------------------------------------------
# Counting logic (independent of YOLO so it can be tested on its own)
# ----------------------------------------------------------------------------
class LineCounter:
    """
    Counts tracks crossing ONE finite line segment.

    side value = signed perpendicular distance (pixels) from the infinite line.
        negative -> S1 (left of P1->P2 on screen), positive -> S2 (right).
    A crossing is registered only when
        (a) the track's side flips (with a small hysteresis band to ignore jitter),
        (b) the motion segment meets the infinite line INSIDE the finite segment
            P1-P2 (projection parameter 0..1), and
        (c) the track ID has not been counted on this line before.
    """

    def __init__(self, index, p1, p2, hysteresis_px=2.0):
        self.index = index
        self.name = f"L{index + 1}"
        self.p1 = (int(p1[0]), int(p1[1]))
        self.p2 = (int(p2[0]), int(p2[1]))
        self.color = LINE_COLORS[index % len(LINE_COLORS)]
        self.hysteresis = float(hysteresis_px)
        self.length = math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])
        if self.length == 0:
            raise ValueError("Line endpoints are identical")
        self._last = {}               # track_id -> (signed side, point)
        self.counted_ids = set()
        self.near_ids = set()         # tracks that came within NEAR_LINE_PX of the segment
        self.direction_counts = Counter()
        self.stats = Counter()

    def total(self):
        return sum(self.direction_counts.values())

    def signed_side(self, p):
        (x1, y1), (x2, y2) = self.p1, self.p2
        return ((x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)) / self.length

    def update(self, track_id, point):
        """Feed one observation. Returns a dict if this observation is a counted crossing."""
        if point_segment_distance(point, self.p1, self.p2) <= NEAR_LINE_PX:
            self.near_ids.add(track_id)

        side = self.signed_side(point)
        if abs(side) < self.hysteresis:
            return None                       # on the line: keep previous side, wait

        last = self._last.get(track_id)
        self._last[track_id] = (side, point)
        if last is None or last[0] * side > 0:
            return None                       # first sighting, or same side

        last_side, last_point = last
        self.stats["side_changes"] += 1

        frac = last_side / (last_side - side)          # where the motion meets the line
        cx = last_point[0] + frac * (point[0] - last_point[0])
        cy = last_point[1] + frac * (point[1] - last_point[1])

        dx, dy = self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]
        u = ((cx - self.p1[0]) * dx + (cy - self.p1[1]) * dy) / (self.length ** 2)
        if u < 0.0 or u > 1.0:
            self.stats["changes_outside_segment"] += 1   # crossed the extension only
            return None

        if track_id in self.counted_ids:
            self.stats["repeat_changes_ignored"] += 1
            return None

        self.counted_ids.add(track_id)
        direction = "S1_to_S2" if last_side < 0 else "S2_to_S1"
        self.direction_counts[direction] += 1
        return {"direction": direction, "x": cx, "y": cy}


# ----------------------------------------------------------------------------
# Video / preview / line selection
# ----------------------------------------------------------------------------
def probe_video(video_path, preview_frame):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    reported_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps != fps or fps <= 0:
        fps = 30.0

    if preview_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read a frame from the video.")

    h, w = frame.shape[:2]
    if (w, h) != (reported_w, reported_h):
        print(f"WARNING: container reports {reported_w}x{reported_h} but decoded frames are "
              f"{w}x{h}. Using the decoded size.", flush=True)

    return frame, {"width": w, "height": h, "fps": fps, "total_frames": total}


def select_lines(frame, n_lines):
    """Interactive selection. Returns a list of ((x1, y1), (x2, y2)) in ORIGINAL pixels."""
    h, w = frame.shape[:2]
    screen_w, screen_h = get_screen_size()
    scale = min(1.0, 0.92 * screen_w / w, 0.80 * screen_h / h)
    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))
    print(f"Frame size (original): {w} x {h}", flush=True)
    print(f"Screen size: {screen_w} x {screen_h} | preview shown at {disp_w} x {disp_h} "
          f"(scale {scale:.3f}). Clicks are mapped back to original pixels.", flush=True)

    window = "Select counting line(s)"
    state = {"lines": [], "points": [], "mouse": None, "msg": ""}

    def on_mouse(event, x, y, flags, param):
        pos = display_to_original(x, y, w, h, disp_w, disp_h)
        state["mouse"] = pos
        if event == cv2.EVENT_LBUTTONDOWN and len(state["lines"]) < n_lines:
            state["points"].append(pos)
            if len(state["points"]) == 2:
                p1, p2 = state["points"]
                if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < MIN_LINE_LENGTH_PX:
                    state["msg"] = "Line too short - click two points further apart"
                else:
                    state["lines"].append((p1, p2))
                    state["msg"] = ""
                    print(f"  Line L{len(state['lines'])}: P1={p1}  P2={p2}", flush=True)
                state["points"] = []

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window, 10, 10)
    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            canvas = frame.copy()
            draw_lines(canvas, state["lines"])
            color = LINE_COLORS[len(state["lines"]) % len(LINE_COLORS)]
            for pt in state["points"]:
                cv2.circle(canvas, pt, 7, color, -1, cv2.LINE_AA)
                if state["mouse"] is not None:
                    cv2.line(canvas, pt, state["mouse"], color, 1, cv2.LINE_AA)

            if scale < 1.0:
                shown = cv2.resize(canvas, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            else:
                shown = canvas

            if len(state["lines"]) < n_lines:
                prompt = (f"Line {len(state['lines']) + 1}/{n_lines}: click point "
                          f"{len(state['points']) + 1} of 2")
            else:
                prompt = "All lines set - press C to confirm"
            put_text(shown, prompt, (10, 24), 0.6, (255, 255, 255), 1)
            put_text(shown, "C = confirm   R = reset all   U = undo   Q / Esc = cancel",
                     (10, 48), 0.5, (255, 255, 255), 1)
            if state["mouse"] is not None:
                put_text(shown, f"Cursor (original px): {state['mouse']}   frame {w}x{h}",
                         (10, 72), 0.5, (0, 255, 255), 1)
            if state["msg"]:
                put_text(shown, state["msg"], (10, 96), 0.5, (0, 0, 255), 1)

            cv2.imshow(window, shown)
            key = cv2.waitKey(20) & 0xFF

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise SystemExit("Selection window closed - cancelled.")
            if key in (ord("q"), 27):
                raise SystemExit("Cancelled.")
            elif key == ord("r"):
                state["lines"].clear()
                state["points"] = []
                state["msg"] = ""
            elif key == ord("u"):
                if state["points"]:
                    state["points"].pop()
                elif state["lines"]:
                    state["lines"].pop()
            elif key == ord("c"):
                if len(state["lines"]) == n_lines:
                    break
                state["msg"] = f"Select all {n_lines} line(s) first"
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    return list(state["lines"])


def load_lines(path, frame_w, frame_h):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if (data.get("frame_width"), data.get("frame_height")) != (frame_w, frame_h):
        raise SystemExit(
            f"Saved lines were made for {data.get('frame_width')}x{data.get('frame_height')} "
            f"but this video is {frame_w}x{frame_h}. Re-select the lines.")
    lines = []
    for item in data["lines"]:
        p1 = (int(item["p1"][0]), int(item["p1"][1]))
        p2 = (int(item["p2"][0]), int(item["p2"][1]))
        for x, y in (p1, p2):
            if not (0 <= x < frame_w and 0 <= y < frame_h):
                raise SystemExit(f"Saved point {(x, y)} is outside the frame.")
        lines.append((p1, p2))
    if not lines:
        raise SystemExit("No lines found in the file.")
    return lines


# ----------------------------------------------------------------------------
# Tracking + counting
# ----------------------------------------------------------------------------
def run_tracking(args, lines, run_dir, info):
    from ultralytics import YOLO          # imported late so selection starts quickly
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except Exception:
        ultralytics_version = "unknown"

    width, height, fps = info["width"], info["height"], info["fps"]
    total_frames = info["total_frames"]
    if args.max_frames and total_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    counters = [LineCounter(i, p1, p2, args.hysteresis) for i, (p1, p2) in enumerate(lines)]

    print("Loading model...", flush=True)
    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit("Could not re-open the video for processing.")

    video_out = run_dir / "line_crossing_result.mp4"
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit("Could not create the output video.")

    class_votes = defaultdict(Counter)      # track_id -> Counter(vehicle name)
    trails = defaultdict(lambda: deque(maxlen=40))
    flash_until = {}                        # track_id -> frame number
    ticker = deque(maxlen=4)                # (text, expire_frame)
    events = []
    frame_number = 0
    interrupted = False
    t0 = time.time()

    coord_text = [f"{c.name} ({c.p1[0]},{c.p1[1]})-({c.p2[0]},{c.p2[1]})" for c in counters]

    print("Processing... (Ctrl+C stops early and still saves partial results)", flush=True)
    try:
        while True:
            if args.max_frames and frame_number >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                raise SystemExit(f"Frame {frame_number} is {frame.shape[1]}x{frame.shape[0]}, "
                                 f"expected {width}x{height}.")

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
            )[0]

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().tolist()
                ids = boxes.id.int().cpu().tolist()
                clss = boxes.cls.int().cpu().tolist()

                for box, track_id, class_id in zip(xyxy, ids, clss):
                    if class_id not in VEHICLE_CLASSES:
                        continue
                    bx1, by1, bx2, by2 = box
                    ref = ((bx1 + bx2) / 2.0, float(by2))     # bottom-centre = ground contact
                    class_votes[track_id][VEHICLE_CLASSES[class_id]] += 1
                    vehicle = class_votes[track_id].most_common(1)[0][0]
                    trails[track_id].append((int(ref[0]), int(ref[1])))

                    for counter in counters:
                        hit = counter.update(track_id, ref)
                        if hit:
                            timestamp = frame_number / fps
                            events.append({
                                "line_id": counter.name,
                                "frame_number": frame_number,
                                "timestamp_seconds": round(timestamp, 2),
                                "video_time": format_time(timestamp),
                                "track_id": track_id,
                                "vehicle_type": vehicle,
                                "direction": hit["direction"],
                                "crossing_x": round(hit["x"], 1),
                                "crossing_y": round(hit["y"], 1),
                            })
                            flash_until[track_id] = frame_number + 20
                            ticker.append((f"{counter.name}: {vehicle} ID {track_id} "
                                           f"{hit['direction']}", frame_number + int(fps * 2)))

                    # ---- draw this vehicle ----
                    flashing = flash_until.get(track_id, -1) >= frame_number
                    box_color = (0, 0, 255) if flashing else (0, 200, 0)
                    cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), box_color, 2)
                    put_text(frame, f"{vehicle} #{track_id}",
                             (int(bx1), max(18, int(by1) - 6)), 0.5, box_color, 1)
                    trail = list(trails[track_id])
                    for a, b in zip(trail, trail[1:]):
                        cv2.line(frame, a, b, (255, 200, 0), 1, cv2.LINE_AA)
                    cv2.circle(frame, (int(ref[0]), int(ref[1])), 4, (0, 0, 255), -1)

            # ---- lines + HUD (all from the same original-frame coordinates) ----
            draw_lines(frame, lines)

            y = 28
            for counter in counters:
                text = (f"{counter.name}: {counter.total()}   "
                        f"(S1>S2 {counter.direction_counts['S1_to_S2']} | "
                        f"S2>S1 {counter.direction_counts['S2_to_S1']})")
                put_text(frame, text, (15, y), 0.7, counter.color, 2)
                y += 28
            for text, expire in ticker:
                if expire >= frame_number:
                    put_text(frame, text, (15, y), 0.55, (255, 255, 255), 1)
                    y += 22

            put_text(frame, f"Frame {frame_number}   t = {format_time(frame_number / fps)}",
                     (15, height - 15 - 22 * len(coord_text)), 0.55, (255, 255, 255), 1)
            for i, text in enumerate(coord_text):
                put_text(frame, text, (15, height - 15 - 22 * (len(coord_text) - 1 - i)),
                         0.5, counters[i].color, 1)

            writer.write(frame)
            frame_number += 1

            if frame_number % 100 == 0:
                elapsed = time.time() - t0
                rate = frame_number / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_number) / rate if rate > 0 and total_frames > 0 else 0
                counts = " | ".join(f"{c.name}={c.total()}" for c in counters)
                print(f"Frame {frame_number}/{total_frames} | {rate:.1f} fps | "
                      f"ETA {eta / 60:.1f} min | crossings: {counts}", flush=True)

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted - saving partial results.", flush=True)
    finally:
        cap.release()
        writer.release()

    # ---------------- write outputs ----------------
    events_csv = run_dir / "vehicle_crossings.csv"
    fields = ["line_id", "frame_number", "timestamp_seconds", "video_time", "track_id",
              "vehicle_type", "direction", "crossing_x", "crossing_y"]
    with open(events_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(events)

    counts = Counter((e["line_id"], e["vehicle_type"], e["direction"]) for e in events)
    summary_csv = run_dir / "vehicle_count_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["line_id", "vehicle_type", "S1_to_S2", "S2_to_S1", "total"])
        for counter in counters:
            for vehicle in VEHICLE_CLASSES.values():
                a = counts[(counter.name, vehicle, "S1_to_S2")]
                b = counts[(counter.name, vehicle, "S2_to_S1")]
                w.writerow([counter.name, vehicle, a, b, a + b])
            a = counter.direction_counts["S1_to_S2"]
            b = counter.direction_counts["S2_to_S1"]
            w.writerow([counter.name, "all_vehicles", a, b, a + b])

    run_info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "video": str(args.video),
        "frame_width": width,
        "frame_height": height,
        "fps": fps,
        "frames_processed": frame_number,
        "interrupted": interrupted,
        "model": args.model,
        "tracker": "bytetrack.yaml",
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "hysteresis_px": args.hysteresis,
        "reference_point": "bottom-centre of bounding box",
        "lines": [{"id": c.name, "p1": list(c.p1), "p2": list(c.p2)} for c in counters],
        "ultralytics_version": ultralytics_version,
        "opencv_version": cv2.__version__,
    }
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    # ---------------- console report ----------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED" + (" (PARTIAL - interrupted)" if interrupted else ""))
    print("=" * 70)
    print(f"Frames processed          : {frame_number}  ({frame_number / fps:.1f} s of video)")
    print(f"Distinct vehicle track IDs: {len(class_votes)}  (NOT unique vehicles)")
    for c in counters:
        s = c.stats
        print(f"\nLine {c.name}: P1={c.p1}  P2={c.p2}  length={c.length:.0f}px  (original-frame pixels)")
        print(f"  Counted crossings : {c.total()}  "
              f"(S1_to_S2: {c.direction_counts['S1_to_S2']}, S2_to_S1: {c.direction_counts['S2_to_S1']})")
        print(f"  Track-side changes: {s['side_changes']} total = "
              f"{c.total()} counted + {s['changes_outside_segment']} outside segment + "
              f"{s['repeat_changes_ignored']} repeat (already counted)")
        print(f"  Distinct tracks that came within {NEAR_LINE_PX}px of the line: {len(c.near_ids)}")
        by_type = Counter(e["vehicle_type"] for e in events if e["line_id"] == c.name)
        print(f"  By type           : {dict(by_type)}")
        if c.total() == 0 and len(c.near_ids) == 0:
            print("  HINT: no vehicle track came near this line - it may not lie on any vehicle path.")
        elif c.total() == 0:
            print("  HINT: tracks came near the line but none switched sides inside the segment - "
                  "check line placement/length, or track IDs changing at the line.")
        elif s["changes_outside_segment"] > c.total():
            print("  HINT: many side changes were outside the segment - the line may be too short "
                  "for the road width.")

    print(f"\nOutput folder : {run_dir}")
    print(f"  Video       : {video_out.name}")
    print(f"  Events CSV  : {events_csv.name}")
    print(f"  Summary CSV : {summary_csv.name}")
    print("\nNOT VALIDATED until compared with a manual count. Counts can be affected by "
          "occlusion, ID switches, missed detections and wide-angle perspective. Do not add "
          "counts across different lines and call the sum 'unique vehicles'.")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Line-crossing vehicle counting (YOLO + ByteTrack)")
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lines", type=int, default=1,
                   help=f"number of counting lines to select (1-{MAX_LINES}, default 1)")
    p.add_argument("--load-lines", type=Path, default=None,
                   help="reuse a selected_lines.json instead of clicking again")
    p.add_argument("--select-only", action="store_true",
                   help="select lines, save preview + JSON, then stop (no tracking)")
    p.add_argument("--preview-frame", type=int, default=0,
                   help="frame index to show in the selection preview (default 0)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="process only the first N frames (0 = whole video); good for quick tests")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=960,
                   help="YOLO input size (960 like the dashboard; use 640 to match tracking_test.py)")
    p.add_argument("--hysteresis", type=float, default=2.0,
                   help="pixels either side of the line ignored to suppress jitter (default 2)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return p.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.lines <= MAX_LINES:
        raise SystemExit(f"--lines must be between 1 and {MAX_LINES}")
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video.resolve()}")

    frame, info = probe_video(args.video, args.preview_frame)
    print(f"Video: {args.video}")
    print(f"Decoded frame size: {info['width']} x {info['height']} | "
          f"FPS: {info['fps']:.3f} | Frames: {info['total_frames']}", flush=True)

    run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.load_lines:
        lines = load_lines(args.load_lines, info["width"], info["height"])
        print(f"Loaded {len(lines)} line(s) from {args.load_lines}")
    else:
        lines = select_lines(frame, args.lines)

    print("\nSelected line coordinates (ORIGINAL frame pixels, used for drawing AND counting):")
    for i, (p1, p2) in enumerate(lines):
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        print(f"  L{i + 1}: P1={p1}  P2={p2}  length={length:.0f}px")

    with open(run_dir / "selected_lines.json", "w", encoding="utf-8") as f:
        json.dump({
            "frame_width": info["width"],
            "frame_height": info["height"],
            "lines": [{"id": f"L{i + 1}", "p1": list(p1), "p2": list(p2)}
                      for i, (p1, p2) in enumerate(lines)],
        }, f, indent=2)

    check = frame.copy()
    draw_lines(check, lines)
    cv2.imwrite(str(run_dir / "line_preview.png"), check)
    print(f"Saved: {run_dir / 'selected_lines.json'}")
    print(f"Saved: {run_dir / 'line_preview.png'}  (compare with frame 0 of the output video)")

    if args.select_only:
        print("--select-only given: stopping before tracking.")
        return

    run_tracking(args, lines, run_dir, info)


if __name__ == "__main__":
    main()