import json
import os
import re
import shutil
import subprocess
import sys
import zlib
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart City Traffic Twin", layout="wide")

# ---------------------------------------------------------------------------
# Paths (absolute, independent of the terminal's working directory)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"
LINE_OUTPUT_DIR = OUTPUT_DIR / "line_crossing"
UPLOAD_DIR = OUTPUT_DIR / "uploaded_videos"
INPUT_DIR = OUTPUT_DIR / "dashboard_inputs"
LINE_SCRIPT = BASE_DIR / "line_crossing.py"
MODEL_PATH = BASE_DIR / "yolo11n.pt"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_LINES = 4
MIN_LINE_LENGTH_PX = 30
PROGRESS_RE = re.compile(r"Frame (\d+)/(\d+)")

# BGR colours - same order as line_crossing.py
LINE_COLORS = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255)]

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

st.title("Smart City Digital Twin")
st.subheader("Intelligent Traffic Monitoring System")


@st.cache_resource
def load_model():
    from ultralytics import YOLO  # imported here so the other tabs work without it
    return YOLO(str(MODEL_PATH))


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------
def list_project_videos():
    if not VIDEO_DIR.is_dir():
        return []
    return sorted(
        (p for p in VIDEO_DIR.iterdir()
         if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda p: p.name.lower()
    )


def save_upload(uploaded):
    """Save an uploaded video to disk so line_crossing.py can read it."""
    data = uploaded.getbuffer()
    checksum = zlib.crc32(data) & 0xFFFFFFFF
    original = Path(uploaded.name)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{original.stem}_{checksum:08x}{original.suffix or '.mp4'}"
    if not dest.exists() or dest.stat().st_size != len(data):
        dest.write_bytes(data)
    return dest


def video_source_picker(key):
    """Returns (path, display_name) or (None, None)."""
    mode = st.radio(
        "Video source",
        ["Project video (videos/ folder)", "Upload a video"],
        horizontal=True,
        key=f"{key}_mode"
    )

    if mode.startswith("Project"):
        files = list_project_videos()
        if not files:
            st.warning(f"No videos found in {VIDEO_DIR}. Put a video there or use Upload.")
            return None, None
        choice = st.selectbox(
            "Choose a video",
            files,
            format_func=lambda p: p.name,
            key=f"{key}_choice"
        )
        return choice, choice.name

    uploaded = st.file_uploader(
        "Upload a traffic video",
        type=[e.lstrip(".") for e in sorted(VIDEO_EXTENSIONS)],
        key=f"{key}_upload"
    )
    if uploaded is None:
        return None, None
    return save_upload(uploaded), uploaded.name


@st.cache_data(show_spinner=False)
def get_video_info(path_str, mtime):
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]          # decoded size = what line_crossing.py will see
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    return {"width": w, "height": h, "fps": fps, "total_frames": total}


@st.cache_data(show_spinner=False, max_entries=20)
def read_frame_bgr(path_str, mtime, index):
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        return None
    if index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw_preview_lines(frame, lines):
    """Draw the chosen lines on a full-resolution frame (original pixel coordinates)."""
    for i, (p1, p2) in enumerate(lines):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)
        cv2.circle(frame, p1, 7, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p2, 7, color, -1, cv2.LINE_AA)
        length = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        if length == 0:
            continue
        mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
        nx = -(p2[1] - p1[1]) / length      # unit normal toward S2
        ny = (p2[0] - p1[0]) / length
        cv2.arrowedLine(
            frame,
            (int(mx - nx * 25), int(my - ny * 25)),
            (int(mx + nx * 25), int(my + ny * 25)),
            color, 2, cv2.LINE_AA, tipLength=0.35
        )
        for label, sign in (("S1", -50), ("S2", 50)):
            pos = (int(mx + nx * sign) - 10, int(my + ny * sign) + 5)
            cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"L{i + 1} P1", (p1[0] + 10, p1[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"L{i + 1} P1", (p1[0] + 10, p1[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def safe_read_csv(path):
    """Returns (DataFrame or None, error message or None). Empty files are handled."""
    try:
        return pd.read_csv(path), None
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), None
    except Exception as error:
        return None, str(error)


# ---------------------------------------------------------------------------
# Sampled-frame analysis (original behaviour, now takes a file path)
# ---------------------------------------------------------------------------
def run_sampled_analysis(video_path, display_name):
    cap = None

    try:
        model = load_model()
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            st.error("Could not open this video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            st.error("Video duration or FPS could not be read.")
            return

        interval = max(1, round(fps))
        rows = []
        latest_image = None
        progress = st.progress(0)
        status = st.empty()
        frame_number = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_number % interval == 0:
                timestamp = frame_number / fps

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

                total = sum(counts.values())

                if total >= 10:
                    level = "High detected vehicle presence"
                elif total >= 6:
                    level = "Moderate detected vehicle presence"
                else:
                    level = "Low detected vehicle presence"

                rows.append({
                    "timestamp_seconds": round(timestamp, 2),
                    **counts,
                    "total_vehicles_detected": total,
                    "traffic_level": level
                })

                latest_image = result.plot()

            frame_number += 1

            if frame_number % 10 == 0 or frame_number == total_frames:
                progress.progress(min(frame_number / total_frames, 1.0))
                status.text(f"Processing frame {frame_number}/{total_frames}")

        cap.release()
        cap = None

        if not rows or latest_image is None:
            st.error("No frames could be analysed.")
            return

        st.session_state["analysis_df"] = pd.DataFrame(rows)
        st.session_state["analysis_image"] = cv2.cvtColor(
            latest_image, cv2.COLOR_BGR2RGB
        )
        st.session_state["analysis_name"] = display_name

        status.success("Video analysis completed!")

    except Exception as error:
        st.error(f"Analysis failed: {error}")

    finally:
        if cap is not None:
            cap.release()


def render_sampled_results():
    df = st.session_state["analysis_df"]

    st.divider()
    st.header("Sampled-Frame Analysis")
    st.caption(f"Video: {st.session_state['analysis_name']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples Analysed", len(df))
    c2.metric(
        "Average Detected",
        f"{df['total_vehicles_detected'].mean():.2f}"
    )
    c3.metric(
        "Maximum Detected",
        int(df["total_vehicles_detected"].max())
    )
    c4.metric(
        "Minimum Detected",
        int(df["total_vehicles_detected"].min())
    )

    st.subheader("Vehicle Counts Over Time")
    st.line_chart(
        df.set_index("timestamp_seconds")[
            ["car", "motorcycle", "bus", "truck"]
        ]
    )

    st.subheader("Detected Vehicle Presence Levels")
    st.bar_chart(df["traffic_level"].value_counts())

    st.subheader("Latest Sampled Frame")
    st.image(
        st.session_state["analysis_image"],
        caption="YOLO vehicle detections",
        width="stretch"
    )

    st.subheader("Detailed Results")
    st.dataframe(df, width="stretch")

    st.download_button(
        "Download Analysis CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="traffic_analysis.csv",
        mime="text/csv"
    )

    st.caption(
        "Counts represent vehicles detected in sampled frames, "
        "not unique vehicles. Traffic levels use provisional "
        "thresholds and are not validated congestion predictions."
    )


# ---------------------------------------------------------------------------
# Running the existing line_crossing.py (unchanged) as a subprocess
# ---------------------------------------------------------------------------
def list_run_dirs():
    """Run folders that contain vehicle_crossings.csv, newest first."""
    if not LINE_OUTPUT_DIR.is_dir():
        return []
    runs = [
        p for p in LINE_OUTPUT_DIR.iterdir()
        if p.is_dir() and (p / "vehicle_crossings.csv").is_file()
    ]
    return sorted(
        runs,
        key=lambda p: (p / "vehicle_crossings.csv").stat().st_mtime,
        reverse=True
    )


def run_line_crossing(video_path, lines, max_frames, imgsz, progress, status):
    """
    Writes the selected lines to a JSON file and runs:
        python line_crossing.py --video ... --load-lines <json> --output-dir ...
    --load-lines makes the script skip its OpenCV selection window.
    Returns a dict describing the outcome.
    """
    info = get_video_info(str(video_path), video_path.stat().st_mtime)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    LINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines_file = INPUT_DIR / f"lines_{datetime.now():%Y%m%d_%H%M%S}.json"
    payload = {
        "frame_width": info["width"],
        "frame_height": info["height"],
        "lines": [
            {"id": f"L{i + 1}", "p1": list(p1), "p2": list(p2)}
            for i, (p1, p2) in enumerate(lines)
        ],
    }
    lines_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    before = {p.name for p in LINE_OUTPUT_DIR.iterdir() if p.is_dir()}

    command = [
        sys.executable, "-u", str(LINE_SCRIPT),
        "--video", str(video_path),
        "--model", str(MODEL_PATH),
        "--load-lines", str(lines_file),
        "--output-dir", str(LINE_OUTPUT_DIR),
        "--imgsz", str(imgsz),
    ]
    if max_frames > 0:
        command += ["--max-frames", str(max_frames)]

    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    log = []
    process = None
    returncode = None

    try:
        process = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        for raw in process.stdout:
            line = raw.rstrip()
            log.append(line)
            match = PROGRESS_RE.search(line)
            if match and int(match.group(2)) > 0:
                progress.progress(min(int(match.group(1)) / int(match.group(2)), 1.0))
                status.text(line[:160])
            elif line.strip():
                status.text(line[:160])
        returncode = process.wait()
    except OSError as error:
        return {"ok": False, "run": None, "log": "\n".join(log),
                "message": f"Could not start line_crossing.py: {error}"}
    finally:
        # If Streamlit interrupts this script (user touched a widget), stop the child process.
        if process is not None and process.poll() is None:
            process.terminate()

    log_text = "\n".join(log)
    new_runs = [
        p for p in LINE_OUTPUT_DIR.iterdir()
        if p.is_dir() and p.name not in before and (p / "vehicle_crossings.csv").is_file()
    ]

    if returncode != 0:
        return {"ok": False, "run": None, "log": log_text,
                "message": f"line_crossing.py exited with code {returncode}."}
    if not new_runs:
        return {"ok": False, "run": None, "log": log_text,
                "message": "The script finished but no new run folder with "
                           "vehicle_crossings.csv was created."}

    newest = max(new_runs, key=lambda p: p.name)
    return {"ok": True, "run": newest.name, "log": log_text,
            "message": f"Line-crossing analysis completed: {newest.name}"}


def convert_for_browser(source, destination):
    """Optional: H.264 copy for the browser player. Needs ffmpeg on PATH."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg was not found on PATH, so a browser-playable copy cannot be made."
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", str(source), "-an", "-vcodec", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)],
            capture_output=True, text=True, timeout=1800
        )
    except Exception as error:
        return False, f"ffmpeg failed: {error}"
    if result.returncode != 0 or not destination.is_file():
        return False, "ffmpeg could not convert the video."
    return True, "Browser-playable copy created."


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Traffic Video Analysis",
    "Line-Crossing Setup & Run",
    "Line-Crossing Results"
])


# ===== Tab 1: sampled-frame analysis ======================================
with tab1:
    video_path, video_name = video_source_picker("sampled")

    if video_path is not None:
        st.video(str(video_path))

        if st.button("Analyse Video", type="primary"):
            run_sampled_analysis(video_path, video_name)

    if "analysis_df" in st.session_state:
        render_sampled_results()
    else:
        st.info("Choose or upload a traffic video and click Analyse Video to begin.")


# ===== Tab 2: line selection + run ========================================
with tab2:
    st.header("Line-Crossing Setup & Run")
    st.write(
        "Choose a video, set the counting line(s) with the coordinate controls "
        "while watching the preview, then start the existing line_crossing.py. "
        "The dashboard passes your coordinates to the script through a JSON file "
        "(--load-lines), so the script's own OpenCV selection window is not used."
    )
    st.caption(
        "Streamlit cannot capture mouse clicks on an image without an extra package, "
        "so points are set with sliders. To click on a frame instead, run "
        "`python line_crossing.py --video videos\\video3.mp4 --select-only` in a terminal "
        "(the window shows the cursor's original-pixel position), then enter those numbers here."
    )

    setup_ok = True
    if not LINE_SCRIPT.is_file():
        st.error(f"line_crossing.py not found at {LINE_SCRIPT}")
        setup_ok = False
    if not MODEL_PATH.is_file():
        st.error(f"Model file not found at {MODEL_PATH}")
        setup_ok = False

    lc_path, lc_name = video_source_picker("linecross")

    if lc_path is not None:
        info = get_video_info(str(lc_path), lc_path.stat().st_mtime)

        if info is None:
            st.error("This video could not be opened or decoded.")
        else:
            width, height = info["width"], info["height"]
            total = info["total_frames"]
            st.caption(
                f"{lc_name}: {width} x {height} px, {info['fps']:.2f} FPS, "
                f"{total} frames. All coordinates below are original-frame pixels."
            )

            with st.expander("Preview video"):
                st.video(str(lc_path))

            skey = f"{lc_path.name}_{width}x{height}"

            frame_index = 0
            if total > 1:
                frame_index = st.slider(
                    "Preview frame", 0, total - 1, 0, key=f"lc_frame_{skey}"
                )

            n_lines = st.radio(
                "Number of counting lines",
                list(range(1, MAX_LINES + 1)),
                horizontal=True,
                key=f"lc_n_{skey}"
            )

            lines = []
            line_errors = []
            for k in range(n_lines):
                st.markdown(f"**Line L{k + 1}**")
                default_y = int(height * (0.5 + 0.08 * k))
                c1, c2, c3, c4 = st.columns(4)
                x1 = c1.slider("P1 x", 0, width - 1, int(width * 0.2), key=f"lc_{skey}_{k}_x1")
                y1 = c2.slider("P1 y", 0, height - 1, min(default_y, height - 1), key=f"lc_{skey}_{k}_y1")
                x2 = c3.slider("P2 x", 0, width - 1, int(width * 0.8), key=f"lc_{skey}_{k}_x2")
                y2 = c4.slider("P2 y", 0, height - 1, min(default_y, height - 1), key=f"lc_{skey}_{k}_y2")
                lines.append(((x1, y1), (x2, y2)))
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if length < MIN_LINE_LENGTH_PX:
                    line_errors.append(f"L{k + 1} is only {length:.0f}px long "
                                       f"(minimum {MIN_LINE_LENGTH_PX}px).")

            frame = read_frame_bgr(str(lc_path), lc_path.stat().st_mtime, frame_index)
            if frame is None:
                st.error("Could not read the preview frame.")
            else:
                preview = frame.copy()
                draw_preview_lines(preview, lines)
                st.image(
                    cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                    caption=(f"Frame {frame_index} at original resolution ({width}x{height}). "
                             "S1 = left of the P1->P2 direction, S2 = right. "
                             "Direction in the results refers to these sides."),
                    width="stretch"
                )

            for message in line_errors:
                st.error(message)

            o1, o2 = st.columns(2)
            max_frames = o1.number_input(
                "Frames to process (0 = whole video)",
                min_value=0, value=0, step=100, key=f"lc_max_{skey}",
                help="Use e.g. 900 for a quick test of the first 30 seconds."
            )
            imgsz = o2.selectbox(
                "YOLO input size", [960, 640], index=0, key=f"lc_imgsz_{skey}",
                help="960 matches the sampled-frame analysis; 640 is faster."
            )

            can_run = setup_ok and frame is not None and not line_errors
            st.warning(
                "Keep this page open while the analysis runs. Changing any control "
                "during processing cancels the run."
            )

            if st.button("Run line-crossing analysis", type="primary",
                         disabled=not can_run, key="lc_run"):
                progress = st.progress(0)
                status = st.empty()
                with st.spinner("Running line_crossing.py ..."):
                    outcome = run_line_crossing(
                        lc_path, lines, int(max_frames), int(imgsz), progress, status
                    )
                outcome["video"] = lc_name
                st.session_state["lc_last"] = outcome
                progress.empty()
                status.empty()

    last = st.session_state.get("lc_last")
    if last:
        if last["ok"]:
            st.success(f"{last['message']} (video: {last.get('video', '?')}). "
                       "See the Line-Crossing Results tab.")
        else:
            st.error(f"Line-crossing analysis failed: {last['message']}")
        if last["log"]:
            with st.expander("Console output from line_crossing.py"):
                st.code("\n".join(last["log"].splitlines()[-60:]) or "(no output)")


# ===== Tab 3: results ======================================================
with tab3:
    st.header("Vehicle Line-Crossing Analysis")
    st.write(
        "Newest completed run from outputs/line_crossing/ is shown first. "
        "Use the Setup & Run tab to create a new one."
    )

    run_dirs = list_run_dirs()

    if not LINE_OUTPUT_DIR.is_dir():
        st.warning(
            f"No line-crossing output folder found at {LINE_OUTPUT_DIR}. "
            "Run a line-crossing analysis first."
        )
    elif not run_dirs:
        st.info("No completed line-crossing runs found in outputs/line_crossing/.")
    else:
        selected_run = st.selectbox(
            "Select a line-crossing run (newest first)",
            run_dirs,
            format_func=lambda p: p.name,
            key="results_run"
        )

        events_path = selected_run / "vehicle_crossings.csv"
        summary_path = selected_run / "vehicle_count_summary.csv"
        video_file = selected_run / "line_crossing_result.mp4"
        browser_copy = selected_run / "line_crossing_result_browser.mp4"
        info_path = selected_run / "run_info.json"
        preview_path = selected_run / "line_preview.png"

        st.caption(
            "Files in this run: "
            + (", ".join(sorted(p.name for p in selected_run.iterdir() if p.is_file())) or "none")
        )

        # ---- run information ----
        if info_path.is_file():
            try:
                run_info = json.loads(info_path.read_text(encoding="utf-8"))
                if run_info.get("interrupted"):
                    st.warning("This run was interrupted - results are partial.")
                with st.expander("Run information"):
                    st.json(run_info)
            except (OSError, json.JSONDecodeError):
                st.caption("Run information could not be read.")
        else:
            st.caption("run_info.json not found for this run.")

        if preview_path.is_file():
            with st.expander("Line placement (line_preview.png)"):
                st.image(str(preview_path), width="stretch")

        # ---- events ----
        st.subheader("Crossing Events")
        events_df, events_error = safe_read_csv(events_path)

        if events_error:
            st.error(f"Could not read vehicle_crossings.csv: {events_error}")
        elif events_df.empty:
            st.info("No crossing events were recorded in this run.")
        else:
            st.metric("Recorded Crossing Events", len(events_df))

            if "line_id" in events_df.columns:
                per_line = events_df["line_id"].value_counts().sort_index()
                cols = st.columns(max(1, len(per_line)))
                for col, (line_name, count) in zip(cols, per_line.items()):
                    col.metric(f"Crossings on {line_name}", int(count))

            if "direction" in events_df.columns:
                st.subheader("Crossings by Direction")
                st.bar_chart(events_df["direction"].fillna("Unknown").value_counts())

            type_col = next(
                (c for c in events_df.columns
                 if c.lower() in {"vehicle_type", "class", "vehicle_class", "class_name"}),
                None
            )
            if type_col:
                st.subheader("Crossings by Vehicle Type")
                st.bar_chart(events_df[type_col].fillna("Unknown").value_counts())

            st.dataframe(events_df, width="stretch")

        if events_error is None:
            st.download_button(
                "Download Crossing Events CSV",
                data=events_df.to_csv(index=False).encode("utf-8"),
                file_name="vehicle_crossings.csv",
                mime="text/csv",
                key="download_crossings"
            )

        # ---- summary ----
        st.subheader("Vehicle Count Summary")
        if summary_path.is_file():
            summary_df, summary_error = safe_read_csv(summary_path)
            if summary_error:
                st.error(f"Could not read vehicle_count_summary.csv: {summary_error}")
            elif summary_df.empty:
                st.info("The summary file is empty.")
            else:
                st.dataframe(summary_df, width="stretch")
                st.download_button(
                    "Download Count Summary CSV",
                    data=summary_df.to_csv(index=False).encode("utf-8"),
                    file_name="vehicle_count_summary.csv",
                    mime="text/csv",
                    key="download_summary"
                )
        else:
            st.warning("vehicle_count_summary.csv was not found in this run folder.")

        # ---- annotated video ----
        st.subheader("Annotated Line-Crossing Video")
        if not video_file.is_file():
            st.warning("The annotated video (line_crossing_result.mp4) was not found in this run folder.")
        else:
            playable = browser_copy if browser_copy.is_file() else video_file
            try:
                st.video(str(playable))
            except Exception as error:
                st.warning(f"The video could not be displayed: {error}")

            st.caption(
                "line_crossing.py writes the video with the mp4v codec, which many "
                "browsers cannot play. If the player above stays blank, open the file "
                f"directly (VLC / Windows media player): {video_file}"
            )

            if not browser_copy.is_file():
                if st.button("Create browser-playable copy (needs ffmpeg)", key="convert_video"):
                    with st.spinner("Converting with ffmpeg..."):
                        ok, message = convert_for_browser(video_file, browser_copy)
                    (st.success if ok else st.warning)(message)
                    if ok:
                        st.rerun()

            size_mb = video_file.stat().st_size / (1024 * 1024)
            if size_mb <= 200:
                st.download_button(
                    "Download annotated video",
                    data=video_file.read_bytes(),
                    file_name=video_file.name,
                    mime="video/mp4",
                    key="download_video"
                )

        st.caption(
            "Crossing events are model estimates, not guaranteed unique "
            "vehicles. Check for missed detections and duplicate counts "
            "before reporting results."
        )