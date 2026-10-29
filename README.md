# Smart City Digital Twin — AI-Powered Traffic Monitoring

An AI-powered traffic monitoring prototype built using Python, YOLO, OpenCV, and Streamlit to analyze road traffic, detect vehicles, and visualize traffic activity for smart city infrastructure monitoring.

## Overview

This project demonstrates a video-based traffic monitoring system that processes recorded road footage and presents traffic analysis through an interactive dashboard. It combines computer vision, vehicle tracking, line-crossing analysis, and data visualization to support traffic monitoring in a smart city environment.

The current implementation is a prototype based on recorded video rather than a fully connected, real-time digital twin.

## Features

- **Vehicle Detection:** Detects vehicles in traffic video using YOLO.
- **Vehicle Tracking:** Uses object tracking to follow detected vehicles across video frames.
- **Line-Crossing Analysis:** Counts vehicle crossing events using a configurable counting line.
- **Traffic Analysis:** Generates traffic count data and provisional traffic-level classifications.
- **Interactive Dashboard:** Displays video analysis, metrics, charts, line configuration, and analysis results using Streamlit.
- **Data Export:** Stores analysis results in CSV format for further inspection.

## Tech Stack

- Python
- YOLO / Ultralytics
- OpenCV
- Streamlit
- Pandas
- NumPy
- CSV-based data analysis

## Project Structure

```text
SmartCity-DigitalTwin/
├── dashboard.py
├── detect_test.py
├── tracking_test.py
├── line_crossing.py
├── traffic_analysis.py
├── check_videos.py
├── view_frame.py
├── backup.py
├── requirements.txt
├── .gitignore
├── videos/
└── outputs/
```

*Note: Video files, model weights, and generated MP4 results are not included in this repository. The `requirements.txt` file should be created if it is not already present.*

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Ha12ri/SmartCity-DigitalTwin.git
cd SmartCity-DigitalTwin
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install the required packages used by the project:

```bash
pip install ultralytics opencv-python streamlit pandas numpy
```

Additional packages may be required depending on the scripts and dashboard configuration.

### 4. Prepare Input Files

Place supported traffic video files in the `videos/` directory and ensure the required YOLO model weights are available locally.

Large video files and model weights are intentionally excluded from this repository.

### 5. Run the Dashboard

```bash
python -m streamlit run dashboard.py
```

Open the local URL displayed in the terminal to access the dashboard.

## Analysis Workflow

1. Load a recorded traffic video.
2. Detect vehicles using the YOLO model.
3. Track vehicles across frames where supported.
4. Configure a counting line and analyze crossing events.
5. Review vehicle counts, traffic summaries, charts, and exported CSV files through the dashboard.

## Output

Depending on the selected workflow, the application can generate:

- Vehicle detection previews
- Annotated traffic videos
- Vehicle crossing event records
- Vehicle count summaries
- Traffic count and traffic-level analysis CSV files

Generated outputs are stored locally and are not automatically included in the GitHub repository.

## Limitations

- The current prototype analyzes recorded video; live CCTV and IoT sensor integration are not implemented.
- Vehicle counts may contain duplicate detections or tracking errors.
- Traffic-level thresholds are provisional and require further validation.
- Unusual traffic volume does not confirm an accident or a traffic jam.
- Performance depends on video quality, camera angle, model accuracy, and hardware.
- The system does not currently control traffic signals or provide verified real-world vehicle speeds.

## Future Improvements

- Live camera and sensor integration
- Improved vehicle counting and duplicate-event handling
- Validation against manually counted traffic data
- Historical traffic trend analysis
- Enhanced dashboard visualizations and alerting
- Integration with additional smart city infrastructure data

## Project Purpose

This academic project explores how computer vision and data visualization can support traffic monitoring as part of a smart city infrastructure system.

## Author

**Ha12ri**

GitHub: https://github.com/Ha12ri

---

*Developed as an academic prototype for Smart City Infrastructure Monitoring.*
