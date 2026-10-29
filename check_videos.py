from pathlib import Path

video_folder = Path(__file__).resolve().parent / "videos"

print("Searching folder:", video_folder)
print("Folder exists:", video_folder.exists())

if video_folder.exists():
    for file in video_folder.iterdir():
        print(file.name)

    videos = [
        p for p in video_folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    ]

    print("\nVideos found:", len(videos))

    for video in videos:
        print(video)
else:
    print("Videos folder does not exist at this location.")