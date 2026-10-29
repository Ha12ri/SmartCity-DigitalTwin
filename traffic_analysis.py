
import csv
from pathlib import Path
from collections import Counter

input_path = Path("outputs/traffic_counts.csv")
output_path = Path("outputs/traffic_analysis.csv")

if not input_path.exists():
    print("Traffic counts CSV not found")
    raise SystemExit(1)

with open(input_path, newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

if not rows:
    print("No traffic data found")
    raise SystemExit(1)

counts = [int(row["total_vehicles_detected"]) for row in rows]

average_count = sum(counts) / len(counts)
maximum_count = max(counts)
minimum_count = min(counts)

threshold = 10
levels = Counter()

for row in rows:
    count = int(row["total_vehicles_detected"])

    if count >= threshold:
        level = "High detected vehicle presence"
    elif count >= 6:
        level = "Moderate detected vehicle presence"
    else:
        level = "Low detected vehicle presence"

    row["traffic_level"] = level
    levels[level] += 1

with open(output_path, "w", newline="", encoding="utf-8") as file:
    fields = list(rows[0].keys())
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("Traffic analysis completed")
print("Samples analysed:", len(rows))
print(f"Average detected vehicles per sample: {average_count:.2f}")
print("Maximum detected vehicles in a sample:", maximum_count)
print("Minimum detected vehicles in a sample:", minimum_count)
print("Traffic level sample counts:", dict(levels))
print("Saved:", output_path)
print("Note: Levels use provisional thresholds, not validated congestion measurements.")
