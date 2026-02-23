# src/main.py
import os
from YOLO_DETECTION import detect_vehicle_and_plate

input_dir = "input_images"
all_results = []

# Process each image
for file in os.listdir(input_dir):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(input_dir, file)
        print(f"\nProcessing {path}...")
        results = detect_vehicle_and_plate(path)
        all_results.extend(results)

print("\n=== SUMMARY ===\n")

grouped_results = {}
for entry in all_results:
    file_tag = f"[{entry['file']}]"
    vehicle_id = entry['vehicle_id'] or ''
    plates = entry['plates']
    states = entry.get('states', [])   # safely fetch states list

    detail = f"🚘 {entry['vehicle']} {vehicle_id} → Plates: {plates} → States: {states}"

    if file_tag not in grouped_results:
        grouped_results[file_tag] = []
    grouped_results[file_tag].append(detail)

# Print nicely
for file_tag, details in grouped_results.items():
    print(file_tag)
    for d in details:
        print("  " + d)
    print()
