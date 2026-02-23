# src/YOLO_DETECTION.py
from ultralytics import YOLO
import cv2
import os
import re
from plate_ocr import run_ocr, map_to_state   # ✅ include state mapper

# Load models once
vehicle_model = YOLO("yolov8m.pt")   # COCO vehicle detector
plate_model = YOLO("../model/number_plate_detection.pt")   # adjust path if needed


def clean_plate_text(text: str) -> str:
    """
    Normalize OCR text: uppercase, strip, keep only A-Z0-9.
    Returns "NO PLATE" if nothing valid remains.
    """
    if not text:
        return "NO PLATE"
    text = text.strip().upper()
    text = re.sub(r"[^A-Z0-9]", "", text)   # remove non-alphanum
    return text if text else "NO PLATE"


def postprocess_plate_list(plate_texts: list) -> list:
    """
    Deduplicate plate list and remove 'NO PLATE' if real plates exist.
    """
    seen, cleaned = set(), []
    for p in plate_texts:
        if p not in seen:
            seen.add(p)
            cleaned.append(p)

    if len(cleaned) > 1 and "NO PLATE" in cleaned:
        cleaned = [p for p in cleaned if p != "NO PLATE"]

    return cleaned or ["NO PLATE"]


def map_states(plate_texts: list) -> list:
    """
    Convert plate texts into states.
    Always return 'Unknown' if plate == 'NO PLATE'.
    """
    states = []
    for p in plate_texts:
        if p == "NO PLATE":
            states.append("Unknown")
        else:
            states.append(map_to_state(p))
    return states or ["Unknown"]


def detect_vehicle_and_plate(image_path, save_dir="plates_detected"):
    os.makedirs(save_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Could not read image: {image_path}")
        return []

    base = os.path.splitext(os.path.basename(image_path))[0]
    results = []

    vehicle_results = vehicle_model(image, conf=0.5)

    # === Case 1: No vehicles detected → scan full image for plates
    if len(vehicle_results[0].boxes) == 0:
        print(f"[{base}] ❌ No vehicle detected → checking full image for plates...")
        vehicle_crop = image.copy()
        plate_results = plate_model(vehicle_crop, conf=0.5)

        plate_texts = []
        for j, plate_box in enumerate(plate_results[0].boxes, start=1):
            px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            save_path = os.path.join(save_dir, f"{base}_unknown_plate_{j}.jpg")
            cv2.imwrite(save_path, plate_crop)

            raw_text = run_ocr(save_path)
            cleaned = clean_plate_text(raw_text)
            plate_texts.append(cleaned)

        plate_texts = postprocess_plate_list(plate_texts)
        states = map_states(plate_texts)

        results.append({
            "file": base,
            "vehicle": "UNKNOWN",
            "vehicle_id": None,
            "plates": plate_texts,
            "states": states
        })
        print(f"[{base}] 🚘 UNKNOWN VEHICLE → Plates: {plate_texts} → States: {states}")
        return results

    # === Case 2: Vehicles detected
    for i, box in enumerate(vehicle_results[0].boxes, start=1):
        cls_id = int(box.cls[0])
        cls_name = vehicle_model.names[cls_id].upper()
        if cls_name not in ["CAR", "BUS", "TRUCK", "MOTORCYCLE", "BICYCLE", "VAN"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        vehicle_crop = image[y1:y2, x1:x2]

        plate_results = plate_model(vehicle_crop, conf=0.5)
        plate_texts = []

        for j, plate_box in enumerate(plate_results[0].boxes, start=1):
            px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            save_path = os.path.join(save_dir, f"{base}_{cls_name}_{i}_plate_{j}.jpg")
            cv2.imwrite(save_path, plate_crop)

            raw_text = run_ocr(save_path)
            cleaned = clean_plate_text(raw_text)
            plate_texts.append(cleaned)

        plate_texts = postprocess_plate_list(plate_texts)
        states = map_states(plate_texts)

        results.append({
            "file": base,
            "vehicle": cls_name,
            "vehicle_id": i,
            "plates": plate_texts,
            "states": states
        })

        if plate_texts and plate_texts != ["NO PLATE"]:
            print(f"[{base}] 🚘 {cls_name} {i} → Plates: {plate_texts} → States: {states}")
        else:
            print(f"[{base}] ⚠️ {cls_name} {i} → No plates found")

    return results
