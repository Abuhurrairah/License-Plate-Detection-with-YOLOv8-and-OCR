# src/plate_ocr.py
import cv2
import easyocr

reader = easyocr.Reader(["en"])

state_map = {
    "W": "Kuala Lumpur",
    "A": "Perak",
    "B": "Selangor",
    "C": "Pahang",
    "D": "Kelantan",
    "F": "Putrajaya",
    "J": "Johor",
    "K": "Kedah",
    "M": "Melaka",
    "N": "Negeri Sembilan",
    "P": "Pulau Pinang",
    "Q": "Sarawak",
    "R": "Perlis",
    "T": "Terengganu",
    "V": "Labuan",   # keep as you used earlier
    "S": "Sabah",
}

def map_to_state(plate_text: str) -> str:
    plate_text = plate_text.upper().strip()
    if not plate_text:
        return "Unknown"
    if len(plate_text) >= 2 and plate_text[:2] in state_map:
        return state_map[plate_text[:2]]
    return state_map.get(plate_text[0], "Unknown")

def run_ocr(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None or img.size == 0:
        print(f"⚠️ OCR skipped: could not load {image_path}")
        return ""
    results = reader.readtext(img, detail=0, paragraph=False)
    return results[0] if results else ""
