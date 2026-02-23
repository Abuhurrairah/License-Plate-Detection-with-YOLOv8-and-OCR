# License-Plate-Detection-with-YOLOv8-and-OCR
System that detects number plates and tells which state they are from in Malaysia  
## Structure
- `src/`: Python scripts including YOLO-based detection
- `matlab/`: MATLAB scripts for segmentation and OCR
- `model/`: Pretrained YOLOv8 model
- `input_images/`: Sample input images
- `plates_detected/`: Output cropped plates

## Requirements
```bash
pip install -r requirements.txt
```

## Run
```bash
python src/YOLO_DETECTION.py
```
