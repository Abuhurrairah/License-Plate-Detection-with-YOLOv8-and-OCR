import os
import glob
import traceback
from typing import List, Tuple, Dict, Any

import streamlit as st
import pandas as pd

# -------------------------
# Page setup
# -------------------------
st.set_page_config(
    page_title="LPR + SIS — Streamlit GUI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚗 License Plate Recognition (LPR) + State Identification (SIS)")
st.caption("Upload images or use your input_images/ folder. The app will detect vehicles, find plates, OCR the text, and map to Malaysian states.")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def _save_uploaded_images(files: List[Any], dest_dir: str) -> List[str]:
    os.makedirs(dest_dir, exist_ok=True)
    saved = []
    for f in files:
        name = os.path.basename(getattr(f, "name", "upload.jpg"))
        root, ext = os.path.splitext(name)
        if not ext:
            ext = ".jpg"
        out_path = os.path.join(dest_dir, name)
        i = 1
        while os.path.exists(out_path):
            out_path = os.path.join(dest_dir, f"{root}_{i}{ext}")
            i += 1
        bytes_data = f.read()
        with open(out_path, "wb") as w:
            w.write(bytes_data)
        saved.append(out_path)
    return saved

def _lazy_import_detector():
    """
    Import lazily so the app can render even if models aren't ready.
    """
    try:
        from YOLO_DETECTION import detect_vehicle_and_plate  # your pipeline
        return detect_vehicle_and_plate, None
    except Exception as e:
        return None, e

def _run_detection_on_paths(paths: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    detect_vehicle_and_plate, err = _lazy_import_detector()
    all_results, errors = [], []

    if detect_vehicle_and_plate is None:
        errors.append(f"Failed to import detection pipeline: {err}")
        return [], errors

    for p in paths:
        try:
            res = detect_vehicle_and_plate(p)
            if isinstance(res, list):
                all_results.extend(res)
            else:
                errors.append(f"{os.path.basename(p)} → Unexpected return type (expected list, got {type(res)})")
        except Exception as e:
            tb = traceback.format_exc(limit=1)
            errors.append(f"{os.path.basename(p)} → Error during detection: {e}\n{tb}")
    return all_results, errors

def _collect_plate_crops(base_name: str, dir_name: str = "plates_detected") -> List[str]:
    """
    Finds crops saved by your detector:
    '{base}{CLS}{i}plate{j}.jpg' OR '{base}unknown_plate{j}.jpg'.
    """
    if not os.path.isdir(dir_name):
        return []
    patt1 = os.path.join(dir_name, f"{base_name}*_plate*.jpg")
    patt2 = os.path.join(dir_name, f"{base_name}unknown_plate*.jpg")
    return sorted(glob.glob(patt1) + glob.glob(patt2))

def _build_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["file", "vehicle", "vehicle_id", "plates", "states"])
    df = pd.DataFrame(results)
    if "plates" in df.columns:
        df["plates"] = df["plates"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
    if "states" in df.columns:
        df["states"] = df["states"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
    return df

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Controls")
    mode = st.radio(
        label="Input source",
        options=["Upload images", "Use folder: input_images/"],
        index=0,
    )
    st.divider()
    st.markdown("*Tips*")
    st.markdown("- Run from your project root so relative folders are created under src/ or the CWD.")
    st.markdown("- Ensure YOLO_DETECTION.py can find your model files (vehicle + plate).")
    st.markdown("- Plate crops will be saved under plates_detected/.")

# -------------------------
# Tabs
# -------------------------
tab1, tab2 = st.tabs(["🔎 Detection", "ℹ About"])

with tab2:
    st.subheader("About this app")
    st.markdown(
        """
        This Streamlit GUI wraps your existing pipeline to satisfy the *GUI requirement*.

        *Pipeline*
        - YOLO vehicle detection (COCO classes).
        - License plate detection on the vehicle regions (custom weights).
        - OCR on plate crops.
        - Mapping of plate prefix → Malaysian state (SIS).

        *Run locally*
        bash
        streamlit run src/streamlit_gui.py
        
        Put test images in input_images/ (optional). Plate crops appear in plates_detected/.
        """
    )

with tab1:
    all_results = []
    errors = []

    if mode == "Upload images":
        uploads = st.file_uploader(
            "Upload .jpg/.jpeg/.png files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        run_btn = st.button("Run detection on uploaded image(s)")

        if run_btn:
            if not uploads:
                st.warning("Please upload at least one image.")
            else:
                with st.spinner("Saving uploads..."):
                    saved_paths = _save_uploaded_images(uploads, dest_dir="uploaded_images")
                st.success(f"Saved {len(saved_paths)} image(s). Running detection...")
                with st.spinner("Detecting vehicles and plates..."):
                    all_results, errors = _run_detection_on_paths(saved_paths)

    else:
        run_batch = st.button("Run detection on folder: input_images/")
        if run_batch:
            if not os.path.isdir("input_images"):
                st.error("Folder input_images/ not found. Please create it and add images.")
            else:
                paths = [os.path.join("input_images", f) for f in os.listdir("input_images")
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if not paths:
                    st.warning("No images found in input_images/.")
                else:
                    with st.spinner(f"Running detection on {len(paths)} image(s)..."):
                        all_results, errors = _run_detection_on_paths(paths)

    # Errors
    if errors:
        st.warning("Some images could not be processed:")
        for e in errors[:50]:
            st.code(e)

    # Results
    if all_results:
        st.success(f"Done! Processed {len(set([r['file'] for r in all_results]))} file(s).")

        df = _build_dataframe(all_results)
        st.subheader("Results (per detected vehicle)")
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("State frequency summary"):
            def _split_states(x):
                if isinstance(x, str):
                    return [s.strip() for s in x.split(",") if s.strip()]
                return x if isinstance(x, list) else []
            states_series = df["states"].apply(_split_states).explode().value_counts(dropna=False)
            agg_df = pd.DataFrame({"count": states_series})
            st.dataframe(agg_df, use_container_width=True)

        st.subheader("Per-image details")
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for r in all_results:
            by_file.setdefault(r["file"], []).append(r)

        for base, rows in by_file.items():
            with st.container(border=True):
                st.markdown(f"*File:* {base}")

                # Show original if present in uploaded_images or input_images
                candidate_paths = [
                    glob.glob(os.path.join("uploaded_images", f"{base}.")),
                    glob.glob(os.path.join("input_images", f"{base}."))
                ]
                if candidate_paths:
                    try:
                        st.image(candidate_paths[0], caption=f"Original — {os.path.basename(candidate_paths[0])}", use_container_width=True)
                    except Exception:
                        pass

                for row in rows:
                    veh = row.get("vehicle", "UNKNOWN")
                    vid = row.get("vehicle_id", "")
                    plates = row.get("plates", [])
                    states = row.get("states", [])
                    st.write(f"- *{veh} {vid}* → Plates: {plates} → States: {states}")

                crops = _collect_plate_crops(base)
                if crops:
                    st.markdown("*Detected plate crops:*")
                    st.image(crops, use_column_width=True)
                else:
                    st.caption("No plate crops saved for this image (or none detected).")
    else:
        st.info("Upload images or run the batch mode from the sidebar to begin.")