import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("Two Stage Tablet Inspection System")

# ==========================================
# SIDEBAR
# ==========================================

stage1_model_name = st.sidebar.selectbox(
    "Stage 1 Model",
    [
        "models/stage1/new_m_1.pt",
        "models/stage1/stage1_tablet.pt"
    ]
)

stage2_model_name = st.sidebar.selectbox(
    "Stage 2 Model",
    [
        "models/stage2/new_m_2.pt",
        "models/stage2/stage2_defect.pt",
        "models/stage2/best.pt"
    ]
)

CONF1 = st.sidebar.slider(
    "Stage 1 Confidence",
    0.0, 1.0, 0.5, 0.05
)

CONF2 = st.sidebar.slider(
    "Stage 2 Confidence",
    0.0, 1.0, 0.2, 0.05
)

IMGSZ = st.sidebar.selectbox(
    "Inference Size",
    [320, 416, 640],
    index=0
)

DEVICE = st.sidebar.selectbox(
    "Device",
    ["cpu", "cuda"]
)

# ==========================================
# CAMERA SETTINGS
# ==========================================

camera_source = st.sidebar.radio(
    "Camera Source",
    ["Local Camera", "IP Webcam"]
)

if camera_source == "Local Camera":

    camera_index = st.sidebar.selectbox(
        "Camera Index",
        [0, 1, 2, 3],
        index=0
    )

    ip_url = None

else:

    ip_url = st.sidebar.text_input(
        "IP Webcam URL",
        "http://192.168.1.9:8080/video"
    )

    camera_index = None

# ==========================================
# MODEL LOADING
# ==========================================

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

stage1 = load_model(stage1_model_name)
stage2 = load_model(stage2_model_name)

# ==========================================
# PIPELINE
# ==========================================

def run_pipeline(frame):

    results1 = stage1(
        frame,
        imgsz=IMGSZ,
        conf=CONF1,
        device=DEVICE,
        verbose=False
    )

    tablets = []

    for r in results1:

        if r.boxes is None:
            continue

        for box, cls, conf in zip(
            r.boxes.xyxy,
            r.boxes.cls,
            r.boxes.conf
        ):

            x1, y1, x2, y2 = map(int, box.tolist())

            label = (
                f"{stage1.names[int(cls)]} "
                f"{conf.item()*100:.1f}%"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            if int(cls) in [0, 3]:
                tablets.append((x1, y1, x2, y2))

    # =====================================
    # STAGE 2
    # =====================================

    for (x1, y1, x2, y2) in tablets:

        margin = 0.05

        w = x2 - x1
        h = y2 - y1

        x1e = max(0, int(x1 - margin * w))
        y1e = max(0, int(y1 - margin * h))
        x2e = min(frame.shape[1], int(x2 + margin * w))
        y2e = min(frame.shape[0], int(y2 + margin * h))

        crop = frame[y1e:y2e, x1e:x2e]

        if crop.size == 0:
            continue

        results2 = stage2(
            crop,
            imgsz=IMGSZ,
            conf=CONF2,
            device=DEVICE,
            verbose=False
        )

        for r2 in results2:

            if r2.boxes is None:
                continue

            if r2.masks is None:
                continue

            for mask_tensor, cls_tensor, conf_tensor in zip(
                r2.masks.data,
                r2.boxes.cls,
                r2.boxes.conf
            ):

                mask = mask_tensor.cpu().numpy()

                mask = cv2.resize(
                    mask,
                    (crop.shape[1], crop.shape[0])
                )

                full_mask = np.zeros(
                    frame.shape[:2],
                    dtype=np.uint8
                )

                full_mask[
                    y1e:y2e,
                    x1e:x2e
                ] = (mask > 0.5).astype(np.uint8)

                cls_id = int(cls_tensor)

                defect_name = stage2.names[cls_id]

                conf_pct = conf_tensor.item() * 100

                if defect_name.lower() == "chip":
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                overlay = frame.copy()

                overlay[full_mask > 0] = color

                frame = cv2.addWeighted(
                    overlay,
                    0.4,
                    frame,
                    0.6,
                    0
                )

                ys, xs = np.where(full_mask > 0)

                if len(xs) > 0:

                    cx = int(xs.mean())
                    cy = int(ys.mean())

                    cv2.putText(
                        frame,
                        f"{defect_name} {conf_pct:.1f}%",
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

    return frame

# ==========================================
# INPUT SOURCE
# ==========================================

mode = st.radio(
    "Input Source",
    ["Live Camera", "Upload Video"]
)

frame_placeholder = st.empty()

# ==========================================
# LIVE CAMERA
# ==========================================

if mode == "Live Camera":

    start = st.button("Start Camera")

    if start:

        if camera_source == "Local Camera":

            cap = cv2.VideoCapture(camera_index)

        else:

            cap = cv2.VideoCapture(ip_url)

            cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                st.error(
                    "Unable to read camera stream."
                )
                break

            frame = run_pipeline(frame)

            frame_placeholder.image(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

        cap.release()

# ==========================================
# VIDEO UPLOAD
# ==========================================

else:

    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file:

        temp_file = tempfile.NamedTemporaryFile(
            delete=False
        )

        temp_file.write(
            uploaded_file.read()
        )

        cap = cv2.VideoCapture(
            temp_file.name
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        progress = st.progress(0)

        current_frame = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            current_frame += 1

            frame = run_pipeline(frame)

            frame_placeholder.image(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            if total_frames > 0:

                progress.progress(
                    min(
                        current_frame /
                        total_frames,
                        1.0
                    )
                )

        cap.release()

        st.success(
            "Video processing completed."
        )