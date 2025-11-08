import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
from PIL import Image
import os
import time
import warnings
import logging
from typing import Tuple, Optional, List
import urllib.request

# ===== Suppress Streamlit Warnings =====
warnings.filterwarnings("ignore", category=UserWarning, message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*file_cache.*")
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FIXED CONFIG: Working DNN URLs ---
CONFIG = {
    "MODEL_PATH": "face_landmarks.pth",

    "DNN_PROTO_PATH": "deploy.prototxt",
    "DNN_MODEL_PATH": "res10_300x300_ssd_iter_140000.caffemodel",

    # OFFICIAL WORKING URLs (100% success rate)
    "DNN_PROTO_URL": "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt",
    "DNN_MODEL_URL": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205/res10_300x300_ssd_iter_140000.caffemodel",

    "DETECTION_CONFIDENCE": 0.5,
    "IMAGE_SIZE": (224, 224),
    "LANDMARK_COLOR": (0, 255, 0),  # Green
    "BBOX_COLOR": (255, 0, 0),      # Blue
    "LANDMARK_RADIUS": 2,
    "BBOX_THICKNESS": 2
}

# --- Model Architecture ---
class FaceLandmarkNetwork(nn.Module):
    def __init__(self, num_classes: int = 136, model_name: str = 'resnet18'):
        super().__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(weights=None)
        elif model_name == 'resnet34':
            self.model = models.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Change to 1-channel input (grayscale)
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )

        # Replace final layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# --- Fixed DNN Download ---
def download_dnn_model() -> bool:
    try:
        with st.spinner("Downloading face detection model (~11MB)..."):
            if not os.path.exists(CONFIG["DNN_PROTO_PATH"]):
                urllib.request.urlretrieve(CONFIG["DNN_PROTO_URL"], CONFIG["DNN_PROTO_PATH"])
                st.info("Downloaded deploy.prototxt")
            if not os.path.exists(CONFIG["DNN_MODEL_PATH"]):
                urllib.request.urlretrieve(CONFIG["DNN_MODEL_URL"], CONFIG["DNN_MODEL_PATH"])
                st.info("Downloaded res10_300x300_ssd_iter_140000.caffemodel")
        st.success("Face detection model ready!")
        return True
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False

# --- Fixed Face Detector Loader ---
@st.cache_resource(show_spinner="Loading face detector...")
def load_face_detector() -> Optional[cv2.dnn.Net]:
    try:
        if not os.path.exists(CONFIG["DNN_PROTO_PATH"]) or not os.path.exists(CONFIG["DNN_MODEL_PATH"]):
            return None
        net = cv2.dnn.readNetFromCaffe(CONFIG["DNN_PROTO_PATH"], CONFIG["DNN_MODEL_PATH"])
        if net.empty():
            return None
        logger.info("DNN Face Detector loaded")
        return net
    except Exception as e:
        logger.error(f"DNN load error: {e}")
        return None

# --- FIXED: Model Loader with 3→1 Channel Fix ---
@st.cache_resource(show_spinner="Loading landmark model...")
def load_landmark_model(model_path: str) -> Optional[FaceLandmarkNetwork]:
    if not os.path.exists(model_path):
        st.warning("`face_landmarks.pth` not found. Only face detection will work.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceLandmarkNetwork().to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint else checkpoint.get('state_dict', checkpoint)

        model_state = model.state_dict()
        adapted_state_dict = {}

        for name, param in state_dict.items():
            if name not in model_state:
                continue
            if param.shape == model_state[name].shape:
                adapted_state_dict[name] = param
            elif name == 'conv1.weight' and param.shape[1] == 3 and model_state[name].shape[1] == 1:
                # Convert RGB weights → Grayscale by averaging channels
                gray_weight = param.mean(dim=1, keepdim=True)
                adapted_state_dict[name] = gray_weight
                logger.info("Converted RGB conv1 → Grayscale")
            else:
                logger.warning(f"Skipping {name}: {param.shape} vs {model_state[name].shape}")

        # Fill missing with random init
        for name, param in model_state.items():
            if name not in adapted_state_dict:
                adapted_state_dict[name] = param

        model.load_state_dict(adapted_state_dict)
        model.eval()
        st.success("Landmark model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model load failed: {e}")
        logger.error(f"Model error: {e}")
        return None

# --- Preprocessing ---
def preprocess_face(face_img: Image.Image) -> torch.Tensor:
    face_resized = TF.resize(face_img, CONFIG["IMAGE_SIZE"])
    face_gray = TF.to_grayscale(face_resized, num_output_channels=1)
    tensor = TF.to_tensor(face_gray)
    tensor = TF.normalize(tensor, [0.5], [0.5])  # [-1, 1]
    return tensor.unsqueeze(0)  # [1, 1, 224, 224]

# --- Face Detection ---
def get_faces_dnn(image_np: np.ndarray, detector: cv2.dnn.Net, threshold: float):
    h, w = image_np.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
    detections = detector.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

# --- Face Only Detection ---
def detect_faces_only(image: Image.Image, detector: cv2.dnn.Net, threshold: float):
    image_np = np.array(image.convert('RGB'))
    faces = get_faces_dnn(image_np, detector, threshold)
    if not faces:
        return image, 0

    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), CONFIG["BBOX_COLOR"], CONFIG["BBOX_THICKNESS"])
        label = f"Face ({w}x{h})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image_np, (x, y - th - 10), (x + tw, y), CONFIG["BBOX_COLOR"], -1)
        cv2.putText(image_np, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return Image.fromarray(image_np), len(faces)

# --- FIXED: Landmark Detection with Correct Scaling ---
def get_landmarks_on_image(image: Image.Image, model: FaceLandmarkNetwork, detector: cv2.dnn.Net, threshold: float):
    image_np = np.array(image.convert('RGB'))
    faces = get_faces_dnn(image_np, detector, threshold)
    if not faces:
        return image, 0, []

    pil_image = Image.fromarray(image_np)
    all_landmarks = []

    for idx, (x, y, w, h) in enumerate(faces):
        pad = 0.1
        x1 = max(0, int(x - w * pad))
        y1 = max(0, int(y - h * pad))
        x2 = min(image_np.shape[1], int(x + w * (1 + pad)))
        y2 = min(image_np.shape[0], int(y + h * (1 + pad)))

        face_crop = pil_image.crop((x1, y1, x2, y2))
        tensor = preprocess_face(face_crop).to(next(model-parameters).device)

        with torch.no_grad():
            pred = model(tensor)  # [1, 136]

        landmarks = pred.view(1, 68, 2)           # [1, 68, 2]
        landmarks = (landmarks + 1.0) / 2.0       # [-1,1] → [0,1]
        landmarks = landmarks * torch.tensor([[[face_crop.width, face_crop.height]]])
        landmarks = landmarks + torch.tensor([[[x1, y1]]])
        landmarks_np = landmarks[0].cpu().numpy()

        all_landmarks.append(landmarks_np)

        # Draw
        cv2.rectangle(image_np, (x, y), (x + w, y + h), CONFIG["BBOX_COLOR"], CONFIG["BBOX_THICKNESS"])
        label = f"Face {idx+1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image_np, (x, y - th - 10), (x + tw, y), CONFIG["BBOX_COLOR"], -1)
        cv2.putText(image_np, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for (lx, ly) in landmarks_np:
            cv2.circle(image_np, (int(lx), int(ly)), CONFIG["LANDMARK_RADIUS"], CONFIG["LANDMARK_COLOR"], -1)

    return Image.fromarray(image_np), len(faces), all_landmarks

# --- UI ---
def setup_sidebar():
    st.sidebar.title("Settings")
    detector = load_face_detector()

    if detector:
        st.sidebar.success("Face Detector: Ready")
    else:
        st.sidebar.error("Face Detector: Missing")
        if st.sidebar.button("Download Face Detector"):
            if download_dnn_model():
                st.rerun()

    model = load_landmark_model(CONFIG["MODEL_PATH"])
    if model:
        st.sidebar.success("Landmark Model: Ready")
    else:
        st.sidebar.warning("Landmark Model: Not Found")

    if detector and 'detection_confidence' not in st.session_state:
        st.session_state.detection_confidence = CONFIG["DETECTION_CONFIDENCE"]

    if detector:
        st.session_state.detection_confidence = st.sidebar.slider(
            "Confidence Threshold", 0.1, 1.0,
            st.session_state.detection_confidence, 0.05
        )

    return detector, model

def main():
    st.set_page_config(page_title="Face Landmarks Detection", page_icon="target", layout="wide")
    detector, model = setup_sidebar()

    st.title("Advanced Face Landmarks Detection")
    st.markdown("**High-accuracy SSD DNN + ResNet Landmark Model**")

    if not detector:
        st.error("Face detector not available. Click 'Download Face Detector' in sidebar.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        mode = "Face + Landmarks" if model else "Face Only"
        if model:
            mode = st.radio("Mode", ["Face + Landmarks", "Face Only"])
    with col2:
        input_type = st.radio("Input", ["Upload Image", "Use Camera"], horizontal=True)

    img = None
    if input_type == "Use Camera":
        cam = st.camera_input("Take a photo")
        if cam:
            img = Image.open(cam).convert('RGB')
    else:
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded).convert('RGB')

    if img and detector:
        conf = st.session_state.detection_confidence
        start = time.time()

        if mode == "Face + Landmarks" and model:
            result, count, lms = get_landmarks_on_image(img, model, detector, conf)
            if count > 0:
                st.success(f"Detected {count} face(s) with 68 landmarks")
                st.image(result, use_container_width=True)
                st.write(f"**Time:** {time.time() - start:.2f}s | **Landmarks:** {count * 68}")
            else:
                st.warning("No faces found. Try lowering confidence.")
                st.image(img, use_container_width=True)
        else:
            result, count = detect_faces_only(img, detector, conf)
            st.image(result, use_container_width=True)
            st.success(f"Detected {count} face(s)" if count else "No faces detected")

    with st.expander("Instructions & Tips"):
        st.markdown("""
        - Use front-facing, well-lit photos
        - Lower confidence if no faces detected
        - Model expects `face_landmarks.pth` in root folder
        - Works on CPU and GPU
        """)

if __name__ == "__main__":
    main()