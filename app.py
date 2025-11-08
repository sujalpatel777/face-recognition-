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

# ===== FIX: Suppress Streamlit Warnings =====
warnings.filterwarnings("ignore", category=UserWarning, message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*file_cache.*")
logging.getLogger("streamlit").setLevel(logging.ERROR)
# ============================================

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
CONFIG = {
    "MODEL_PATH": "face_landmarks.pth",
    "CASCADE_URL": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "DETECTION_SCALE_FACTOR": 1.1,
    "DETECTION_MIN_NEIGHBORS": 5,
    "DETECTION_MIN_SIZE": (30, 30),
    "IMAGE_SIZE": (224, 224),
    "LANDMARK_COLOR": (0, 255, 0),  # Green
    "BBOX_COLOR": (255, 0, 0),  # Blue
    "LANDMARK_RADIUS": 2,
    "BBOX_THICKNESS": 2
}

# --- 1. Enhanced Model Architecture ---
class FaceLandmarkNetwork(nn.Module):
    """Improved face landmark detection model with better error handling"""
    
    def __init__(self, num_classes: int = 136, model_name: str = 'resnet18'):
        super().__init__()
        self.model_name = model_name
        
        try:
            if model_name == 'resnet18':
                self.model = models.resnet18(weights=None)
            elif model_name == 'resnet34':
                self.model = models.resnet34(weights=None)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
                
            # Modify for grayscale input
            original_conv1 = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                1, 64, 
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias is not None
            )
            
            # Modify final layer for landmarks
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# --- 2. Enhanced Helper Functions ---

def download_cascade_file() -> bool:
    """Download cascade file with progress indicator"""
    try:
        import urllib.request
        with st.spinner("📥 Downloading face detection model..."):
            urllib.request.urlretrieve(CONFIG["CASCADE_URL"], "haarcascade_frontalface_default.xml")
        st.success("✅ Face detection model downloaded successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        return False

@st.cache_resource(show_spinner=False)
def load_face_cascade(cascade_path: str) -> Optional[cv2.CascadeClassifier]:
    """Loads Haar Cascade classifier with enhanced error handling"""
    try:
        if not os.path.exists(cascade_path):
            return None
            
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            logger.error("Cascade classifier loaded but empty")
            return None
            
        logger.info("Cascade classifier loaded successfully")
        return cascade
        
    except Exception as e:
        logger.error(f"Error loading cascade: {e}")
        return None

@st.cache_resource(show_spinner="Loading landmark model...")
def load_landmark_model(model_path: str) -> Optional[FaceLandmarkNetwork]:
    """Loads trained PyTorch model with comprehensive error handling"""
    try:
        if not os.path.exists(model_path):
            return None
            
        model = FaceLandmarkNetwork()
        
        # Load with device detection
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle both state_dict and full model saves
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Model loading error: {e}")
        return None

def find_cascade_file() -> Optional[str]:
    """Enhanced cascade file locator"""
    possible_paths = [
        "haarcascade_frontalface_default.xml",  # Local
        os.path.join("opencv", "data", "haarcascades", "haarcascade_frontalface_default.xml"),
        os.path.join("cv2", "data", "haarcascade_frontalface_default.xml"),
    ]
    
    # Try OpenCV's data directory
    try:
        opencv_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
        possible_paths.insert(0, opencv_path)
    except:
        pass
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found cascade at: {path}")
            return path
    
    logger.warning("Cascade file not found in any known location")
    return None

# --- 3. Enhanced Processing Functions ---

def preprocess_face(face_img: Image.Image) -> torch.Tensor:
    """Enhanced face preprocessing pipeline"""
    try:
        # Resize
        face_resized = TF.resize(face_img, CONFIG["IMAGE_SIZE"])
        
        # Convert to grayscale
        face_gray = TF.to_grayscale(face_resized, num_output_channels=1)
        
        # Convert to tensor and normalize
        face_tensor = TF.to_tensor(face_gray)
        face_tensor = TF.normalize(face_tensor, [0.5], [0.5])
        
        return face_tensor.unsqueeze(0)  # Add batch dimension
        
    except Exception as e:
        logger.error(f"Face preprocessing failed: {e}")
        raise

def detect_faces_only(image: Image.Image, cascade: cv2.CascadeClassifier) -> Tuple[Image.Image, int]:
    """Enhanced face detection without landmarks"""
    try:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=CONFIG["DETECTION_SCALE_FACTOR"],
            minNeighbors=CONFIG["DETECTION_MIN_NEIGHBORS"],
            minSize=CONFIG["DETECTION_MIN_SIZE"]
        )

        if len(faces) == 0:
            return image, 0

        # Draw enhanced bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(image_np, (x, y), (x + w, y + h), 
                         CONFIG["BBOX_COLOR"], CONFIG["BBOX_THICKNESS"])
            
            # Enhanced label with background
            label = f"Face ({w}x{h})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(image_np, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), CONFIG["BBOX_COLOR"], -1)
            
            # Text
            cv2.putText(image_np, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return Image.fromarray(image_np), len(faces)
        
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return image, 0

def get_landmarks_on_image(image: Image.Image, model: FaceLandmarkNetwork, 
                          cascade: cv2.CascadeClassifier) -> Tuple[Image.Image, int, List[np.ndarray]]:
    """Enhanced landmark detection with multiple faces support"""
    try:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=CONFIG["DETECTION_SCALE_FACTOR"],
            minNeighbors=CONFIG["DETECTION_MIN_NEIGHBORS"],
            minSize=CONFIG["DETECTION_MIN_SIZE"]
        )

        if len(faces) == 0:
            return image, 0, []

        all_landmarks = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # Crop and preprocess face
            face_img = image.crop((x, y, x + w, y + h))
            face_tensor = preprocess_face(face_img)
            
            # Predict landmarks
            with torch.no_grad():
                predictions = model(face_tensor)
            
            # Process predictions
            landmarks = predictions.view(-1, 68, 2)
            landmarks = (landmarks + 0.5) * CONFIG["IMAGE_SIZE"][0]
            landmarks_np = landmarks.cpu().numpy().squeeze()
            
            # Scale to original image coordinates
            scale_x, scale_y = w / CONFIG["IMAGE_SIZE"][0], h / CONFIG["IMAGE_SIZE"][1]
            landmarks_scaled = landmarks_np * np.array([scale_x, scale_y])
            landmarks_scaled = landmarks_scaled + np.array([x, y])
            
            all_landmarks.append(landmarks_scaled)
            
            # Draw bounding box
            cv2.rectangle(image_np, (x, y), (x + w, y + h), 
                         CONFIG["BBOX_COLOR"], CONFIG["BBOX_THICKNESS"])
            
            # Face label
            label = f"Face {i+1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image_np, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), CONFIG["BBOX_COLOR"], -1)
            cv2.putText(image_np, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw landmarks
            for (lx, ly) in landmarks_scaled:
                cv2.circle(image_np, (int(lx), int(ly)), 
                          CONFIG["LANDMARK_RADIUS"], CONFIG["LANDMARK_COLOR"], -1)

        return Image.fromarray(image_np), len(faces), all_landmarks
        
    except Exception as e:
        logger.error(f"Landmark detection failed: {e}")
        return image, 0, []

# --- 4. Streamlit UI Components ---

def setup_sidebar() -> Tuple[Optional[cv2.CascadeClassifier], Optional[FaceLandmarkNetwork]]:
    """Setup sidebar with status indicators"""
    st.sidebar.title("⚙️ Settings")
    
    # System status
    st.sidebar.subheader("System Status")
    
    # Cascade status
    cascade_path = find_cascade_file()
    cascade = load_face_cascade(cascade_path) if cascade_path else None
    
    if cascade:
        st.sidebar.success("✅ Face Detector: Ready")
    else:
        st.sidebar.error("❌ Face Detector: Not Found")
        if st.sidebar.button("📥 Download Face Detector", key="download_cascade"):
            if download_cascade_file():
                st.rerun()
    
    # Model status
    model_exists = os.path.exists(CONFIG["MODEL_PATH"])
    model = load_landmark_model(CONFIG["MODEL_PATH"]) if model_exists else None
    
    if model:
        st.sidebar.success("✅ Landmark Model: Ready")
    else:
        st.sidebar.warning("⚠️ Landmark Model: Not Found")
    
    # Detection settings
    st.sidebar.subheader("Detection Settings")
    if cascade:
        # Use session state to persist settings
        if 'detection_settings' not in st.session_state:
            st.session_state.detection_settings = CONFIG.copy()
        
        st.session_state.detection_settings["DETECTION_SCALE_FACTOR"] = st.sidebar.slider(
            "Scale Factor", 1.01, 1.5, 1.1, 0.01,
            help="How much to reduce image size at each scale"
        )
        st.session_state.detection_settings["DETECTION_MIN_NEIGHBORS"] = st.sidebar.slider(
            "Min Neighbors", 1, 10, 5,
            help="Higher values reduce false positives"
        )
    
    return cascade, model

def render_main_interface(cascade: Optional[cv2.CascadeClassifier], 
                         model: Optional[FaceLandmarkNetwork]):
    """Render the main application interface"""
    
    st.title("🎯 Advanced Face Landmarks Detection")
    st.markdown("---")
    
    # Mode selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if model:
            mode = st.radio(
                "🔍 Detection Mode:",
                ["Face + Landmarks", "Face Only"],
                index=0,
                help="Choose between face detection only or with landmarks"
            )
        else:
            mode = "Face Only"
            st.info("💡 Train a model to enable landmark detection")
    
    with col2:
        input_method = st.radio(
            "📸 Input Method:",
            ["Upload Image", "Use Camera"],
            horizontal=True
        )
    
    # Image input
    image_to_process = None
    image_source = None
    
    if input_method == "Use Camera":
        st.info("👆 Click below to capture from camera")
        camera_image = st.camera_input("Take a picture", label_visibility="collapsed")
        if camera_image:
            image_to_process = Image.open(camera_image).convert('RGB')
            image_source = "Camera Capture"
    else:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
        )
        if uploaded_file:
            image_to_process = Image.open(uploaded_file).convert('RGB')
            image_source = uploaded_file.name
    
    # Process image
    if image_to_process and cascade:
        process_and_display_image(image_to_process, image_source, cascade, model, mode)

def process_and_display_image(image: Image.Image, image_source: str,
                            cascade: cv2.CascadeClassifier, 
                            model: Optional[FaceLandmarkNetwork], mode: str):
    """Process image and display results"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Input Image")
        caption = f"📷 {image_source}" if image_source != "Camera Capture" else "📹 Camera Capture"
        st.image(image, caption=caption, use_container_width=True)
        
        # Image info
        with st.expander("📋 Image Details"):
            st.write(f"**Dimensions:** {image.size[0]} × {image.size[1]} pixels")
            st.write(f"**Mode:** {image.mode}")
            if hasattr(image, 'filename') and image.filename:
                st.write(f"**Format:** {image.format}")
    
    with col2:
        st.subheader("📊 Results")
        
        start_time = time.time()
        
        if mode == "Face + Landmarks" and model:
            with st.spinner('🔍 Detecting faces and landmarks...'):
                result_image, num_faces, landmarks = get_landmarks_on_image(image, model, cascade)
                processing_time = time.time() - start_time
                
            if num_faces > 0:
                st.success(f"✅ Detected {num_faces} face(s) with 68 landmarks each")
                st.image(result_image, caption=f"🎯 {num_faces} face(s) with landmarks", 
                        use_container_width=True)
                
                # Landmarks info
                with st.expander("📊 Detection Details"):
                    st.write(f"**Processing Time:** {processing_time:.2f}s")
                    st.write(f"**Total Landmarks:** {num_faces * 68}")
                    for i, face_landmarks in enumerate(landmarks):
                        st.write(f"**Face {i+1}:** {len(face_landmarks)} points")
            else:
                st.warning("⚠️ No faces detected. Try adjusting the detection settings.")
                st.image(image, use_container_width=True)
                
        else:  # Face detection only
            with st.spinner('🔍 Detecting faces...'):
                result_image, num_faces = detect_faces_only(image, cascade)
                processing_time = time.time() - start_time
                
            if num_faces > 0:
                st.success(f"✅ Detected {num_faces} face(s)")
                st.image(result_image, caption=f"👤 {num_faces} face(s) detected", 
                        use_container_width=True)
                
                with st.expander("📊 Detection Details"):
                    st.write(f"**Processing Time:** {processing_time:.2f}s")
            else:
                st.warning("⚠️ No faces detected")
                st.image(image, use_container_width=True)

def render_instructions():
    """Render usage instructions"""
    with st.expander("📚 How to Use & Tips"):
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Choose Detection Mode**: 
           - *Face Only*: Just detect face boundaries
           - *Face + Landmarks*: Detect 68 facial landmarks (requires model)
        
        2. **Select Input Method**:
           - *Upload Image*: Use existing photos
           - *Use Camera*: Real-time capture
        
        3. **View Results**: Detection appears instantly
        
        ### 💡 Tips for Best Results
        
        - **Lighting**: Ensure good, even lighting
        - **Angle**: Face the camera directly
        - **Clarity**: Keep face clearly visible
        - **Settings**: Adjust detection parameters if needed
        - **Format**: Use high-quality images
        
        ### 🔧 Technical Details
        
        - **Face Detection**: Haar Cascade Classifier
        - **Landmark Model**: ResNet-based neural network
        - **Landmarks**: 68 points per face (eyes, nose, mouth, etc.)
        - **Processing**: Real-time on CPU/GPU
        """)

# --- 5. Main Application ---

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="Face Landmarks Detection",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load components
    cascade, model = setup_sidebar()
    
    # Main interface
    if cascade:
        render_main_interface(cascade, model)
        render_instructions()
    else:
        st.error("❌ Face detector required")
        st.markdown("""
        **To enable face detection:**
        
        1. Click the "Download Face Detector" button in the sidebar, or
        2. Manually download from [OpenCV GitHub](https://github.com/opencv/opencv)
        
        Place `haarcascade_frontalface_default.xml` in the application directory.
        """)
        
        if st.button("🔄 Check Again", key="recheck_cascade"):
            st.rerun()

if __name__ == "__main__":
    main()