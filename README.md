# Face Recognition & Landmarks Detection

A web application for detecting facial landmarks using a trained PyTorch model. The app uses OpenCV's Haar Cascade for face detection and a custom ResNet18-based model for landmark prediction.

## 🚀 Project Setup

### Prerequisites

- Python 3.7 or higher
- Trained model file (`face_landmarks.pth`)
- Haar Cascade XML file (`haarcascade_frontalface_default.xml`)

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Haar Cascade file:**
   - **Option 1 (Recommended):** Run `python download_cascade.py` to automatically download it
   - **Option 2:** Download `haarcascade_frontalface_default.xml` manually from [OpenCV GitHub](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
   - **Option 3:** The app includes a button to download it automatically when you run it
   - Save it in the same folder as `app.py`

3. **Place your trained model:**
   - Ensure your trained model file `face_landmarks.pth` is in the same folder as `app.py`

### File Structure

Your project folder should contain:
```
sujal_patel_06/
├── app.py
├── requirements.txt
├── face_landmarks.pth          # Your trained model
├── haarcascade_frontalface_default.xml  # Face detector
└── README.md
```

## 🏃 Running the App

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   - The app will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

3. **Use the app:**
   - Click "Choose an image..." to upload a photo
   - Click "Find Landmarks" to detect and visualize facial landmarks
   - The app will show the original image and the image with detected landmarks side by side

## 📝 Features

- **Face Detection**: Automatically detects faces in uploaded images using Haar Cascade
- **Landmark Prediction**: Predicts 68 facial landmarks using the trained ResNet18 model
- **Visualization**: Displays landmarks as green dots on the detected faces
- **Multiple Faces**: Supports detection of multiple faces in a single image

## 🔧 Troubleshooting

- **Missing files error**: Make sure both `face_landmarks.pth` and `haarcascade_frontalface_default.xml` are in the same directory as `app.py`
- **No faces detected**: Try uploading a clearer image with a visible face
- **Model loading errors**: Ensure your model architecture matches the `Network` class defined in `app.py`

## 📦 Deployment

To deploy this app to Streamlit Community Cloud:
1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository and deploy

Make sure to include all required files (`face_landmarks.pth` and `haarcascade_frontalface_default.xml`) in your repository.

