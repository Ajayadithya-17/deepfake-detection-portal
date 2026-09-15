# AI-Based Deepfake Detection System Using Deep Learning

**BSc Data Science Final-Year Capstone Project**  
*Project Title*: **AI-Based Deepfake Detection System Using Deep Learning**  
*Technology Stack*: Python, TensorFlow, Keras, OpenCV, MTCNN, Flask, HTML5, CSS3, JavaScript

---

## 1. Abstract & Project Motivation

With the exponential proliferation of deep generative neural networks (Generative Adversarial Networks, Latent Diffusion Models, and neural face-swapping pipelines), synthetic media manipulation has become a critical challenge to digital authentication, information integrity, and cybersecurity. 

This project presents an end-to-end, web-based deepfake detection system engineered specifically for academic rigor and real-world deployment. The architecture couples **Transfer Learning via EfficientNet-B3** for high-resolution spatial facial artifact classification with a **Bidirectional Long Short-Term Memory (LSTM)** sequence model for temporal inter-frame anomaly detection across video streams.

---

## 2. System Architecture

```
                                  [ User Upload ]
                                         │
                                         ▼
                            [ Face Detection & Crop ]
                       MTCNN (Multi-task Cascaded CNN)
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼ (Static Images)                           ▼ (Video Streams)
      [ Image Preprocessing ]                      [ Temporal Frame Extraction ]
      • Aspect-ratio preserved padding             • 20 Equidistant keyframes
      • Standardized to (300, 300, 3)              • Face localized per frame
      • ImageNet normalization                     • Batch sequence: (20, 300, 300, 3)
                   │                                           │
                   ▼                                           ▼
      [ EfficientNet-B3 Backbone ]                 [ EfficientNet-B3 Feature Pooling ]
      • Pretrained ImageNet weights                • Global Average Pooling
      • GlobalAveragePooling2D + Dense             • Feature Sequence: (20, 1536)
      • Sigmoid Classification Probability                     │
                   │                                           ▼
                   │                               [ Bidirectional LSTM Network ]
                   │                               • Temporal jitter & flicker analysis
                   │                               • Sigmoid Video Classification
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                             [ Flask REST API Layer ]
                         • Endpoints: /predict_image & /predict_video
                         • Probabilistic calibration & JSON telemetry
                                         │
                                         ▼
                        [ Modern Dark AI-Themed Web Portal ]
                        • Drag & Drop interface with live previews
                        • Real-time laser scanning animations
                        • REAL (Emerald) vs DEEPFAKE (Crimson) results
```

---

## 3. Deep Learning & Mathematical Methodology

### A. Spatial Feature Extraction (EfficientNet-B3)
EfficientNet-B3 utilizes **compound scaling** to uniformly scale depth $d$, width $w$, and input image resolution $r$ with a fixed set of scaling coefficients:

$$\text{Depth}: d = \alpha^\phi, \quad \text{Width}: w = \beta^\phi, \quad \text{Resolution}: r = \gamma^\phi$$

$$\text{Subject to}: \quad \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2, \quad \alpha \ge 1, \beta \ge 1, \gamma \ge 1$$

For EfficientNet-B3, the native input resolution is **$300 \times 300 \times 3$**. Transfer learning is applied by initializing with ImageNet pretrained weights and appending a custom binary classification head:
- `GlobalAveragePooling2D()`
- `BatchNormalization()`
- `Dense(256, activation='relu')`
- `Dropout(0.4)`
- `Dense(1, activation='sigmoid')`

### B. Temporal Sequence Modeling (CNN-LSTM)
Deepfake videos frequently exhibit temporal discrepancies, such as abnormal eye-blink cadence, unnatural lip-sync fluttering, and frame-to-frame boundary jitter. 
1. **Feature Extraction**: Each of the 20 sampled face frames is passed through the frozen EfficientNet-B3 convolutional base to generate a 1,536-dimensional feature representation:
   $$\mathbf{f}_t = \text{EfficientNetB3}(\mathbf{x}_t) \in \mathbb{R}^{1536}, \quad t \in \{1, 2, \dots, 20\}$$
2. **Bidirectional LSTM**: Captures bidirectional temporal dependencies through recurrent gating:
   $$i_t = \sigma(W_{xi} \mathbf{f}_t + W_{hi} h_{t-1} + b_i)$$
   $$f_t = \sigma(W_{xf} \mathbf{f}_t + W_{hf} h_{t-1} + b_f)$$
   $$o_t = \sigma(W_{xo} \mathbf{f}_t + W_{ho} h_{t-1} + b_o)$$
   $$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_{xc} \mathbf{f}_t + W_{hc} h_{t-1} + b_c)$$
   $$h_t = o_t \odot \tanh(c_t)$$

---

## 4. Project Directory Structure

```
Deepfake-Detection-System/
├── app.py                         # Flask application & prediction API endpoints
├── requirements.txt               # Pinned Python package dependencies
├── README.md                      # Academic project documentation
│
├── models/
│   ├── image_model.keras          # Trained EfficientNet-B3 model
│   └── video_model.keras          # Trained CNN-LSTM video model
│
├── dataset/
│   ├── real/
│   │   ├── images/                # Authentic reference portraits
│   │   └── videos/                # Authentic camera videos
│   └── fake/
│       ├── images/                # Deepfake & synthetic portraits
│       └── videos/                # AI-generated & face-swapped videos
│
├── preprocessing/
│   ├── __init__.py
│   ├── face_detection.py          # MTCNN face extraction & boundary margin padding
│   ├── image_preprocessing.py     # Resize (300x300), normalization, data augmentation
│   └── video_processing.py        # 20-frame equidistant extraction & sequence tensor prep
│
├── training/
│   ├── __init__.py
│   ├── train_image_model.py       # Transfer learning, 70/15/15 split, callbacks, metrics
│   └── train_video_model.py       # CNN-LSTM sequence training & evaluation
│
├── static/
│   ├── css/
│   │   └── style.css              # Custom Dark AI-themed CSS (glassmorphism, animations)
│   ├── js/
│   │   └── script.js              # Drag-and-drop, client-side preview, async AJAX calls
│   ├── frames/                    # Extracted keyframe previews for UI timeline
│   └── uploads/                   # Upload storage
│
└── templates/
    └── index.html                 # Responsive HTML5 web portal template
```

---

## 5. Step-by-Step Installation & Setup in VS Code

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed.
- VS Code with the Python extension installed.

### Step 1: Open Project in VS Code
Open VS Code, press `Ctrl+Shift+P` (or `Cmd+Shift+P`), select **File: Open Folder...**, and select the project directory:
```powershell
cd g:\deepfake_detection_portal
```

### Step 2: Create and Activate a Virtual Environment
In the VS Code integrated terminal (`Ctrl + ~`):
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 6. Training the Deep Learning Models

The training scripts automatically partition the dataset into **70% Training**, **15% Validation**, and **15% Testing** splits.

### Train Image Model (EfficientNet-B3)
```powershell
python training/train_image_model.py
```
- Applies data augmentation (RandomFlip, RandomRotation, RandomZoom, RandomContrast).
- Integrates `EarlyStopping(patience=4)`, `ModelCheckpoint`, and `ReduceLROnPlateau`.
- Evaluates on the unseen test partition and saves `models/image_model.keras` and `models/image_confusion_matrix.png`.

### Train Video Model (CNN-LSTM)
```powershell
python training/train_video_model.py
```
- Extracts 20 equidistant frames per video sample.
- Extracts spatial feature representations using pretrained EfficientNet-B3.
- Trains the Bidirectional LSTM sequence classifier.
- Saves the trained model to `models/video_model.keras`.

---

## 7. Running the Web Application

Launch the Flask server:
```powershell
python app.py
```

The server will initialize the deep learning models and host the application locally:
```text
[SYSTEM] Loaded EfficientNet-B3 image model from models/image_model.keras
[SYSTEM] Loaded CNN-LSTM video model from models/video_model.keras
[SYSTEM] Initializing Video Feature Extractor Backbone...
[SYSTEM] System Core Online & Ready for Inference.
 * Running on http://127.0.0.1:5000
```

Open your web browser and navigate to:  
👉 **`http://127.0.0.1:5000`**

---

## 8. Web Portal Usage Guide

1. **Image Forensics**:
   - Select the **"Image Detection (EfficientNet-B3)"** tab.
   - Drag and drop or browse a portrait image (`.jpg`, `.png`, `.webp`).
   - The interactive preview displays the loaded image.
   - Click **"Analyze Authenticity"**.
   - The HUD scanner animates during MTCNN face localization and model inference.
   - The **Result Card** reveals the classification:
     - **REAL / AUTHENTIC** (positive emerald glow)
     - **DEEPFAKE DETECTED** (warning crimson glow)
     - Primary confidence score and dual-channel softmax probability distribution.

2. **Video Forensics**:
   - Select the **"Video Detection (CNN-LSTM)"** tab.
   - Drag and drop or browse a video file (`.mp4`, `.avi`, `.mov`).
   - Click **"Analyze Authenticity"**.
   - The system extracts 20 equidistant frames, crops faces, and evaluates temporal sequence dynamics.
   - The **Keyframe Decomposition Sequence** displays the sampled face thumbnails alongside the final prediction.

---

## 9. Performance Evaluation & Academic Disclaimer

### Evaluation Metrics
The system evaluates models using standard Data Science classification metrics:
- **Accuracy**: Overall proportion of correctly classified media.
- **Precision**: $\frac{TP}{TP + FP}$ (Minimizing false accusations of deepfakes).
- **Recall**: $\frac{TP}{TP + FN}$ (Catching all manipulated media).
- **F1 Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ (Harmonic mean).
- **Confusion Matrix**: Visual representation of True Positives, True Negatives, False Positives, and False Negatives.

### Academic Disclaimer
In accordance with ethical data science guidelines, predictions generated by this system represent **calibrated probabilistic likelihood scores** based on learned convolutional artifact representations and temporal sequence consistency. No automated computer vision system can guarantee 100% deterministic certainty.

