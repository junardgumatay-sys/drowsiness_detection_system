# Driver Drowsiness and Attention Detection System

**Course:** CSC 126 – Computer Vision Midterm Project  
**Institution:** Caraga State University - CCIS  

## Overview
This computer vision application detects driver drowsiness and fatigue from images using OpenCV and MediaPipe. The system isolates face regions using Haar Cascade Classifiers, extracts detailed facial mesh coordinates, and measures both Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to evaluate drowsiness and yawning states.

---

## Applied OpenCV & Computer Vision Techniques
1. **Image Operations (Topic a):** Image scaling, geometric padding around detected Regions of Interest (ROI), color-space conversions (BGR to Grayscale/RGB), and annotated overlays.
2. **Image Analysis & Transformation (Topic d):** Gaussian blurring for high-frequency noise smoothing and Contrast Limited Adaptive Histogram Equalization (CLAHE) to normalize varied light levels.
3. **Haar Cascade Classifier (Topic c):** Rapid detection and bounding box extraction for human face region candidates.
4. **Facial Landmark Detection (Topic f):** MediaPipe Face Landmarker model extracting exact geometric points for structural evaluation of eyes and mouth.

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher

### Installation Steps
1. Clone the repository:
   ```bash
   git clone [https://github.com/junardgumatay/drowsiness-detection-system.git](https://github.com/junardgumatay/drowsiness-detection-system.git)
   cd drowsiness-detection-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running via Python Script
To run the detection script on a sample input:
```bash
python main.py data/sample_inputs/alert_face.jpg
```
The processed output image with bounding boxes and status labels will be saved to `result.jpg`.

### Running in Google Colab
Open `drowsiness_detector.ipynb` directly in Google Colab, execute cells sequentially, and upload your image when prompted.

---

## System Evaluation & Thresholds
- **EAR Threshold (< 0.21):** Eye Aspect Ratio falling below 0.21 triggers a `DROWSY` state warning overlay in red.
- **MAR Threshold (> 0.60):** Mouth Aspect Ratio exceeding 0.60 triggers a `Yawning` indicator in yellow.