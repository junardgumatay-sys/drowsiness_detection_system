"""
CSC 126 - Computer Vision Midterm Project
Drowsiness and Attention Detection System

This script processes input images to detect faces using OpenCV Haar Cascades,
extracts facial landmarks via MediaPipe Face Landmarker, and computes the 
Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to monitor driver alertness.
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# System Constants and Thresholds
IMAGE_WIDTH = 1600
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.60
FACE_PADDING = 0.1

# Landmark Indices for MediaPipe Face Mesh
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14]

# BGR Color Definitions
GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def download_model():
    """Downloads the MediaPipe Face Landmarker task model if not present locally."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe Face Landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")


def preprocess(image):
    """
    Topic (a) Image Operations & (d) Image Transformation:
    Resizes image, converts to grayscale, applies Gaussian Blur for noise reduction,
    and applies CLAHE for contrast enhancement.
    """
    h, w = image.shape[:2]
    scale = IMAGE_WIDTH / float(w)
    image = cv2.resize(image, (IMAGE_WIDTH, int(h * scale)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return image, gray


def detect_faces(gray):
    """
    Topic (c) Haar Cascade Classifier:
    Detects bounding boxes around faces in the grayscale image.
    """
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    return cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))


def padded_roi(face, image_shape):
    """Calculates ROI coordinates with padding around detected face."""
    x, y, w, h = face
    ih, iw = image_shape[:2]
    px, py = int(w * FACE_PADDING), int(h * FACE_PADDING)
    return max(0, x - px), max(0, y - py), min(iw, x + w + px), min(ih, y + h + py)


def create_landmarker():
    """Initializes the MediaPipe FaceLandmarker object."""
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_landmarks(image, roi, landmarker):
    """
    Topic (f) Facial Landmark Detection:
    Extracts 3D spatial facial landmarks using MediaPipe.
    """
    x0, y0, x1, y1 = roi
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    ch, cw = crop.shape[:2]
    lm = result.face_landmarks[0]
    return np.array([[x0 + p.x * cw, y0 + p.y * ch] for p in lm], dtype=np.float32)


def dist(a, b):
    """Calculates Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts, idx):
    """Computes Eye Aspect Ratio (EAR) using vertical and horizontal landmark distances."""
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)


def mouth_aspect_ratio(pts):
    """Computes Mouth Aspect Ratio (MAR) to detect yawning behavior."""
    left, right, top, bottom = [pts[i] for i in MOUTH]
    return dist(top, bottom) / (dist(left, right) + 1e-6)


def put_text(image, text, pos, color=WHITE, scale=0.6, thick=2):
    """Draws text with a black outline for high contrast display."""
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_eye(image, pts, idx, color):
    """Draws polylines and keypoint markers around eyes."""
    poly = np.array([pts[i] for i in idx], dtype=np.int32)
    cv2.polylines(image, [poly], True, color, 1, cv2.LINE_AA)
    for p in poly:
        cv2.circle(image, (int(p[0]), int(p[1])), 1, color, -1)


def detect_drowsiness(image_path, output_path="result.jpg"):
    """Main execution pipeline for processing input image and detecting drowsiness."""
    download_model()
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError("Could not read image path: " + image_path)

    image, gray = preprocess(image)
    faces = detect_faces(gray)
    print(f"Faces detected: {len(faces)}")

    if len(faces) == 0:
        put_text(image, "No face detected", (10, 30), YELLOW, 0.8)
    else:
        landmarker = create_landmarker()
        for idx_num, face in enumerate(faces, start=1):
            x, y, w, h = face
            cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)

            pts = get_landmarks(image, padded_roi(face, image.shape), landmarker)
            if pts is None:
                put_text(image, "No landmarks", (x, max(20, y - 10)), YELLOW, 0.5, 1)
                continue

            ear = (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0
            mar = mouth_aspect_ratio(pts)
            drowsy = ear < EAR_THRESHOLD
            yawning = mar > MAR_THRESHOLD

            color = RED if drowsy else GREEN
            label = "DROWSY" if drowsy else "Alert"

            draw_eye(image, pts, LEFT_EYE, color)
            draw_eye(image, pts, RIGHT_EYE, color)
            put_text(image, f"{idx_num}: {label}", (x, max(20, y - 28)), color, 0.5, 1)
            put_text(image, f"EAR {ear:.2f}", (x, max(20, y - 8)), WHITE, 0.5, 1)
            if yawning:
                put_text(image, "Yawning", (x, y + h + 18), YELLOW, 0.5, 1)

            print(f"Face {idx_num}: EAR={ear:.3f}, MAR={mar:.3f} -> {label}" + (", Yawning" if yawning else ""))
        landmarker.close()

    cv2.imwrite(output_path, image)
    print(f"Result image saved to {output_path}")
    return image


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/sample_inputs/alert_face.jpg"
    detect_drowsiness(input_file)