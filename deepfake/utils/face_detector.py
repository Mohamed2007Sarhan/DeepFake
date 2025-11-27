"""
Face Detection and Alignment Module
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
from pathlib import Path

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    dlib = None


class FaceDetector:
    """
    Advanced face detection and alignment using multiple methods
    """
    
    def __init__(self, method: str = "dlib", model_path: Optional[str] = None):
        """
        Initialize face detector
        
        Args:
            method: "dlib", "mediapipe", or "opencv"
            model_path: Path to dlib shape predictor model (if using dlib)
        """
        self.method = method
        self.detector = None
        self.predictor = None
        self.face_cascade = None
        
        if method == "dlib":
            self._init_dlib(model_path)
        elif method == "opencv":
            self._init_opencv()
        elif method == "mediapipe":
            self._init_mediapipe()
    
    def _init_dlib(self, model_path: Optional[str] = None):
        """Initialize dlib face detector and landmark predictor"""
        if not DLIB_AVAILABLE:
            print("Warning: dlib not available. Falling back to OpenCV.")
            self.method = "opencv"
            self._init_opencv()
            return
        
        try:
            self.detector = dlib.get_frontal_face_detector()
            
            # Try to load shape predictor
            if model_path and os.path.exists(model_path):
                self.predictor = dlib.shape_predictor(model_path)
            else:
                # Try default path
                default_path = Path(__file__).parent.parent.parent / "models" / "shape_predictor_68_face_landmarks.dat"
                if default_path.exists():
                    self.predictor = dlib.shape_predictor(str(default_path))
                else:
                    print("Warning: dlib shape predictor not found. Using basic detection only.")
                    self.predictor = None
        except ImportError:
            print("Warning: dlib not available. Falling back to OpenCV.")
            self.method = "opencv"
            self._init_opencv()
    
    def _init_opencv(self):
        """Initialize OpenCV face detector"""
        try:
            # Load Haar Cascade for face detection
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"Warning: Could not initialize OpenCV face detector: {e}")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe face detection"""
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        except ImportError:
            print("Warning: MediaPipe not available. Falling back to OpenCV.")
            self.method = "opencv"
            self._init_opencv()
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image
        
        Args:
            image: Input image (RGB or BGR)
            
        Returns:
            List of face detection dictionaries with bounding boxes
        """
        if self.method == "dlib":
            return self._detect_faces_dlib(image)
        elif self.method == "mediapipe":
            return self._detect_faces_mediapipe(image)
        else:
            return self._detect_faces_opencv(image)
    
    def _detect_faces_dlib(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using dlib"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        faces = self.detector(gray)
        
        results = []
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            results.append({
                "bbox": (x, y, w, h),
                "confidence": 1.0,
                "method": "dlib"
            })
        return results
    
    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe"""
        results_list = []
        detection_results = self.face_detection.process(image)
        
        if detection_results.detections:
            h, w = image.shape[:2]
            for detection in detection_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                results_list.append({
                    "bbox": (x, y, width, height),
                    "confidence": detection.score[0],
                    "method": "mediapipe"
                })
        
        return results_list
    
    def _detect_faces_opencv(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                "bbox": (x, y, w, h),
                "confidence": 1.0,
                "method": "opencv"
            })
        return results
    
    def get_face_landmarks(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Get facial landmarks for a detected face
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Array of landmark points (68 points for dlib, or simplified)
        """
        if self.method == "dlib" and self.predictor:
            return self._get_landmarks_dlib(image, bbox)
        elif self.method == "mediapipe":
            return self._get_landmarks_mediapipe(image, bbox)
        else:
            return self._get_landmarks_basic(image, bbox)
    
    def _get_landmarks_dlib(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Get landmarks using dlib"""
        x, y, w, h = bbox
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Create dlib rectangle
        rect = dlib.rectangle(x, y, x + w, y + h)
        shape = self.predictor(gray, rect)
        
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        return landmarks
    
    def _get_landmarks_mediapipe(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Get landmarks using MediaPipe"""
        # Extract face region
        x, y, w, h = bbox
        face_roi = image[y:y+h, x:x+w]
        
        results = self.face_mesh.process(face_roi)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h_roi, w_roi = face_roi.shape[:2]
            
            # Convert to absolute coordinates
            points = []
            for landmark in landmarks.landmark:
                px = int(landmark.x * w_roi) + x
                py = int(landmark.y * h_roi) + y
                points.append([px, py])
            
            return np.array(points)
        
        # Fallback to basic landmarks
        return self._get_landmarks_basic(image, bbox)
    
    def _get_landmarks_basic(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Generate basic facial landmarks (estimated)"""
        x, y, w, h = bbox
        
        # Estimate key facial points (simplified 68-point approximation)
        landmarks = np.array([
            [x + w * 0.5, y + h * 0.3],  # Nose tip (0)
            [x + w * 0.33, y + h * 0.4],  # Left eye (1)
            [x + w * 0.67, y + h * 0.4],  # Right eye (2)
            [x + w * 0.33, y + h * 0.6],  # Left mouth (3)
            [x + w * 0.67, y + h * 0.6],  # Right mouth (4)
            [x + w * 0.5, y + h * 0.65],  # Mouth center (5)
        ])
        
        return landmarks
    
    def extract_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                    padding: int = 50) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract face region from image with padding
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            padding: Padding around face
            
        Returns:
            Tuple of (face_image, adjusted_bbox)
        """
        h, w = image.shape[:2]
        x, y, face_w, face_h = bbox
        
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + face_w + padding)
        y2 = min(h, y + face_h + padding)
        
        face_roi = image[y1:y2, x1:x2]
        adjusted_bbox = (x1, y1, x2 - x1, y2 - y1)
        
        return face_roi, adjusted_bbox
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray,
                  target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align face using landmarks
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            target_size: Output size (width, height)
            
        Returns:
            Tuple of (aligned_image, transform_matrix)
        """
        # Reference landmarks for alignment (68-point standard)
        if len(landmarks) >= 6:
            # Use eye points for alignment
            if len(landmarks) >= 68:
                # Full dlib landmarks
                left_eye = landmarks[36:42].mean(axis=0)
                right_eye = landmarks[42:48].mean(axis=0)
            else:
                # Simplified landmarks
                left_eye = landmarks[1] if len(landmarks) > 1 else landmarks[0]
                right_eye = landmarks[2] if len(landmarks) > 2 else landmarks[0]
            
            # Calculate angle
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.arctan2(dy, dx) * 180 / np.pi
            
            # Calculate center
            center = landmarks.mean(axis=0)
            
            # Rotation matrix
            M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
            
            # Apply rotation
            aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            
            # Crop and resize
            aligned = cv2.resize(aligned, target_size)
            
            return aligned, M
        else:
            # Fallback: just resize
            aligned = cv2.resize(image, target_size)
            return aligned, np.eye(2, 3)
    
    def get_face_mask(self, image: np.ndarray, landmarks: np.ndarray,
                     dilation: int = 10) -> np.ndarray:
        """
        Create face mask from landmarks
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            dilation: Mask dilation in pixels
            
        Returns:
            Binary mask
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(landmarks) >= 6:
            # Create convex hull from landmarks
            hull = cv2.convexHull(landmarks.astype(np.int32))
            cv2.fillPoly(mask, [hull], 255)
        else:
            # Simple elliptical mask
            if len(landmarks) > 0:
                center = landmarks.mean(axis=0).astype(int)
                axes = (int(np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])),
                       int(np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])))
                cv2.ellipse(mask, tuple(center), (axes[0]//2, axes[1]//2), 0, 0, 360, 255, -1)
        
        # Dilate mask
        if dilation > 0:
            kernel = np.ones((dilation, dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask

