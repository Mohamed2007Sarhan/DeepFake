utf-8"""
Body Size Estimation Module
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional, List
from .utils.image_processor import ImageProcessor
from .utils.gender_detector import GenderDetector


class BodySizeEstimator:
    """
    Advanced body size estimation system
    Estimates body measurements and dimensions from images
    """
    
    def __init__(self, use_mediapipe: bool = True, reference_height: float = 170.0):
        """
        Initialize body size estimator
        
        Args:
            use_mediapipe: Use MediaPipe pose estimation
            reference_height: Reference height in cm for measurement estimation
        """
        self.use_mediapipe = use_mediapipe
        self.reference_height = reference_height
        self.image_processor = ImageProcessor()
        self.gender_detector = GenderDetector()
        
        if use_mediapipe:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
    
    def detect_pose_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect pose landmarks using MediaPipe
        
        Returns:
            Dictionary with landmarks and dimensions
        """
        if not self.use_mediapipe:
            return None
        
        
        results = self.pose.process(image)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        h, w = image.shape[:2]
        
        
        key_points = {}
        for idx, landmark in enumerate(landmarks):
            key_points[idx] = {
                "x": landmark.x * w,
                "y": landmark.y * h,
                "z": landmark.z * w,  
                "visibility": landmark.visibility
            }
        
        return {
            "landmarks": key_points,
            "width": w,
            "height": h,
            "segmentation_mask": results.segmentation_mask
        }
    
    def calculate_body_measurements(self, pose_data: Dict) -> Dict:
        """
        Calculate body measurements from pose landmarks
        
        Args:
            pose_data: Pose detection results
            
        Returns:
            Dictionary with body measurements
        """
        if pose_data is None:
            return {"error": "No pose data available"}
        
        landmarks = pose_data["landmarks"]
        h, w = pose_data["height"], pose_data["width"]
        
        
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        
        def distance_2d(p1: Dict, p2: Dict) -> float:
            """Calculate 2D distance between two points"""
            return np.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)
        
        def distance_3d(p1: Dict, p2: Dict) -> float:
            """Calculate 3D distance between two points"""
            return np.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2 + (p1["z"] - p2["z"])**2)
        
        measurements = {}
        
        
        
        shoulder_avg_y = (landmarks[LEFT_SHOULDER]["y"] + landmarks[RIGHT_SHOULDER]["y"]) / 2
        ankle_avg_y = (landmarks[LEFT_ANKLE]["y"] + landmarks[RIGHT_ANKLE]["y"]) / 2
        pixel_height = abs(shoulder_avg_y - ankle_avg_y)
        
        
        if LEFT_SHOULDER in landmarks and RIGHT_SHOULDER in landmarks:
            shoulder_width = distance_2d(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
            measurements["shoulder_width_pixels"] = shoulder_width
            measurements["shoulder_width_cm"] = (shoulder_width / pixel_height) * self.reference_height
        
        
        if LEFT_HIP in landmarks and RIGHT_HIP in landmarks:
            hip_width = distance_2d(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
            measurements["hip_width_pixels"] = hip_width
            measurements["hip_width_cm"] = (hip_width / pixel_height) * self.reference_height
        
        
        if LEFT_WRIST in landmarks and RIGHT_WRIST in landmarks:
            arm_span = distance_2d(landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST])
            measurements["arm_span_pixels"] = arm_span
            measurements["arm_span_cm"] = (arm_span / pixel_height) * self.reference_height
        
        
        if LEFT_HIP in landmarks and LEFT_ANKLE in landmarks:
            leg_length = distance_2d(landmarks[LEFT_HIP], landmarks[LEFT_ANKLE])
            measurements["leg_length_pixels"] = leg_length
            measurements["leg_length_cm"] = (leg_length / pixel_height) * self.reference_height
        
        
        measurements["pixel_height"] = pixel_height
        measurements["estimated_height_cm"] = self.reference_height
        measurements["image_height"] = h
        measurements["image_width"] = w
        
        
        if "shoulder_width_cm" in measurements and "hip_width_cm" in measurements:
            shoulder_hip_ratio = measurements["shoulder_width_cm"] / measurements["hip_width_cm"] if measurements["hip_width_cm"] > 0 else 1.0
            if shoulder_hip_ratio > 1.1:
                measurements["body_type"] = "Inverted Triangle"
            elif shoulder_hip_ratio < 0.9:
                measurements["body_type"] = "Pear"
            else:
                measurements["body_type"] = "Rectangle"
        else:
            measurements["body_type"] = "Unknown"
        
        return measurements
    
    def estimate_body_size(self, image_path: str) -> Dict:
        """
        Main method to estimate body size from image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with body size information
        """
        
        image = self.image_processor.load_image(image_path)
        
        
        pose_data = self.detect_pose_landmarks(image)
        
        if pose_data is None:
            return {
                "error": "Could not detect pose in image",
                "measurements": {}
            }
        
        
        measurements = self.calculate_body_measurements(pose_data)
        
        
        gender_info = self.gender_detector.detect_gender(image, pose_data)
        
        return {
            "pose_detected": True,
            "measurements": measurements,
            "gender": gender_info,
            "image_dimensions": {
                "width": pose_data["width"],
                "height": pose_data["height"]
            }
        }
    
    def get_body_bbox(self, pose_data: Dict, padding: int = 20) -> Tuple[int, int, int, int]:
        """
        Get bounding box around body
        
        Args:
            pose_data: Pose detection results
            padding: Padding around bounding box
            
        Returns:
            Tuple (x, y, w, h) bounding box
        """
        if pose_data is None or "landmarks" not in pose_data:
            return (0, 0, 0, 0)
        
        landmarks = pose_data["landmarks"]
        
        
        x_coords = [p["x"] for p in landmarks.values()]
        y_coords = [p["y"] for p in landmarks.values()]
        
        x_min = max(0, int(min(x_coords)) - padding)
        y_min = max(0, int(min(y_coords)) - padding)
        x_max = min(pose_data["width"], int(max(x_coords)) + padding)
        y_max = min(pose_data["height"], int(max(y_coords)) + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def segment_body(self, image: np.ndarray) -> np.ndarray:
        """
        Segment body from background
        
        Returns:
            Binary mask of body region
        """
        pose_data = self.detect_pose_landmarks(image)
        
        if pose_data is None or pose_data.get("segmentation_mask") is None:
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return mask
        
        
        mask = (pose_data["segmentation_mask"] * 255).astype(np.uint8)
        return mask

