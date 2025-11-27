"""
Gender Detection Module
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
try:
    import mediapipe as mp
except ImportError:
    mp = None


class GenderDetector:
    """
    Gender detection based on body proportions and facial features
    """
    
    def __init__(self):
        """Initialize gender detector"""
        if mp is None:
            raise ImportError("MediaPipe is required for gender detection. Install with: pip install mediapipe")
        
        self.mp_face = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose
        self.face_detection = self.mp_face.FaceDetection(
            model_selection=0, 
            min_detection_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
    
    def detect_gender_from_proportions(self, pose_data: Dict) -> Dict:
        """
        Detect gender based on body proportions
        
        Args:
            pose_data: Pose detection results with landmarks
            
        Returns:
            Dictionary with gender prediction and confidence
        """
        if pose_data is None or "landmarks" not in pose_data:
            return {"gender": "unknown", "confidence": 0.0, "method": "proportions"}
        
        landmarks = pose_data["landmarks"]
        
        # MediaPipe landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        
        def distance_2d(p1: Dict, p2: Dict) -> float:
            return np.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)
        
        # Calculate key measurements
        shoulder_width = 0
        hip_width = 0
        waist_to_hip_ratio = 0
        
        if LEFT_SHOULDER in landmarks and RIGHT_SHOULDER in landmarks:
            shoulder_width = distance_2d(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
        
        if LEFT_HIP in landmarks and RIGHT_HIP in landmarks:
            hip_width = distance_2d(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
        
        if shoulder_width > 0 and hip_width > 0:
            waist_to_hip_ratio = shoulder_width / hip_width
        
        # Gender classification based on body proportions
        # Generally: males have wider shoulders, females have wider hips
        score = 0.0
        confidence = 0.0
        
        if waist_to_hip_ratio > 1.05:  # Wider shoulders than hips
            score += 0.4  # Likely male
        elif waist_to_hip_ratio < 0.95:  # Wider hips than shoulders
            score -= 0.4  # Likely female
        
        # Additional checks
        if shoulder_width > hip_width * 1.1:
            score += 0.3
        elif hip_width > shoulder_width * 1.1:
            score -= 0.3
        
        # Normalize score to confidence
        confidence = min(abs(score), 1.0)
        gender = "male" if score > 0.1 else "female" if score < -0.1 else "unknown"
        
        return {
            "gender": gender,
            "confidence": confidence,
            "method": "proportions",
            "shoulder_width": float(shoulder_width),
            "hip_width": float(hip_width),
            "waist_to_hip_ratio": float(waist_to_hip_ratio)
        }
    
    def detect_gender_from_face(self, image: np.ndarray) -> Dict:
        """
        Detect gender from facial features (simplified)
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Dictionary with gender prediction
        """
        results = self.face_detection.process(image)
        
        if not results.detections:
            return {"gender": "unknown", "confidence": 0.0, "method": "face"}
        
        # Simplified: based on face size and position
        # In production, use a proper gender classification model
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # Face width/height ratio can indicate gender (males typically have wider faces)
        face_width = bbox.width
        face_height = bbox.height
        
        if face_height > 0:
            aspect_ratio = face_width / face_height
            if aspect_ratio > 0.75:
                return {"gender": "male", "confidence": 0.6, "method": "face"}
            else:
                return {"gender": "female", "confidence": 0.6, "method": "face"}
        
        return {"gender": "unknown", "confidence": 0.0, "method": "face"}
    
    def detect_gender(self, image: np.ndarray, pose_data: Optional[Dict] = None) -> Dict:
        """
        Main gender detection method combining multiple approaches
        
        Args:
            image: Input image (RGB)
            pose_data: Optional pre-computed pose data
            
        Returns:
            Combined gender prediction with confidence
        """
        # Method 1: Body proportions
        if pose_data is None:
            pose_results = self.pose.process(image)
            if pose_results.pose_landmarks:
                h, w = image.shape[:2]
                landmarks = {}
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    landmarks[idx] = {
                        "x": landmark.x * w,
                        "y": landmark.y * h,
                        "z": landmark.z * w,
                        "visibility": landmark.visibility
                    }
                pose_data = {"landmarks": landmarks, "width": w, "height": h}
        
        prop_result = self.detect_gender_from_proportions(pose_data)
        
        # Method 2: Facial features
        face_result = self.detect_gender_from_face(image)
        
        # Combine results
        if prop_result["confidence"] > 0.7:
            return prop_result
        elif face_result["confidence"] > 0.5:
            return face_result
        elif prop_result["gender"] != "unknown":
            return prop_result
        else:
            return face_result

