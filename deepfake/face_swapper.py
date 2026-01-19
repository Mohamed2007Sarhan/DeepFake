utf-8"""
Face Swapping Module
Supports image-to-image and image-to-video face swapping
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from .utils.face_detector import FaceDetector
from .utils.image_processor import ImageProcessor


class FaceSwapper:
    """
    Advanced face swapping system
    Supports swapping faces between two images or from image to video
    """
    
    def __init__(self, method: str = "dlib", model_path: Optional[str] = None):
        """
        Initialize face swapper
        
        Args:
            method: Detection method ("dlib", "mediapipe", "opencv")
            model_path: Path to dlib shape predictor model
        """
        self.face_detector = FaceDetector(method=method, model_path=model_path)
        self.image_processor = ImageProcessor()
    
    def swap_faces(self, source_image: np.ndarray,
                   target_image: np.ndarray,
                   source_face_idx: int = 0,
                   target_face_idx: int = 0,
                   blend_mode: str = "seamless",
                   blend_factor: float = 0.8) -> Tuple[np.ndarray, Dict]:
        """
        Swap face from source image to target image
        
        Args:
            source_image: Image containing source face (RGB)
            target_image: Image containing target face (RGB)
            source_face_idx: Index of face to use from source (if multiple)
            target_face_idx: Index of face to replace in target (if multiple)
            blend_mode: "seamless", "linear", or "feathered"
            blend_factor: Blending strength (0-1)
            
        Returns:
            Tuple of (swapped_image, metadata_dict)
        """
        
        source_faces = self.face_detector.detect_faces(source_image)
        target_faces = self.face_detector.detect_faces(target_image)
        
        if not source_faces:
            raise ValueError("No face detected in source image")
        if not target_faces:
            raise ValueError("No face detected in target image")
        
        
        source_face = source_faces[min(source_face_idx, len(source_faces) - 1)]
        target_face = target_faces[min(target_face_idx, len(target_faces) - 1)]
        
        
        source_landmarks = self.face_detector.get_face_landmarks(
            source_image, source_face["bbox"]
        )
        target_landmarks = self.face_detector.get_face_landmarks(
            target_image, target_face["bbox"]
        )
        
        if source_landmarks is None or target_landmarks is None:
            raise ValueError("Could not detect facial landmarks")
        
        
        result = self._perform_swap(
            source_image, target_image,
            source_face["bbox"], target_face["bbox"],
            source_landmarks, target_landmarks,
            blend_mode, blend_factor
        )
        
        metadata = {
            "source_faces_detected": len(source_faces),
            "target_faces_detected": len(target_faces),
            "source_face_used": source_face_idx,
            "target_face_used": target_face_idx,
            "blend_mode": blend_mode
        }
        
        return result, metadata
    
    def _perform_swap(self, source_img: np.ndarray, target_img: np.ndarray,
                     source_bbox: Tuple[int, int, int, int],
                     target_bbox: Tuple[int, int, int, int],
                     source_landmarks: np.ndarray,
                     target_landmarks: np.ndarray,
                     blend_mode: str,
                     blend_factor: float) -> np.ndarray:
        """
        Core face swapping algorithm
        """
        
        source_face, source_bbox_adj = self.face_detector.extract_face(
            source_img, source_bbox, padding=50
        )
        
        
        x1, y1, _, _ = source_bbox_adj
        source_landmarks_adj = source_landmarks.copy()
        source_landmarks_adj[:, 0] -= x1
        source_landmarks_adj[:, 1] -= y1
        
        
        aligned_source, source_transform = self.face_detector.align_face(
            source_face, source_landmarks_adj, target_size=(256, 256)
        )
        
        
        target_face_roi, target_bbox_adj = self.face_detector.extract_face(
            target_img, target_bbox, padding=50
        )
        
        
        x2, y2, _, _ = target_bbox_adj
        target_landmarks_adj = target_landmarks.copy()
        target_landmarks_adj[:, 0] -= x2
        target_landmarks_adj[:, 1] -= y2
        
        
        aligned_target, target_transform = self.face_detector.align_face(
            target_face_roi, target_landmarks_adj, target_size=(256, 256)
        )
        
        
        source_mask = self.face_detector.get_face_mask(
            aligned_source, 
            np.array([[p[0], p[1]] for p in source_landmarks_adj[:6]]),
            dilation=15
        )
        
        
        target_roi_h, target_roi_w = target_face_roi.shape[:2]
        resized_source = cv2.resize(aligned_source, (target_roi_w, target_roi_h))
        resized_mask = cv2.resize(source_mask, (target_roi_w, target_roi_h))
        
        
        color_corrected = self._color_correct(resized_source, target_face_roi, resized_mask)
        
        
        if blend_mode == "seamless":
            swapped_roi = self._seamless_blend(
                color_corrected, target_face_roi, resized_mask
            )
        elif blend_mode == "linear":
            swapped_roi = self._linear_blend(
                color_corrected, target_face_roi, resized_mask, blend_factor
            )
        else:  
            swapped_roi = self._feathered_blend(
                color_corrected, target_face_roi, resized_mask
            )
        
        
        result = target_img.copy()
        x, y, w, h = target_bbox_adj
        result[y:y+h, x:x+w] = swapped_roi
        
        return result
    
    def _color_correct(self, source: np.ndarray, target: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
        """
        Color correct source face to match target skin tone
        """
        
        source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
        
        
        mask_3d = (mask / 255.0)[:, :, np.newaxis]
        
        source_mean = (source_lab * mask_3d).sum(axis=(0, 1)) / (mask_3d.sum() + 1e-6)
        target_mean = (target_lab * mask_3d).sum(axis=(0, 1)) / (mask_3d.sum() + 1e-6)
        
        
        corrected = source_lab.copy()
        corrected[:, :, 1] = corrected[:, :, 1] - source_mean[1] + target_mean[1]
        corrected[:, :, 2] = corrected[:, :, 2] - source_mean[2] + target_mean[2]
        
        
        corrected = np.clip(corrected, 0, 255)
        
        
        corrected_rgb = cv2.cvtColor(corrected.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return corrected_rgb
    
    def _seamless_blend(self, source: np.ndarray, target: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
        """Seamless cloning blend"""
        
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cy, cx = [s // 2 for s in mask.shape]
        
        
        mask_3d = np.stack([mask] * 3, axis=2)
        result = cv2.seamlessClone(
            source, target, mask, (cx, cy), cv2.NORMAL_CLONE
        )
        
        return result
    
    def _linear_blend(self, source: np.ndarray, target: np.ndarray,
                     mask: np.ndarray, blend_factor: float) -> np.ndarray:
        """Linear blending"""
        mask_normalized = (mask / 255.0)[:, :, np.newaxis] * blend_factor
        
        result = (source.astype(np.float32) * mask_normalized +
                 target.astype(np.float32) * (1 - mask_normalized))
        
        return result.astype(np.uint8)
    
    def _feathered_blend(self, source: np.ndarray, target: np.ndarray,
                        mask: np.ndarray) -> np.ndarray:
        """Feathered edge blending"""
        
        blurred_mask = cv2.GaussianBlur(mask, (21, 21), 0) / 255.0
        blurred_mask_3d = blurred_mask[:, :, np.newaxis]
        
        result = (source.astype(np.float32) * blurred_mask_3d +
                 target.astype(np.float32) * (1 - blurred_mask_3d))
        
        return result.astype(np.uint8)
    
    def swap_face_to_video(self, source_image_path: str,
                          target_video_path: str,
                          output_video_path: str,
                          source_face_idx: int = 0,
                          blend_mode: str = "seamless",
                          fps: Optional[int] = None) -> Dict:
        """
        Swap face from image to video
        
        Args:
            source_image_path: Path to source image
            target_video_path: Path to target video
            output_video_path: Path to save output video
            source_face_idx: Index of face to use from source
            blend_mode: Blending mode
            fps: Output FPS (uses video FPS if None)
            
        Returns:
            Dictionary with processing metadata
        """
        
        source_image = self.image_processor.load_image(source_image_path)
        
        
        source_faces = self.face_detector.detect_faces(source_image)
        if not source_faces:
            raise ValueError("No face detected in source image")
        
        source_face = source_faces[min(source_face_idx, len(source_faces) - 1)]
        source_landmarks = self.face_detector.get_face_landmarks(
            source_image, source_face["bbox"]
        )
        
        if source_landmarks is None:
            raise ValueError("Could not detect landmarks in source image")
        
        
        cap = cv2.VideoCapture(target_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {target_video_path}")
        
        
        video_fps = fps if fps else int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (width, height))
        
        frame_count = 0
        faces_detected = 0
        
        print(f"Processing video: {total_frames} frames...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                
                target_faces = self.face_detector.detect_faces(frame_rgb)
                
                if target_faces:
                    
                    target_face = target_faces[0]
                    target_landmarks = self.face_detector.get_face_landmarks(
                        frame_rgb, target_face["bbox"]
                    )
                    
                    if target_landmarks is not None:
                        
                        swapped_frame = self._perform_swap(
                            source_image, frame_rgb,
                            source_face["bbox"], target_face["bbox"],
                            source_landmarks, target_landmarks,
                            blend_mode, 0.8
                        )
                        faces_detected += 1
                    else:
                        swapped_frame = frame_rgb
                else:
                    swapped_frame = frame_rgb
                
                
                swapped_bgr = cv2.cvtColor(swapped_frame, cv2.COLOR_RGB2BGR)
                out.write(swapped_bgr)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames...")
        
        finally:
            cap.release()
            out.release()
        
        metadata = {
            "total_frames": frame_count,
            "faces_detected": faces_detected,
            "fps": video_fps,
            "resolution": (width, height)
        }
        
        print(f"✓ Video processing complete! Faces detected in {faces_detected}/{frame_count} frames")
        
        return metadata
    
    def swap_faces_image(self, source_image_path: str,
                        target_image_path: str,
                        output_path: str,
                        source_face_idx: int = 0,
                        target_face_idx: int = 0,
                        blend_mode: str = "seamless",
                        blend_factor: float = 0.8) -> Dict:
        """
        Swap faces between two images
        
        Args:
            source_image_path: Path to source image
            target_image_path: Path to target image
            output_path: Path to save result
            source_face_idx: Index of source face
            target_face_idx: Index of target face
            blend_mode: Blending mode
            blend_factor: Blending factor
            
        Returns:
            Metadata dictionary
        """
        
        source_image = self.image_processor.load_image(source_image_path)
        target_image = self.image_processor.load_image(target_image_path)
        
        
        result, metadata = self.swap_faces(
            source_image, target_image,
            source_face_idx, target_face_idx,
            blend_mode, blend_factor
        )
        
        
        self.image_processor.save_image(result, output_path)
        metadata["output_path"] = output_path
        
        return metadata

