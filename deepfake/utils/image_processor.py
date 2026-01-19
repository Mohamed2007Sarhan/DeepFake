utf-8"""
Image Processing Utilities
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union


class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load image from path"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str, quality: int = 95):
        """Save image to path"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
            
        cv2.imwrite(output_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)
    
    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """Denormalize image from [0, 1] to [0, 255]"""
        if image.max() <= 1.0:
            return (image * 255).astype(np.uint8)
        return image.astype(np.uint8)
    
    @staticmethod
    def extract_region(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract region from image using bounding box (x, y, w, h)"""
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]
    
    @staticmethod
    def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend two images"""
        return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

