utf-8"""
Clothes Color Detection Module
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from .utils.image_processor import ImageProcessor
from .utils.color_utils import ColorUtils


class ClothesColorDetector:
    """
    Advanced clothes color detection system
    Detects and extracts color information from clothing in images
    """
    
    def __init__(self, algorithm: str = "kmeans", num_colors: int = 5):
        """
        Initialize color detector
        
        Args:
            algorithm: "kmeans" or "dominant_color"
            num_colors: Number of dominant colors to extract
        """
        self.algorithm = algorithm
        self.num_colors = num_colors
        self.image_processor = ImageProcessor()
        self.color_utils = ColorUtils()
    
    def detect_clothing_region(self, image: np.ndarray, 
                               use_body_detection: bool = True) -> np.ndarray:
        """
        Detect clothing region in image
        Returns binary mask of clothing region
        """
        if not use_body_detection:
            
            return np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        
        
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        
        clothing_mask = cv2.bitwise_not(skin_mask)
        
        
        kernel = np.ones((5, 5), np.uint8)
        clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel)
        clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel)
        
        return clothing_mask
    
    def extract_clothes_colors(self, image: np.ndarray, 
                               clothing_mask: Optional[np.ndarray] = None,
                               region: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Extract color information from clothing
        
        Args:
            image: Input image (RGB)
            clothing_mask: Optional binary mask for clothing region
            region: Optional bounding box (x, y, w, h) for specific region
            
        Returns:
            Dictionary with color information
        """
        
        if region:
            x, y, w, h = region
            image_region = image[y:y+h, x:x+w]
        else:
            image_region = image.copy()
        
        
        if clothing_mask is not None:
            if region:
                mask_region = clothing_mask[y:y+h, x:x+w]
            else:
                mask_region = clothing_mask
            
            
            masked_image = image_region.copy()
            masked_image[mask_region == 0] = [0, 0, 0]
        else:
            masked_image = image_region
        
        
        if self.algorithm == "kmeans":
            dominant_colors = self.color_utils.get_dominant_colors_kmeans(
                masked_image, self.num_colors
            )
        else:
            
            dominant_color = self.color_utils.get_dominant_color_simple(masked_image)
            dominant_colors = [(dominant_color[0], dominant_color[1], dominant_color[2], 100.0)]
        
        
        colors_info = []
        for r, g, b, percentage in dominant_colors:
            hex_color = self.color_utils.rgb_to_hex((r, g, b))
            color_name = self.color_utils.classify_color_name((r, g, b))
            
            colors_info.append({
                "rgb": (r, g, b),
                "hex": hex_color,
                "name": color_name,
                "percentage": round(percentage, 2)
            })
        
        primary_color = colors_info[0] if colors_info else None
        
        return {
            "primary_color": primary_color,
            "dominant_colors": colors_info,
            "total_colors": len(colors_info),
            "algorithm": self.algorithm
        }
    
    def get_clothes_color(self, image_path: str, 
                         clothing_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Main method to get clothes color from image path
        
        Args:
            image_path: Path to input image
            clothing_mask: Optional pre-computed clothing mask
            
        Returns:
            Dictionary with color information
        """
        
        image = self.image_processor.load_image(image_path)
        
        
        if clothing_mask is None:
            clothing_mask = self.detect_clothing_region(image)
        
        
        colors = self.extract_clothes_colors(image, clothing_mask)
        
        return colors
    
    def analyze_color_distribution(self, image: np.ndarray, 
                                  clothing_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze detailed color distribution in clothing
        
        Returns:
            Detailed color statistics
        """
        if clothing_mask is not None:
            masked_image = image.copy()
            masked_image[clothing_mask == 0] = [0, 0, 0]
        else:
            masked_image = image
        
        pixels = masked_image.reshape(-1, 3)
        pixels = pixels[np.sum(pixels, axis=1) > 0]  
        
        if len(pixels) == 0:
            return {"error": "No valid pixels found"}
        
        
        mean_color = np.mean(pixels, axis=0).astype(int)
        std_color = np.std(pixels, axis=0).astype(int)
        median_color = np.median(pixels, axis=0).astype(int)
        
        return {
            "mean_rgb": tuple(mean_color),
            "std_rgb": tuple(std_color),
            "median_rgb": tuple(median_color),
            "total_pixels": len(pixels),
            "color_variance": float(np.mean(std_color))
        }

