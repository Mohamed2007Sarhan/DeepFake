utf-8"""
Color Utility Functions
"""

import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from typing import List, Tuple, Dict


class ColorUtils:
    """Utilities for color analysis and manipulation"""
    
    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex color"""
        return '#%02x%02x%02x' % tuple(rgb)
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def get_dominant_colors_kmeans(image: np.ndarray, n_colors: int = 5) -> List[Tuple[int, int, int, float]]:
        """
        Extract dominant colors using K-Means clustering
        Returns: List of (R, G, B, percentage) tuples
        """
        
        pixels = image.reshape(-1, 3)
        
        
        if pixels.shape[1] == 4:
            pixels = pixels[:, :3]
        
        
        pixels = pixels[np.sum(pixels, axis=1) > 0]
        
        if len(pixels) == 0:
            return []
        
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        
        label_counts = Counter(labels)
        total_pixels = len(labels)
        
        result = []
        for i, color in enumerate(colors):
            percentage = (label_counts[i] / total_pixels) * 100
            result.append((int(color[0]), int(color[1]), int(color[2]), percentage))
        
        
        result.sort(key=lambda x: x[3], reverse=True)
        return result
    
    @staticmethod
    def get_dominant_color_simple(image: np.ndarray) -> Tuple[int, int, int]:
        """Get single dominant color using histogram"""
        pixels = image.reshape(-1, 3)
        
        
        mean_color = np.mean(pixels, axis=0).astype(int)
        return tuple(mean_color)
    
    @staticmethod
    def detect_color_range(image: np.ndarray, target_color: Tuple[int, int, int], 
                          threshold: int = 30) -> np.ndarray:
        """
        Detect pixels within a color range
        Returns binary mask
        """
        target = np.array(target_color)
        diff = np.abs(image.astype(int) - target)
        mask = np.sum(diff, axis=2) < threshold
        return mask.astype(np.uint8) * 255
    
    @staticmethod
    def classify_color_name(rgb: Tuple[int, int, int]) -> str:
        """Classify RGB color into a named color"""
        r, g, b = rgb
        
        
        if r > 200 and g > 200 and b > 200:
            return "White"
        elif r < 50 and g < 50 and b < 50:
            return "Black"
        elif r > 200 and g < 100 and b < 100:
            return "Red"
        elif r < 100 and g > 200 and b < 100:
            return "Green"
        elif r < 100 and g < 100 and b > 200:
            return "Blue"
        elif r > 200 and g > 150 and b < 100:
            return "Yellow"
        elif r > 180 and g > 100 and b < 100:
            return "Orange"
        elif r > 150 and g < 100 and b > 150:
            return "Purple"
        elif r > 150 and g > 100 and b < 100:
            return "Pink"
        elif abs(r - g) < 30 and abs(g - b) < 30:
            return "Gray"
        else:
            return "Brown"

