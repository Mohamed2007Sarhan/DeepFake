utf-8"""
Realistic Skin Texture Generator
Body-aware skin generation with texture synthesis
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    
    def gaussian_filter(image, sigma):
        """Fallback gaussian filter using OpenCV"""
        kernel_size = int(sigma * 6) // 2 * 2 + 1  
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


class RealisticSkinGenerator:
    """
    Generate realistic skin textures based on body type, gender, and measurements
    """
    
    def __init__(self):
        """Initialize skin generator"""
        pass
    
    def generate_skin_tone_gradient(self, base_tone: np.ndarray, 
                                   region_shape: Tuple[int, int],
                                   body_type: str = "Rectangle",
                                   gender: str = "unknown") -> np.ndarray:
        """
        Generate skin tone with natural gradients and variations
        
        Args:
            base_tone: Base skin color (RGB)
            region_shape: (height, width) of region
            body_type: Body type for tone variation
            gender: Gender for tone adjustment
            
        Returns:
            Image with realistic skin tone gradient
        """
        h, w = region_shape
        skin_image = np.ones((h, w, 3), dtype=np.float32) * base_tone.astype(np.float32)
        
        
        
        noise = np.random.randn(h, w, 3) * 8.0
        skin_image = np.clip(skin_image + noise, 0, 255)
        
        
        y_gradient = np.linspace(0, 1, h)[:, np.newaxis]
        x_gradient = np.linspace(0, 1, w)[np.newaxis, :]
        
        
        center_y, center_x = h // 2, w // 2
        dist_from_center = np.sqrt((np.arange(h)[:, np.newaxis] - center_y)**2 + 
                                  (np.arange(w)[np.newaxis, :] - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        center_lighting = 1.0 - (dist_from_center / max_dist) * 0.15
        
        
        for c in range(3):
            skin_image[:, :, c] *= center_lighting
        
        
        if gender == "male":
            
            skin_image[:, :, 0] *= 1.02  
            skin_image[:, :, 1] *= 0.98  
        elif gender == "female":
            
            skin_image[:, :, 0] *= 1.03  
            skin_image[:, :, 2] *= 1.01  
        
        
        if body_type in ["Pear", "Hourglass"]:
            
            skin_image = gaussian_filter(skin_image, sigma=1.0)
        elif body_type == "Inverted Triangle":
            
            skin_image = gaussian_filter(skin_image, sigma=0.5)
        
        return np.clip(skin_image, 0, 255).astype(np.uint8)
    
    def add_skin_texture(self, skin_image: np.ndarray,
                        intensity: float = 0.3) -> np.ndarray:
        """
        Add realistic skin texture (pores, fine details)
        
        Args:
            skin_image: Base skin image
            intensity: Texture intensity (0-1)
            
        Returns:
            Image with added texture
        """
        h, w = skin_image.shape[:2]
        
        
        
        pore_noise = np.random.randn(h, w) * 2.0
        
        
        variation_noise = np.random.randn(h, w) * 4.0
        variation_noise = gaussian_filter(variation_noise, sigma=2.0)
        
        
        texture_noise = np.random.randn(h, w) * 1.5
        texture_noise = gaussian_filter(texture_noise, sigma=5.0)
        
        
        combined_texture = (pore_noise + variation_noise * 0.7 + texture_noise * 0.4) * intensity
        
        
        result = skin_image.copy().astype(np.float32)
        for c in range(3):
            channel_variation = combined_texture * (1.0 + (c - 1) * 0.1)
            result[:, :, c] = np.clip(result[:, :, c] + channel_variation, 0, 255)
        
        return result.astype(np.uint8)
    
    def add_body_shadows(self, skin_image: np.ndarray,
                        mask: np.ndarray,
                        body_type: str = "Rectangle") -> np.ndarray:
        """
        Add natural shadows and highlights based on body shape
        
        Args:
            skin_image: Base skin image
            mask: Body region mask
            body_type: Body type for shadow patterns
            
        Returns:
            Image with added shadows
        """
        h, w = skin_image.shape[:2]
        result = skin_image.copy().astype(np.float32)
        
        
        shadow_map = np.ones((h, w), dtype=np.float32)
        
        
        x_gradient = np.linspace(-1, 1, w)[np.newaxis, :]
        edge_shadows = np.abs(x_gradient) * 0.1  
        
        
        y_gradient = np.linspace(0, 1, h)[:, np.newaxis]
        vertical_shadows = y_gradient * 0.05  
        
        
        shadow_map -= (edge_shadows + vertical_shadows)
        
        
        if body_type == "Pear":
            
            shadow_map[:h//2, :] *= 0.95
        elif body_type == "Inverted Triangle":
            
            shadow_map *= 0.97
        
        
        mask_normalized = (mask / 255.0)[:, :, np.newaxis]
        shadow_map_3d = shadow_map[:, :, np.newaxis]
        
        result = result * shadow_map_3d * mask_normalized + result * (1 - mask_normalized)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def generate_realistic_skin(self, base_tone: np.ndarray,
                               region_shape: Tuple[int, int],
                               mask: np.ndarray,
                               body_type: str = "Rectangle",
                               gender: str = "unknown",
                               add_texture: bool = True,
                               add_shadows: bool = True) -> np.ndarray:
        """
        Generate complete realistic skin with all features
        
        Args:
            base_tone: Base skin color (RGB)
            region_shape: (height, width) of region
            mask: Body region mask
            body_type: Body type classification
            gender: Gender classification
            add_texture: Whether to add skin texture
            add_shadows: Whether to add shadows
            
        Returns:
            Complete realistic skin image
        """
        
        skin = self.generate_skin_tone_gradient(
            base_tone, region_shape, body_type, gender
        )
        
        
        if add_texture:
            skin = self.add_skin_texture(skin, intensity=0.25)
        
        
        if add_shadows:
            skin = self.add_body_shadows(skin, mask, body_type)
        
        
        mask_3d = (mask / 255.0)[:, :, np.newaxis]
        skin = skin * mask_3d
        
        return skin.astype(np.uint8)
    
    def blend_with_surrounding(self, skin_image: np.ndarray,
                              original_image: np.ndarray,
                              mask: np.ndarray,
                              blend_size: int = 15) -> np.ndarray:
        """
        Blend generated skin seamlessly with surrounding image
        
        Args:
            skin_image: Generated skin image
            original_image: Original image
            mask: Blending mask
            blend_size: Size of blending feather
            
        Returns:
            Seamlessly blended image
        """
        
        mask_float = mask.astype(np.float32) / 255.0
        feathered_mask = cv2.GaussianBlur(mask_float, (blend_size * 2 + 1, blend_size * 2 + 1), 0)
        feathered_mask_3d = feathered_mask[:, :, np.newaxis]
        
        
        result = (skin_image * feathered_mask_3d + 
                 original_image * (1 - feathered_mask_3d))
        
        return result.astype(np.uint8)

