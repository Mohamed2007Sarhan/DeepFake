"""
Enhanced Clothing Removal / Nudify Module
Realistic body-aware clothing removal with gender and body type consideration
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from pathlib import Path
import os
from .utils.image_processor import ImageProcessor
from .body_estimator import BodySizeEstimator
from .color_detector import ClothesColorDetector
from .utils.gender_detector import GenderDetector
from .utils.skin_generator import RealisticSkinGenerator


class ClothingRemover:
    """
    Advanced clothing removal system with realistic body-aware processing
    Considers gender, body type, and measurements for authentic results
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: str = "cpu",
                 confidence_threshold: float = 0.7,
                 realistic_mode: bool = True):
        """
        Initialize enhanced clothing remover
        
        Args:
            model_path: Path to ONNX model (if available)
            device: "cuda" or "cpu"
            confidence_threshold: Confidence threshold for clothing detection
            realistic_mode: Enable realistic skin generation
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.realistic_mode = realistic_mode
        self.image_processor = ImageProcessor()
        self.body_estimator = BodySizeEstimator()
        self.color_detector = ClothesColorDetector()
        self.gender_detector = GenderDetector()
        self.skin_generator = RealisticSkinGenerator()
        self.model_loaded = False
        self.model_path = model_path
        
        # Try to load model if path provided
        if model_path and os.path.exists(model_path):
            try:
                self._load_model(model_path)
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
                print("Falling back to enhanced inpainting method")
    
    def _load_model(self, model_path: str):
        """
        Load ONNX model for advanced clothing removal
        
        Args:
            model_path: Path to ONNX model file
        """
        try:
            import onnxruntime as ort
            
            # Create inference session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
            
            self.ort_session = ort.InferenceSession(
                model_path,
                providers=providers
            )
            
            # Get input/output names
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            
            self.model_loaded = True
            print(f"✓ Advanced model loaded: {model_path}")
            
        except ImportError:
            print("Warning: onnxruntime not available. Install with: pip install onnxruntime")
            print("Falling back to traditional inpainting method")
            self.model_loaded = False
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Falling back to traditional inpainting method")
            self.model_loaded = False
    
    def detect_clothing_mask(self, image: np.ndarray, 
                             body_mask: Optional[np.ndarray] = None,
                             pose_data: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced clothing region detection with pose awareness
        
        Args:
            image: Input image (RGB)
            body_mask: Optional body segmentation mask
            pose_data: Optional pose landmarks for region-specific detection
            
        Returns:
            Tuple of (clothing_mask, skin_mask) binary masks
        """
        h, w = image.shape[:2]
        clothing_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Enhanced skin detection with multiple ranges
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extended skin color ranges (covers various ethnicities)
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # Additional ranges for darker skin tones
        lower_skin3 = np.array([0, 50, 50], dtype=np.uint8)
        upper_skin3 = np.array([30, 200, 255], dtype=np.uint8)
        
        skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask3 = cv2.inRange(hsv, lower_skin3, upper_skin3)
        skin_mask = cv2.bitwise_or(cv2.bitwise_or(skin_mask1, skin_mask2), skin_mask3)
        
        # Use pose data for region-specific detection
        if pose_data and "landmarks" in pose_data:
            landmarks = pose_data["landmarks"]
            # Define body regions for better clothing detection
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_HIP = 23
            RIGHT_HIP = 24
            
            if all(k in landmarks for k in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]):
                # Upper body region (torso)
                shoulder_y = min(landmarks[LEFT_SHOULDER]["y"], landmarks[RIGHT_SHOULDER]["y"])
                hip_y = max(landmarks[LEFT_HIP]["y"], landmarks[RIGHT_HIP]["y"])
                
                # Create region mask for torso
                torso_mask = np.zeros((h, w), dtype=np.uint8)
                torso_mask[int(shoulder_y):int(hip_y), :] = 255
                skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(torso_mask))
        
        # Method 1: Use body mask to identify non-skin regions
        if body_mask is not None:
            if body_mask.dtype != np.uint8:
                body_mask = (body_mask * 255).astype(np.uint8)
            
            # Clothing is body region minus skin
            clothing_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(skin_mask))
        else:
            # Method 2: Color-based clothing detection
            # Clothing regions (not skin, not too dark/light)
            clothing_mask = cv2.bitwise_not(skin_mask)
            
            # Remove very dark and very light regions (likely not clothing)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, dark_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
            _, light_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            
            clothing_mask = cv2.bitwise_and(clothing_mask, cv2.bitwise_not(dark_mask))
            clothing_mask = cv2.bitwise_and(clothing_mask, cv2.bitwise_not(light_mask))
        
        # Enhanced morphological operations
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        
        clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel_large)
        clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Remove small noise
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clothing_mask)
        min_area = (h * w) * 0.005  # At least 0.5% of image
        
        cleaned_mask = np.zeros_like(clothing_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = 255
        
        # Smooth mask edges
        cleaned_mask = cv2.GaussianBlur(cleaned_mask, (5, 5), 0)
        _, cleaned_mask = cv2.threshold(cleaned_mask, 127, 255, cv2.THRESH_BINARY)
        
        return cleaned_mask, skin_mask
    
    def generate_skin_tone(self, image: np.ndarray, 
                          skin_regions: np.ndarray,
                          gender: str = "unknown") -> np.ndarray:
        """
        Enhanced skin tone estimation with gender awareness
        
        Args:
            image: Input image
            skin_regions: Binary mask of skin regions
            gender: Detected gender for tone adjustment
            
        Returns:
            Estimated skin color (RGB)
        """
        if skin_regions.sum() == 0:
            # Default skin tone if no skin detected
            default_tone = np.array([220, 180, 150], dtype=np.uint8)
            if gender == "male":
                # Slightly more olive
                default_tone = np.array([210, 175, 145], dtype=np.uint8)
            elif gender == "female":
                # Slightly more rosy
                default_tone = np.array([225, 185, 155], dtype=np.uint8)
            return default_tone
        
        # Extract skin pixels
        skin_pixels = image[skin_regions > 0]
        
        if len(skin_pixels) == 0:
            return np.array([220, 180, 150], dtype=np.uint8)
        
        # Use percentile for more robust estimation (less affected by outliers)
        skin_tone = np.percentile(skin_pixels, 50, axis=0).astype(np.uint8)
        
        # Gender-specific adjustments
        if gender == "male":
            # Slightly more yellow/olive
            skin_tone[0] = min(255, int(skin_tone[0] * 0.98))  # Less red
            skin_tone[1] = min(255, int(skin_tone[1] * 1.02))  # More green
        elif gender == "female":
            # Slightly more pink
            skin_tone[0] = min(255, int(skin_tone[0] * 1.02))  # More red
            skin_tone[2] = min(255, int(skin_tone[2] * 1.01))  # Slightly more blue
        
        return skin_tone
    
    def inpaint_clothing(self, image: np.ndarray, 
                        clothing_mask: np.ndarray,
                        method: str = "telea",
                        iterations: int = 3) -> np.ndarray:
        """
        Enhanced inpainting with multiple iterations
        
        Args:
            image: Input image
            clothing_mask: Binary mask of clothing to remove
            method: "telea" or "ns" (Navier-Stokes)
            iterations: Number of inpainting iterations
            
        Returns:
            Inpainted image
        """
        # Convert to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Apply inpainting with iterations
        result = image_bgr.copy()
        for _ in range(iterations):
            if method == "telea":
                result = cv2.inpaint(result, clothing_mask, 3, cv2.INPAINT_TELEA)
            else:
                result = cv2.inpaint(result, clothing_mask, 3, cv2.INPAINT_NS)
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result_rgb
    
    def remove_clothing_realistic(self, image: np.ndarray,
                                  clothing_mask: np.ndarray,
                                  body_mask: np.ndarray,
                                  skin_tone: np.ndarray,
                                  gender: str,
                                  body_type: str,
                                  body_measurements: Dict) -> np.ndarray:
        """
        Realistic clothing removal using advanced skin generation
        
        Args:
            image: Input image
            clothing_mask: Clothing regions mask
            body_mask: Body segmentation mask
            skin_tone: Estimated skin color
            gender: Detected gender
            body_type: Body type classification
            body_measurements: Body measurements dict
            
        Returns:
            Realistic processed image
        """
        h, w = image.shape[:2]
        
        # Step 1: Initial inpainting to remove clothing structure
        inpainted = self.inpaint_clothing(image, clothing_mask, method="telea", iterations=2)
        
        # Step 2: Generate realistic skin for clothing regions
        if self.realistic_mode:
            # Generate realistic skin texture
            skin_texture = self.skin_generator.generate_realistic_skin(
                base_tone=skin_tone,
                region_shape=(h, w),
                mask=body_mask,
                body_type=body_type,
                gender=gender,
                add_texture=True,
                add_shadows=True
            )
            
            # Apply skin texture to clothing regions
            clothing_mask_normalized = (clothing_mask / 255.0)[:, :, np.newaxis]
            
            # Blend skin texture with inpainted result
            result = (skin_texture * clothing_mask_normalized + 
                     inpainted * (1 - clothing_mask_normalized))
        else:
            # Simple skin tone application
            skin_tone_img = np.ones_like(image) * skin_tone
            clothing_mask_normalized = (clothing_mask / 255.0)[:, :, np.newaxis]
            result = (skin_tone_img * clothing_mask_normalized + 
                     inpainted * (1 - clothing_mask_normalized))
        
        # Step 3: Seamless blending with surrounding areas
        feathered_mask = cv2.GaussianBlur(clothing_mask, (25, 25), 0) / 255.0
        feathered_mask_3d = feathered_mask[:, :, np.newaxis]
        
        result = (result.astype(np.float32) * feathered_mask_3d + 
                 image.astype(np.float32) * (1 - feathered_mask_3d))
        
        # Step 4: Final refinement - blend with original for natural transition
        result = (result * 0.85 + inpainted.astype(np.float32) * 0.15)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _remove_with_ai_model(self, image: np.ndarray,
                              clothing_mask: np.ndarray,
                              body_mask: np.ndarray,
                              skin_tone: np.ndarray) -> np.ndarray:
        """
        Remove clothing using advanced AI model
        
        Args:
            image: Input image
            clothing_mask: Clothing mask
            body_mask: Body mask
            skin_tone: Estimated skin tone
            
        Returns:
            Processed image
        """
        try:
            import onnxruntime as ort
            
            # Preprocess image for model
            # Resize to model input size (typically 512x512)
            h, w = image.shape[:2]
            model_input_size = 512
            
            # Resize image and mask
            image_resized = cv2.resize(image, (model_input_size, model_input_size))
            mask_resized = cv2.resize(clothing_mask, (model_input_size, model_input_size))
            
            # Normalize image
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Prepare input (format: [batch, channels, height, width])
            input_data = np.expand_dims(
                np.transpose(image_normalized, (2, 0, 1)),
                axis=0
            ).astype(np.float32)
            
            # Run inference
            outputs = self.ort_session.run(
                [self.output_name],
                {self.input_name: input_data}
            )
            
            # Get output
            output = outputs[0][0]  # Remove batch dimension
            
            # Post-process output (format: [channels, height, width])
            result = np.transpose(output, (1, 2, 0))
            result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            
            # Resize back to original size
            result = cv2.resize(result, (w, h))
            
            # Apply mask to blend with original
            mask_normalized = (clothing_mask / 255.0)[:, :, np.newaxis]
            result = (result.astype(np.float32) * mask_normalized +
                     image.astype(np.float32) * (1 - mask_normalized))
            
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"Warning: AI model inference failed: {e}")
            print("Falling back to traditional method")
            # Fallback to traditional inpainting
            return self.inpaint_clothing(image, clothing_mask, method="telea")
    
    def remove_clothing(self, image_path: str,
                       output_path: Optional[str] = None,
                       use_advanced: bool = False,
                       blend_factor: float = 0.8,
                       gender: Optional[str] = None,
                       body_type: Optional[str] = None) -> np.ndarray:
        """
        Enhanced main method to remove clothing with body awareness
        
        Args:
            image_path: Path to input image
            output_path: Optional output path to save result
            use_advanced: Use advanced AI model if available
            blend_factor: Blending factor for skin tone (0-1)
            gender: Optional manual gender override
            body_type: Optional manual body type override
            
        Returns:
            Image with clothing removed (RGB)
        """
        # Load image
        image = self.image_processor.load_image(image_path)
        
        # Get body information
        body_info = self.body_estimator.estimate_body_size(image_path)
        pose_data = self.body_estimator.detect_pose_landmarks(image)
        
        # Detect gender if not provided
        if gender is None:
            gender_result = self.gender_detector.detect_gender(image, pose_data)
            gender = gender_result.get("gender", "unknown")
        
        # Get body type if not provided
        if body_type is None:
            measurements = body_info.get("measurements", {})
            body_type = measurements.get("body_type", "Rectangle")
        
        # Segment body
        body_mask = self.body_estimator.segment_body(image)
        if body_mask.dtype != np.uint8:
            body_mask = (body_mask * 255).astype(np.uint8)
        
        # Detect clothing and skin regions
        clothing_mask, skin_mask = self.detect_clothing_mask(image, body_mask, pose_data)
        
        if clothing_mask.sum() == 0:
            print("Warning: No clothing detected in image")
            return image
        
        # Estimate skin tone with gender awareness
        skin_tone = self.generate_skin_tone(image, skin_mask, gender)
        
        # Get body measurements
        body_measurements = body_info.get("measurements", {})
        
        if use_advanced and self.model_loaded:
            # Use advanced AI model
            result = self._remove_with_ai_model(
                image, clothing_mask, body_mask, skin_tone
            )
        else:
            # Enhanced realistic removal
            result = self.remove_clothing_realistic(
                image=image,
                clothing_mask=clothing_mask,
                body_mask=body_mask,
                skin_tone=skin_tone,
                gender=gender,
                body_type=body_type,
                body_measurements=body_measurements
            )
        
        # Save if output path provided
        if output_path:
            self.image_processor.save_image(result, output_path)
        
        return result
    
    def process_with_metadata(self, image_path: str) -> Dict:
        """
        Process image and return comprehensive metadata
        
        Returns:
            Dictionary with processed image and full metadata
        """
        # Load image
        image = self.image_processor.load_image(image_path)
        
        # Get body size and pose
        body_info = self.body_estimator.estimate_body_size(image_path)
        pose_data = self.body_estimator.detect_pose_landmarks(image)
        
        # Detect gender
        gender_info = self.gender_detector.detect_gender(image, pose_data)
        
        # Get clothes color
        body_mask = self.body_estimator.segment_body(image)
        color_info = self.color_detector.extract_clothes_colors(image, body_mask)
        
        # Remove clothing
        gender = gender_info.get("gender", "unknown")
        body_type = body_info.get("measurements", {}).get("body_type", "Rectangle")
        
        result_image = self.remove_clothing(
            image_path,
            gender=gender,
            body_type=body_type
        )
        
        return {
            "processed_image": result_image,
            "body_size": body_info,
            "clothes_color": color_info,
            "gender": gender_info,
            "original_path": image_path,
            "body_type": body_type
        }
