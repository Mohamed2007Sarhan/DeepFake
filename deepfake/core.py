utf-8"""
DeepFake Framework Core Module
Main API for the complete framework
"""

import yaml
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np

from .color_detector import ClothesColorDetector
from .body_estimator import BodySizeEstimator
from .nudifier import ClothingRemover
from .face_swapper import FaceSwapper
from .voice_cloner import VoiceCloner
from .utils.image_processor import ImageProcessor


class DeepFakeFramework:
    """
    Complete DeepFake Framework
    Unified interface for all deepfake operations
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize framework
        
        Args:
            config_path: Path to config.yaml file
            **kwargs: Override config values
        """
        
        self.config = self._load_config(config_path, **kwargs)
        
        
        self.color_detector = ClothesColorDetector(
            algorithm=self.config["color_detection"]["algorithm"],
            num_colors=self.config["color_detection"]["num_colors"]
        )
        
        self.body_estimator = BodySizeEstimator(
            use_mediapipe=self.config["body_size"]["use_mediapipe"],
            reference_height=self.config["body_size"]["reference_height"]
        )
        
        self.clothing_remover = ClothingRemover(
            model_path=self.config["clothing_removal"].get("model_path"),
            device=self.config["model"]["device"],
            confidence_threshold=self.config["clothing_removal"]["confidence_threshold"],
            realistic_mode=self.config["clothing_removal"].get("realistic_mode", True)
        )
        
        
        face_swap_config = self.config.get("face_swap", {})
        self.face_swapper = FaceSwapper(
            method=face_swap_config.get("method", "dlib"),
            model_path=face_swap_config.get("model_path")
        )
        
        
        voice_config = self.config.get("voice_cloning", {})
        self.voice_cloner = VoiceCloner(
            method=voice_config.get("method", "auto"),
            device=self.config["model"]["device"]
        )
        
        self.image_processor = ImageProcessor()
    
    def _load_config(self, config_path: Optional[str], **kwargs) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "model": {"device": "cpu", "batch_size": 1},
            "color_detection": {"algorithm": "kmeans", "num_colors": 5, "color_space": "RGB"},
            "body_size": {"use_mediapipe": True, "estimate_measurements": True, "reference_height": 170.0},
            "clothing_removal": {
                "model_path": None,
                "confidence_threshold": 0.7,
                "blend_factor": 0.8,
                "skin_tone_preservation": True
            },
            "face_swap": {
                "method": "dlib",
                "model_path": None,
                "blend_mode": "seamless",
                "blend_factor": 0.8
            },
            "voice_cloning": {
                "method": "auto",  
                "adjust_to_profile": True,
                "sample_rate": 22050
            },
            "output": {"save_intermediate": True, "output_format": "png", "quality": 95}
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
        
        
        for key, value in kwargs.items():
            if key in default_config:
                if isinstance(value, dict) and isinstance(default_config[key], dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        
        return default_config
    
    def get_clothes_color(self, image_path: str) -> Dict:
        """
        Extract clothes color information
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with color information
        """
        return self.color_detector.get_clothes_color(image_path)
    
    def get_body_size(self, image_path: str) -> Dict:
        """
        Estimate body size and measurements
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with body size information
        """
        return self.body_estimator.estimate_body_size(image_path)
    
    def remove_clothes(self, image_path: str, 
                      output_path: Optional[str] = None,
                      **kwargs) -> np.ndarray:
        """
        Remove clothing from image
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save output
            **kwargs: Additional parameters for removal
            
        Returns:
            Processed image (RGB numpy array)
        """
        return self.clothing_remover.remove_clothing(
            image_path,
            output_path=output_path,
            use_advanced=kwargs.get("use_advanced", False),
            blend_factor=kwargs.get("blend_factor", self.config["clothing_removal"]["blend_factor"]),
            gender=kwargs.get("gender"),
            body_type=kwargs.get("body_type")
        )
    
    def process_complete(self, image_path: str,
                        output_path: Optional[str] = None,
                        return_metadata: bool = True) -> Dict:
        """
        Complete processing pipeline: color detection, body size, clothing removal
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save processed image
            return_metadata: Whether to include metadata in return
            
        Returns:
            Dictionary with processed image and metadata
        """
        results = {}
        
        
        print("Analyzing clothes color...")
        results["clothes_color"] = self.get_clothes_color(image_path)
        
        
        print("Estimating body size...")
        results["body_size"] = self.get_body_size(image_path)
        
        
        print("Removing clothing...")
        processed_image = self.remove_clothes(image_path, output_path=output_path)
        results["processed_image"] = processed_image
        
        if output_path:
            results["output_path"] = output_path
        
        return results
    
    def batch_process(self, image_paths: list,
                     output_dir: str,
                     return_metadata: bool = False) -> list:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory to save outputs
            return_metadata: Whether to return metadata for each image
            
        Returns:
            List of results (dicts with processed images and optionally metadata)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            output_path = str(Path(output_dir) / f"processed_{i+1}.png")
            
            if return_metadata:
                result = self.process_complete(image_path, output_path)
            else:
                processed = self.remove_clothes(image_path, output_path)
                result = {"processed_image": processed, "output_path": output_path}
            
            results.append(result)
        
        return results
    
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
            blend_mode: Blending mode ("seamless", "linear", "feathered")
            blend_factor: Blending factor (0-1)
            
        Returns:
            Metadata dictionary
        """
        return self.face_swapper.swap_faces_image(
            source_image_path, target_image_path, output_path,
            source_face_idx, target_face_idx, blend_mode, blend_factor
        )
    
    def swap_face_to_video(self, source_image_path: str,
                          target_video_path: str,
                          output_video_path: str,
                          source_face_idx: int = 0,
                          blend_mode: str = "seamless") -> Dict:
        """
        Swap face from image to video
        
        Args:
            source_image_path: Path to source image
            target_video_path: Path to target video
            output_video_path: Path to save output video
            source_face_idx: Index of source face
            blend_mode: Blending mode
            
        Returns:
            Metadata dictionary
        """
        return self.face_swapper.swap_face_to_video(
            source_image_path, target_video_path, output_video_path,
            source_face_idx, blend_mode
        )
    
    def clone_voice(self, voice_sample_path: str,
                   text: str,
                   output_path: str) -> Dict:
        """
        Clone voice from sample and synthesize text
        
        Args:
            voice_sample_path: Path to voice sample audio
            text: Text to synthesize
            output_path: Path to save output audio
            
        Returns:
            Dictionary with results
        """
        return self.voice_cloner.clone_voice_from_sample(
            voice_sample_path, text, output_path
        )
    
    def analyze_voice(self, audio_path: str) -> Dict:
        """
        Analyze voice sample and extract characteristics
        
        Args:
            audio_path: Path to voice sample
            
        Returns:
            Voice profile dictionary
        """
        return self.voice_cloner.analyze_voice(audio_path)
    
    def synthesize_speech(self, text: str,
                         output_path: str,
                         voice_sample_path: Optional[str] = None) -> Dict:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            output_path: Path to save output
            voice_sample_path: Optional voice sample to clone
            
        Returns:
            Synthesis result dictionary
        """
        if voice_sample_path:
            
            self.voice_cloner.analyze_voice(voice_sample_path)
        
        return self.voice_cloner.synthesize_speech(
            text=text,
            output_path=output_path,
            adjust_to_profile=True
        )
    
    def get_framework_info(self) -> Dict:
        """Get framework information and capabilities"""
        return {
            "version": "2.1.0",
            "components": {
                "color_detector": "Active",
                "body_estimator": "Active" if self.config["body_size"]["use_mediapipe"] else "Basic",
                "clothing_remover": "Advanced" if self.clothing_remover.model_loaded else "Traditional",
                "face_swapper": "Active",
                "voice_cloner": "Active"
            },
            "config": self.config
        }

