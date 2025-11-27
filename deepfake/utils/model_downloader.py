"""
Model Downloader for Clothing Removal
Downloads and manages AI models for advanced clothing removal
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Dict
import hashlib


class ModelDownloader:
    """
    Downloads and manages AI models for clothing removal
    """
    
    # Best model templates available (prioritized)
    MODELS = {
        "nudify_advanced": {
            "name": "Nudify Advanced Model",
            "description": "Best quality model for clothing removal",
            "url": "https://github.com/ErwannMillon/PhotoVerse/releases/download/v1.0/model.zip",  # Placeholder
            "filename": "nudify_model.onnx",
            "size_mb": 150,
            "recommended": True
        },
        "segment_anything": {
            "name": "Segment Anything Model",
            "description": "High-quality segmentation for clothing detection",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "filename": "sam_model.pth",
            "size_mb": 2400,
            "recommended": False
        }
    }
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model downloader
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def get_recommended_model(self) -> Dict:
        """Get the recommended model configuration"""
        for model_id, model_info in self.MODELS.items():
            if model_info.get("recommended", False):
                return {**model_info, "id": model_id}
        
        # Fallback to first model
        first_model = list(self.MODELS.items())[0]
        return {**first_model[1], "id": first_model[0]}
    
    def download_model(self, model_id: Optional[str] = None, 
                      show_progress: bool = True) -> Optional[str]:
        """
        Download model file or provide instructions
        
        Args:
            model_id: Model ID to download (uses recommended if None)
            show_progress: Show download progress
            
        Returns:
            Path to downloaded model file, or None if manual download needed
        """
        if model_id is None:
            model_info = self.get_recommended_model()
            model_id = model_info["id"]
        else:
            model_info = self.MODELS.get(model_id)
            if not model_info:
                print(f"Error: Unknown model ID: {model_id}")
                return None
        
        model_path = self.models_dir / model_info["filename"]
        
        # Check if already exists
        if model_path.exists():
            print(f"✓ Model already exists: {model_path}")
            return str(model_path)
        
        # Check if URL is available
        if model_info.get("url") is None:
            print(f"\n{'='*60}")
            print(f"Model: {model_info['name']}")
            print(f"{'='*60}")
            print(f"\nDescription: {model_info['description']}")
            print(f"Filename: {model_info['filename']}")
            print(f"Size: {model_info['size_mb']} MB (approximate)")
            
            if model_info.get("instructions"):
                print(f"\n{model_info['instructions']}")
            
            print(f"\n{'='*60}")
            print("Manual Download Instructions:")
            print(f"{'='*60}")
            print(f"\n1. Download an ONNX model for clothing removal/nudify")
            print(f"   - Search on HuggingFace: https://huggingface.co/models")
            print(f"   - Look for: 'nudify', 'clothing-removal', or 'inpaint' models")
            print(f"   - Format must be: ONNX (.onnx)")
            print(f"\n2. Place the downloaded file here:")
            print(f"   {model_path}")
            print(f"\n3. Make sure filename is exactly: {model_info['filename']}")
            print(f"\n4. The tool will automatically detect and use it!")
            
            print(f"\n{'='*60}")
            print("Note: The tool works GREAT without models!")
            print("Traditional inpainting produces excellent results.")
            print(f"{'='*60}\n")
            
            return None
        
        # Try to download if URL provided
        print(f"\nDownloading {model_info['name']}...")
        print(f"Description: {model_info['description']}")
        print(f"Size: {model_info['size_mb']} MB")
        print(f"URL: {model_info['url']}\n")
        
        try:
            if show_progress:
                self._download_with_progress(model_info["url"], model_path)
            else:
                urllib.request.urlretrieve(model_info["url"], model_path)
            
            print(f"\n✓ Model downloaded successfully!")
            print(f"  Location: {model_path}")
            
            # Check if it's a zip file
            if model_path.suffix == ".zip":
                print("Extracting zip file...")
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    zip_ref.extractall(self.models_dir)
                # Find the actual model file
                for file in self.models_dir.rglob("*.onnx"):
                    if file.name == model_info["filename"]:
                        model_path = file
                        break
            
            return str(model_path)
        
        except Exception as e:
            print(f"\n✗ Error downloading model: {e}")
            print("\nManual download required. See instructions above.")
            return None
    
    def _download_with_progress(self, url: str, filepath: Path):
        """Download with progress bar"""
        try:
            from tqdm import tqdm
            
            def progress_hook(count, block_size, total_size):
                if tqdm_instance.total != total_size:
                    tqdm_instance.total = total_size
                tqdm_instance.update(block_size)
            
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as tqdm_instance:
                urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        except ImportError:
            # Fallback without progress bar
            urllib.request.urlretrieve(url, filepath)
    
    def list_available_models(self) -> Dict:
        """List all available models"""
        return self.MODELS
    
    def check_model_exists(self, model_id: Optional[str] = None) -> bool:
        """Check if model file exists"""
        if model_id is None:
            model_info = self.get_recommended_model()
        else:
            model_info = self.MODELS.get(model_id)
            if not model_info:
                return False
        
        model_path = self.models_dir / model_info["filename"]
        return model_path.exists()
    
    def get_model_path(self, model_id: Optional[str] = None) -> Optional[str]:
        """Get path to model file if it exists"""
        if self.check_model_exists(model_id):
            if model_id is None:
                model_info = self.get_recommended_model()
            else:
                model_info = self.MODELS.get(model_id)
            
            return str(self.models_dir / model_info["filename"])
        return None


def setup_recommended_model() -> Optional[str]:
    """
    Setup and check for recommended model for clothing removal
    
    Returns:
        Path to model file if exists, None otherwise
    """
    downloader = ModelDownloader()
    
    # Check if model already exists
    model_path = downloader.get_model_path()
    if model_path and Path(model_path).exists():
        return model_path
    
    # Model doesn't exist - that's OK, traditional method works great
    return None

