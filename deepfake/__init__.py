"""
DeepFake Framework - Advanced Image Manipulation System
"""

__version__ = "1.0.0"

from .core import DeepFakeFramework
from .color_detector import ClothesColorDetector
from .body_estimator import BodySizeEstimator
from .nudifier import ClothingRemover
from .face_swapper import FaceSwapper
from .voice_cloner import VoiceCloner
from .utils.gender_detector import GenderDetector
from .utils.skin_generator import RealisticSkinGenerator
from .utils.face_detector import FaceDetector
from .utils.audio_processor import AudioProcessor

__all__ = [
    "DeepFakeFramework",
    "ClothesColorDetector",
    "BodySizeEstimator",
    "ClothingRemover",
    "FaceSwapper",
    "VoiceCloner",
    "GenderDetector",
    "RealisticSkinGenerator",
    "FaceDetector",
    "AudioProcessor"
]

