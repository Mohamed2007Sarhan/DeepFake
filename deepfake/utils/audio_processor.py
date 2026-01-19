utf-8"""
Audio Processing Utilities
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, Dict
import scipy.signal as signal
from pathlib import Path


class AudioProcessor:
    """Utilities for audio processing and analysis"""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize audio processor
        
        Args:
            sample_rate: Target sample rate for processing
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, audio_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (uses self.sample_rate if None)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        target_sr = target_sr or self.sample_rate
        
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Could not load audio file {audio_path}: {e}")
    
    def save_audio(self, audio: np.ndarray, output_path: str, sample_rate: Optional[int] = None):
        """
        Save audio to file
        
        Args:
            audio: Audio data array
            output_path: Output file path
            sample_rate: Sample rate (uses self.sample_rate if None)
        """
        sr = sample_rate or self.sample_rate
        
        try:
            sf.write(output_path, audio, sr)
        except Exception as e:
            raise ValueError(f"Could not save audio file {output_path}: {e}")
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Extract voice features from audio
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch = pitches[pitches > 0]
        features['pitch_mean'] = float(np.mean(pitch)) if len(pitch) > 0 else 0.0
        features['pitch_std'] = float(np.std(pitch)) if len(pitch) > 0 else 0.0
        
        
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = [float(x) for x in np.mean(mfccs, axis=1)]
        
        
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        
        
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = float(tempo)
        
        
        features['duration'] = len(audio) / sr
        
        return features
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio: Input audio
            
        Returns:
            Normalized audio
        """
        if len(audio) == 0:
            return audio
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio
        
        Args:
            audio: Input audio
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """
        Trim silence from beginning and end
        
        Args:
            audio: Input audio
            top_db: Silence threshold in dB
            
        Returns:
            Trimmed audio
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    
    def adjust_pitch(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """
        Adjust pitch of audio
        
        Args:
            audio: Input audio
            sr: Sample rate
            semitones: Pitch shift in semitones (positive = higher, negative = lower)
            
        Returns:
            Pitch-adjusted audio
        """
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
    
    def adjust_speed(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
        """
        Adjust playback speed
        
        Args:
            audio: Input audio
            speed_factor: Speed multiplier (1.0 = normal, >1 = faster, <1 = slower)
            
        Returns:
            Speed-adjusted audio
        """
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    def apply_filters(self, audio: np.ndarray, sr: int, 
                     lowcut: Optional[float] = None,
                     highcut: Optional[float] = None) -> np.ndarray:
        """
        Apply frequency filters
        
        Args:
            audio: Input audio
            sr: Sample rate
            lowcut: Low frequency cutoff (Hz)
            highcut: High frequency cutoff (Hz)
            
        Returns:
            Filtered audio
        """
        
        nyquist = sr / 2
        
        if lowcut and highcut:
            
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(5, [low, high], btype='band')
        elif lowcut:
            
            low = lowcut / nyquist
            b, a = signal.butter(5, low, btype='high')
        elif highcut:
            
            high = highcut / nyquist
            b, a = signal.butter(5, high, btype='low')
        else:
            return audio
        
        filtered = signal.filtfilt(b, a, audio)
        return filtered

