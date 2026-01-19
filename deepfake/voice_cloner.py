utf-8"""
Voice Cloning and Text-to-Speech Module
Clones a voice from a sample and synthesizes speech from text
"""

import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path
import os
import tempfile

from .utils.audio_processor import AudioProcessor


try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    pyttsx3 = None

try:
    from gtts import gTTS
    import io
    from pydub import AudioSegment
    from pydub.playback import play
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    gTTS = None

try:
    import TTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False
    TTS = None


class VoiceCloner:
    """
    Voice cloning and text-to-speech synthesis
    Analyzes voice samples and generates speech in that voice
    """
    
    def __init__(self, method: str = "auto", device: str = "cpu"):
        """
        Initialize voice cloner
        
        Args:
            method: TTS method ("coqui", "pyttsx3", "gtts", or "auto")
            device: Device for processing ("cpu" or "cuda")
        """
        self.method = method
        self.device = device
        self.audio_processor = AudioProcessor()
        
        
        self.tts_engine = None
        self.coqui_tts = None
        self._init_tts()
        
        
        self.voice_profile: Optional[Dict] = None
    
    def _init_tts(self):
        """Initialize TTS engine based on available libraries"""
        if self.method == "coqui" and COQUI_TTS_AVAILABLE:
            try:
                self.coqui_tts = TTS.tts.TTS("tts_models/multilingual/multi-dataset/your_tts")
                self.method = "coqui"
                return
            except Exception as e:
                print(f"Warning: Could not initialize Coqui TTS: {e}")
        
        if self.method == "pyttsx3" and PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.method = "pyttsx3"
                return
            except Exception as e:
                print(f"Warning: Could not initialize pyttsx3: {e}")
        
        
        if COQUI_TTS_AVAILABLE:
            try:
                self.coqui_tts = TTS.tts.TTS("tts_models/multilingual/multi-dataset/your_tts")
                self.method = "coqui"
                return
            except:
                pass
        
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.method = "pyttsx3"
                return
            except:
                pass
        
        if GTTS_AVAILABLE:
            self.method = "gtts"
            return
        
        raise RuntimeError(
            "No TTS library available. Install one of: "
            "pip install coqui-tts OR pip install pyttsx3 OR pip install gtts"
        )
    
    def analyze_voice(self, audio_path: str) -> Dict:
        """
        Analyze voice sample and extract characteristics
        
        Args:
            audio_path: Path to voice sample audio file
            
        Returns:
            Dictionary with voice characteristics
        """
        
        audio, sr = self.audio_processor.load_audio(audio_path)
        
        
        audio = self.audio_processor.trim_silence(audio)
        
        
        features = self.audio_processor.extract_features(audio, sr)
        
        
        self.voice_profile = {
            "features": features,
            "sample_rate": sr,
            "duration": features["duration"],
            "source_audio": audio_path
        }
        
        return self.voice_profile
    
    def synthesize_speech(self, text: str, 
                         output_path: str,
                         voice_profile: Optional[Dict] = None,
                         adjust_to_profile: bool = True) -> Dict:
        """
        Synthesize speech from text using voice profile
        
        Args:
            text: Text to synthesize
            output_path: Path to save output audio
            voice_profile: Optional voice profile (uses stored if None)
            adjust_to_profile: Whether to adjust synthesized speech to match profile
            
        Returns:
            Dictionary with synthesis metadata
        """
        if voice_profile is None:
            voice_profile = self.voice_profile
        
        if voice_profile is None:
            raise ValueError("No voice profile available. Call analyze_voice() first.")
        
        
        if self.method == "coqui" and self.coqui_tts:
            return self._synthesize_coqui(text, output_path, voice_profile, adjust_to_profile)
        elif self.method == "pyttsx3" and self.tts_engine:
            return self._synthesize_pyttsx3(text, output_path, voice_profile, adjust_to_profile)
        elif self.method == "gtts":
            return self._synthesize_gtts(text, output_path, voice_profile, adjust_to_profile)
        else:
            raise RuntimeError(f"TTS method {self.method} not available")
    
    def _synthesize_coqui(self, text: str, output_path: str,
                         voice_profile: Dict, adjust_to_profile: bool) -> Dict:
        """Synthesize using Coqui TTS with voice cloning"""
        try:
            
            speaker_wav = voice_profile.get("source_audio")
            
            if speaker_wav and os.path.exists(speaker_wav):
                
                self.coqui_tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=speaker_wav
                )
            else:
                
                self.coqui_tts.tts_to_file(
                    text=text,
                    file_path=output_path
                )
            
            
            if adjust_to_profile:
                self._adjust_to_profile(output_path, voice_profile)
            
            return {
                "method": "coqui",
                "output_path": output_path,
                "text": text,
                "duration": self._get_audio_duration(output_path)
            }
        
        except Exception as e:
            raise RuntimeError(f"Coqui TTS synthesis failed: {e}")
    
    def _synthesize_pyttsx3(self, text: str, output_path: str,
                            voice_profile: Dict, adjust_to_profile: bool) -> Dict:
        """Synthesize using pyttsx3 with voice parameter adjustment"""
        try:
            
            features = voice_profile.get("features", {})
            
            
            tempo = features.get("tempo", 120)
            rate = int(150 + (tempo - 120) * 0.5)  
            self.tts_engine.setProperty('rate', rate)
            
            
            pitch_mean = features.get("pitch_mean", 0)
            if pitch_mean > 0:
                
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    
                    if pitch_mean > 200:  
                        
                        for voice in voices:
                            if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                                self.tts_engine.setProperty('voice', voice.id)
                                break
                    else:  
                        for voice in voices:
                            if 'male' in voice.name.lower() or 'man' in voice.name.lower():
                                self.tts_engine.setProperty('voice', voice.id)
                                break
            
            
            self.tts_engine.save_to_file(text, output_path)
            self.tts_engine.runAndWait()
            
            
            if adjust_to_profile:
                self._adjust_to_profile(output_path, voice_profile)
            
            return {
                "method": "pyttsx3",
                "output_path": output_path,
                "text": text,
                "duration": self._get_audio_duration(output_path)
            }
        
        except Exception as e:
            raise RuntimeError(f"pyttsx3 synthesis failed: {e}")
    
    def _synthesize_gtts(self, text: str, output_path: str,
                        voice_profile: Dict, adjust_to_profile: bool) -> Dict:
        """Synthesize using Google TTS (limited voice cloning)"""
        try:
            
            features = voice_profile.get("features", {})
            
            
            lang = 'en'
            
            
            tts = gTTS(text=text, lang=lang, slow=False)
            
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tmp_path = tmp_file.name
                tts.save(tmp_path)
            
            
            if adjust_to_profile:
                audio, sr = self.audio_processor.load_audio(tmp_path)
                
                
                audio = self._adjust_audio_to_profile(audio, sr, voice_profile)
                
                
                self.audio_processor.save_audio(audio, output_path, sr)
                os.remove(tmp_path)
            else:
                
                audio, sr = self.audio_processor.load_audio(tmp_path)
                self.audio_processor.save_audio(audio, output_path, sr)
                os.remove(tmp_path)
            
            return {
                "method": "gtts",
                "output_path": output_path,
                "text": text,
                "duration": self._get_audio_duration(output_path)
            }
        
        except Exception as e:
            raise RuntimeError(f"gTTS synthesis failed: {e}")
    
    def _adjust_to_profile(self, audio_path: str, voice_profile: Dict):
        """
        Adjust synthesized audio to better match voice profile
        
        Args:
            audio_path: Path to audio file to adjust
            voice_profile: Voice profile to match
        """
        
        audio, sr = self.audio_processor.load_audio(audio_path)
        
        
        adjusted = self._adjust_audio_to_profile(audio, sr, voice_profile)
        
        
        self.audio_processor.save_audio(adjusted, audio_path, sr)
    
    def _adjust_audio_to_profile(self, audio: np.ndarray, sr: int,
                                 voice_profile: Dict) -> np.ndarray:
        """
        Adjust audio parameters to match voice profile
        
        Args:
            audio: Input audio
            sr: Sample rate
            voice_profile: Target voice profile
            
        Returns:
            Adjusted audio
        """
        features = voice_profile.get("features", {})
        
        
        target_pitch = features.get("pitch_mean", 0)
        if target_pitch > 0:
            
            current_features = self.audio_processor.extract_features(audio, sr)
            current_pitch = current_features.get("pitch_mean", target_pitch)
            
            if current_pitch > 0:
                
                semitones = 12 * np.log2(target_pitch / current_pitch)
                
                semitones = np.clip(semitones, -4, 4)
                
                if abs(semitones) > 0.1:
                    audio = self.audio_processor.adjust_pitch(audio, sr, semitones)
        
        
        target_tempo = features.get("tempo", 120)
        current_features = self.audio_processor.extract_features(audio, sr)
        current_tempo = current_features.get("tempo", target_tempo)
        
        if current_tempo > 0:
            speed_factor = target_tempo / current_tempo
            speed_factor = np.clip(speed_factor, 0.8, 1.2)  
            
            if abs(speed_factor - 1.0) > 0.05:
                audio = self.audio_processor.adjust_speed(audio, speed_factor)
        
        
        audio = self.audio_processor.normalize_audio(audio)
        
        return audio
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file"""
        try:
            audio, sr = self.audio_processor.load_audio(audio_path)
            return len(audio) / sr
        except:
            return 0.0
    
    def clone_voice_from_sample(self, voice_sample_path: str,
                               text: str,
                               output_path: str) -> Dict:
        """
        Complete voice cloning workflow
        
        Args:
            voice_sample_path: Path to voice sample audio
            text: Text to synthesize
            output_path: Path to save output
            
        Returns:
            Dictionary with results and metadata
        """
        
        print("Analyzing voice sample...")
        voice_profile = self.analyze_voice(voice_sample_path)
        
        print(f"Voice characteristics:")
        print(f"  Pitch: {voice_profile['features'].get('pitch_mean', 0):.1f} Hz")
        print(f"  Tempo: {voice_profile['features'].get('tempo', 0):.1f} BPM")
        print(f"  Duration: {voice_profile['duration']:.2f}s")
        
        
        print(f"Synthesizing speech: '{text[:50]}...'")
        result = self.synthesize_speech(
            text=text,
            output_path=output_path,
            adjust_to_profile=True
        )
        
        print(f"✓ Synthesis complete! Saved to: {output_path}")
        
        return {
            "voice_profile": voice_profile,
            "synthesis": result,
            "success": True
        }

