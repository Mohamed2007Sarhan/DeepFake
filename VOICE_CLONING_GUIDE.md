# Voice Cloning Guide

## Overview

The DeepFake Framework includes advanced voice cloning and text-to-speech capabilities that allow you to:
1. **Analyze a voice sample** - Extract voice characteristics (pitch, tempo, timbre)
2. **Clone the voice** - Create a voice profile from the sample
3. **Synthesize speech** - Generate new speech in that voice from text

## Quick Start

### Command Line

```bash
# Clone voice and synthesize text
python main.py voice --sample voice_sample.wav --text "Hello, this is a test" --output output.wav
```

### Python API

```python
from deepfake import DeepFakeFramework

framework = DeepFakeFramework()

# Clone voice and synthesize
result = framework.clone_voice(
    voice_sample_path="sample.wav",
    text="Your text here",
    output_path="output.wav"
)
```

### GUI Application

1. Launch GUI: `python gui_app.py`
2. Click "Voice Clone" tab
3. Select voice sample audio file
4. Enter text to synthesize
5. Configure options
6. Click "🎤 Clone Voice & Synthesize"

## How It Works

### 1. Voice Analysis

The system analyzes the voice sample to extract:
- **Pitch** (fundamental frequency)
- **Tempo** (speaking rate)
- **Spectral characteristics** (timbre, brightness)
- **MFCC** (Mel-frequency cepstral coefficients)
- **Zero crossing rate**
- **Duration**

### 2. Voice Cloning

Depending on the TTS method used:
- **Coqui TTS**: Full voice cloning with deep learning
- **pyttsx3**: Parameter adjustment (pitch, speed, voice selection)
- **gTTS**: Basic synthesis with post-processing adjustment

### 3. Synthesis & Adjustment

The text is synthesized and then adjusted to match:
- Target pitch from voice sample
- Target tempo/speed
- Spectral characteristics

## TTS Methods

### Coqui TTS (Recommended)

**Pros:**
- True voice cloning with neural networks
- High quality, natural results
- Supports multilingual synthesis

**Requirements:**
```bash
pip install coqui-tts
```

**Usage:**
```python
framework = DeepFakeFramework(
    voice_cloning={"method": "coqui"}
)
```

### pyttsx3

**Pros:**
- Works offline
- Fast synthesis
- Multiple voice options

**Cons:**
- Limited voice cloning (parameter adjustment only)
- Quality depends on system voices

**Usage:**
```python
framework = DeepFakeFramework(
    voice_cloning={"method": "pyttsx3"}
)
```

### Google TTS (gTTS)

**Pros:**
- Free and easy to use
- Good basic quality
- Multiple languages

**Cons:**
- Requires internet
- No true voice cloning (post-processing only)
- Character limit

**Usage:**
```python
framework = DeepFakeFramework(
    voice_cloning={"method": "gtts"}
)
```

## Voice Sample Requirements

### Recommended:
- **Duration**: 3-30 seconds
- **Format**: WAV, MP3, FLAC
- **Quality**: Clear, minimal background noise
- **Content**: Single speaker, clear speech
- **Sample rate**: 16kHz or higher

### Best Practices:
- Use high-quality recordings
- Ensure minimal background noise
- Single speaker only
- Clear pronunciation
- Consistent speaking style

## Examples

### Example 1: Basic Voice Cloning

```python
from deepfake import DeepFakeFramework

framework = DeepFakeFramework()

result = framework.clone_voice(
    voice_sample_path="person_voice.wav",
    text="Hello, my name is John and I am speaking to you now.",
    output_path="cloned_voice.wav"
)

print(f"Method: {result['synthesis']['method']}")
print(f"Duration: {result['synthesis']['duration']:.2f}s")
```

### Example 2: Analyze Voice Only

```python
voice_profile = framework.analyze_voice("sample.wav")

print(f"Pitch: {voice_profile['features']['pitch_mean']:.1f} Hz")
print(f"Tempo: {voice_profile['features']['tempo']:.1f} BPM")
print(f"Duration: {voice_profile['duration']:.2f}s")
```

### Example 3: Custom TTS Method

```python
from deepfake import VoiceCloner

cloner = VoiceCloner(method="coqui")

result = cloner.clone_voice_from_sample(
    "voice.wav",
    "Your custom text here",
    "output.wav"
)
```

### Example 4: Batch Processing

```python
texts = [
    "First sentence to synthesize.",
    "Second sentence with different content.",
    "Third sentence for batch processing."
]

for i, text in enumerate(texts):
    framework.clone_voice(
        "sample.wav",
        text,
        f"output_{i}.wav"
    )
```

## Configuration

### config.yaml

```yaml
voice_cloning:
  method: "auto"  # "coqui", "pyttsx3", "gtts", or "auto"
  adjust_to_profile: true
  sample_rate: 22050
```

## Advanced Features

### Pitch Adjustment

The system automatically adjusts pitch to match the voice sample:

```python
# Analyze voice
profile = framework.analyze_voice("sample.wav")

# Pitch will be automatically matched during synthesis
result = framework.synthesize_speech(
    "Your text",
    "output.wav",
    voice_sample_path="sample.wav"
)
```

### Tempo Matching

Speaking rate is adjusted to match the original:

- Fast speakers → Faster synthesis
- Slow speakers → Slower synthesis
- Natural pacing preserved

### Manual Adjustment

```python
from deepfake.utils.audio_processor import AudioProcessor

processor = AudioProcessor()

# Load audio
audio, sr = processor.load_audio("output.wav")

# Adjust pitch manually
adjusted = processor.adjust_pitch(audio, sr, semitones=2)  # Higher pitch

# Adjust speed
adjusted = processor.adjust_speed(adjusted, speed_factor=1.1)  # 10% faster

# Save
processor.save_audio(adjusted, "adjusted.wav", sr)
```

## Troubleshooting

### "No TTS library available"

**Solution:** Install one of the TTS libraries:
```bash
# Option 1: Coqui TTS (best quality)
pip install coqui-tts

# Option 2: pyttsx3 (offline)
pip install pyttsx3

# Option 3: gTTS (online)
pip install gtts pydub
```

### Poor Voice Match

**Solutions:**
- Use longer voice samples (10+ seconds)
- Ensure high-quality, clear recording
- Try different TTS methods (Coqui TTS works best)
- Check that `adjust_to_profile: true` in config

### Audio Format Issues

**Solution:** Convert audio to WAV format:
```python
from deepfake.utils.audio_processor import AudioProcessor

processor = AudioProcessor()
audio, sr = processor.load_audio("input.mp3")
processor.save_audio(audio, "output.wav", sr)
```

### Coqui TTS Installation Issues

**Solution:**
```bash
# Install system dependencies first
# Linux:
sudo apt-get install espeak-ng

# Then install Coqui TTS
pip install coqui-tts
```

## Performance Tips

1. **Use Coqui TTS** for best quality voice cloning
2. **Sample rate**: 22050 Hz is optimal balance of quality/speed
3. **Voice sample length**: 5-15 seconds is ideal
4. **Batch processing**: Process multiple texts with same sample for efficiency

## Limitations

- **pyttsx3/gTTS**: Limited voice cloning (parameter adjustment only)
- **Coqui TTS**: Requires more computational resources
- **Language**: Some methods support limited languages
- **Quality**: Depends on voice sample quality and length

## Legal and Ethical Notice

⚠️ **IMPORTANT**: Voice cloning technology has serious ethical implications:

- **Obtain consent** before cloning anyone's voice
- **Use responsibly** - do not impersonate others
- **Comply with laws** regarding voice recordings and synthesis
- **Respect privacy** and individual rights
- **Consider implications** of deepfake audio

This technology should only be used:
- With explicit permission
- For legitimate purposes (accessibility, research, creative projects)
- In compliance with all applicable laws

---

**Version**: 2.1.0  
**Last Updated**: 2024

