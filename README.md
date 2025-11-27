# 🔥 DeepFake Framework v2.1 - Ultimate Edition

**The Most Advanced DeepFake Framework** - Complete solution for image manipulation, voice cloning, and realistic content generation.

## 🌟 **FLAGSHIP FEATURE: Advanced Clothing Removal** 👕

**This is our most important and advanced feature** - A state-of-the-art clothing removal system that produces incredibly realistic, natural-looking results using:
- **Body-aware processing** with gender and body type detection
- **Realistic skin generation** with texture synthesis
- **Natural lighting and shadows** for authentic appearance
- **Multi-algorithm inpainting** for seamless results

**[👉 See Complete Clothing Removal Guide](CLOTHING_REMOVAL_GUIDE.md)**

## ✨ New Features in v2.1

- 🎤 **Voice Cloning** - Clone voices from samples and synthesize speech from text
- 🎨 **Realistic Skin Generation** - Body-aware texture synthesis with shadows and lighting
- 👤 **Gender Detection** - Automatic gender identification from body proportions
- 📏 **Body Type Awareness** - Rectangle, Pear, Inverted Triangle, Hourglass classifications
- 🖼️ **Modern GUI** - Beautiful interface with logo, preview, and advanced options
- 🔄 **Face Swapping** - Image-to-image and image-to-video face swapping
- ⚙️ **Enhanced Configuration** - Fine-tune every aspect of processing
- 🎯 **Authentic Results** - Realistic output considering body size, gender, and type

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### ⭐ Clothing Removal (Quick Start)

**The easiest way to remove clothing:**

```bash
# GUI (Recommended)
python gui_app.py
# Select image → Choose "Clothing Removal Only" → Process

# Command Line
python main.py image --input photo.jpg --type remove --output result.png

# Python API
from deepfake import DeepFakeFramework
framework = DeepFakeFramework()
framework.remove_clothes("photo.jpg", "result.png")
```

**For detailed guide, see [CLOTHING_REMOVAL_GUIDE.md](CLOTHING_REMOVAL_GUIDE.md)**

### GUI Application (Easiest)

```bash
python gui_app.py
```

**Features:**
- Modern dark theme interface
- Real-time image preview (Original/Processed toggle)
- Processing mode selection
- Gender and body type controls
- Advanced settings (blend factor, confidence)
- Live information display

### Web Interface (Most Convenient) 🆕

```bash
# Install web requirements
pip install -r web_requirements.txt

# Run web server
python web_app.py

# Open in browser: http://localhost:5000
```

**Features:**
- Modern responsive web interface
- All core features accessible through browser
- Drag and drop file uploads
- Real-time processing with progress indicators
- Downloadable results
- Mobile-friendly design

See [WEB_INTERFACE_README.md](WEB_INTERFACE_README.md) for detailed documentation.

### Command Line

#### Interactive Mode (Recommended)
```bash
# Launch interactive menu-driven CLI
python main.py --interactive
# or
python main.py -i
# or
python cli.py
```

#### Direct Commands
```bash
# Complete processing pipeline
python main.py image --input photo.jpg --type complete --output result.png

# Face swapping
python main.py swap --source face1.jpg --target face2.jpg --output swapped.png

# Voice cloning
python main.py voice --sample voice.wav --text "Hello" --output output.wav

# Legacy mode (backwards compatible)
python main.py --input image.jpg --output result.png --mode complete
```

### Python API

```python
from deepfake import DeepFakeFramework

# Initialize framework
framework = DeepFakeFramework()

# Complete processing with all enhancements
results = framework.process_complete("input.jpg", output_path="output.png")

# Access metadata
print(f"Gender: {results['body_size']['gender']['gender']}")
print(f"Body Type: {results['body_size']['measurements']['body_type']}")
print(f"Colors: {results['clothes_color']['primary_color']['name']}")
```

## 🎯 Core Capabilities

### 1. Clothes Color Detection
- K-Means clustering for multi-color extraction
- Color classification (Red, Blue, Green, etc.)
- RGB/HEX conversion
- Percentage breakdown

### 2. Body Size Estimation
- MediaPipe pose detection
- Automatic measurements (shoulder width, hip width, leg length)
- Body type classification
- Gender detection from proportions

### 3. Face Swapping 🆕
- **Image-to-Image**: Swap faces between two photos
- **Image-to-Video**: Swap face from photo to video
- Multiple face detection and selection
- Advanced blending modes (seamless, feathered, linear)
- Color correction for natural results

### 4. Clothing Removal ⭐ **FLAGSHIP FEATURE**
- **🎯 The Most Advanced Feature** - Ultra-realistic clothing removal
- **Body-Aware Processing** - Considers gender, body type, and measurements
- **Realistic Skin Generation** - Advanced texture synthesis with shadows
- **Multi-Algorithm Inpainting** - Seamless, natural-looking results
- **Automatic Detection** - Gender, body type, skin tones automatically detected
- **Customizable Realism** - Adjustable blend factor for quality control
- **See [CLOTHING_REMOVAL_GUIDE.md](CLOTHING_REMOVAL_GUIDE.md) for complete documentation**

### 5. Voice Cloning 🆕
- **Voice Analysis** - Extract voice characteristics from audio samples
- **Text-to-Speech** - Synthesize speech from text in cloned voice
- **Multiple TTS Methods** - Coqui TTS, pyttsx3, Google TTS support
- **Pitch & Tempo Matching** - Automatic adjustment to match voice profile
- **High Quality** - Neural network-based voice cloning (Coqui TTS)

### 6. Additional Features
- **Realistic mode**: Body-aware skin generation
- **Gender-aware**: Different skin tones for male/female
- **Body-type aware**: Texture adjustments based on body shape
- **Natural shadows**: Realistic lighting and depth
- **Seamless blending**: Smooth transitions with surrounding areas

## 📊 Processing Modes

### Image Processing
1. **complete** - Full pipeline (color + size + removal + metadata)
2. **remove_only** - Only clothing removal with realism
3. **color_only** - Color detection and analysis
4. **size_only** - Body measurements and classification

### Face Swapping 🆕
1. **swap** (image-to-image) - Swap faces between two photos
2. **swap** (image-to-video) - Swap face from photo to video

## ⚙️ Configuration

Edit `config.yaml`:

```yaml
clothing_removal:
  realistic_mode: true          # Enable realistic skin generation
  gender_detection: true        # Auto-detect gender
  body_type_aware: true        # Use body type for processing
  blend_factor: 0.8            # Realism blending (0-1)
  confidence_threshold: 0.7

body_size:
  use_mediapipe: true
  reference_height: 170        # cm

color_detection:
  algorithm: "kmeans"
  num_colors: 5
```

## 🔄 Face Swapping

### Quick Start

```bash
# Swap faces between images
python main.py swap --source person1.jpg --target person2.jpg --output result.png

# Swap face to video
python main.py swap --source face.jpg --target video.mp4 --output output.mp4
```

### Python API

```python
# Image to image
framework.swap_faces_image(
    "source.jpg", "target.jpg", "output.png",
    blend_mode="seamless"
)

# Image to video
framework.swap_face_to_video(
    "face.jpg", "video.mp4", "output.mp4"
)
```

See [FACE_SWAP_GUIDE.md](FACE_SWAP_GUIDE.md) for detailed documentation.

## 🎨 Advanced Features

### Gender Detection

```python
from deepfake.utils.gender_detector import GenderDetector

detector = GenderDetector()
gender_info = detector.detect_gender(image, pose_data)
# Returns: {"gender": "male"/"female"/"unknown", "confidence": 0.0-1.0}
```

### Realistic Skin Generation

```python
from deepfake.utils.skin_generator import RealisticSkinGenerator

generator = RealisticSkinGenerator()
skin = generator.generate_realistic_skin(
    base_tone=(220, 180, 150),
    region_shape=(512, 512),
    mask=body_mask,
    body_type="Pear",
    gender="female",
    add_texture=True,
    add_shadows=True
)
```

### Manual Overrides

```python
# Override automatic detection
result = framework.remove_clothes(
    "image.jpg",
    gender="female",           # Override gender
    body_type="Hourglass",     # Override body type
    blend_factor=0.9          # Adjust realism
)
```

## 📁 Project Structure

```
deepfake/
├── deepfake/
│   ├── core.py                  # Main framework API
│   ├── color_detector.py        # Color detection
│   ├── body_estimator.py        # Body measurements + gender
│   ├── nudifier.py              # Enhanced clothing removal
│   └── utils/
│       ├── gender_detector.py   # Gender detection
│       ├── skin_generator.py    # Realistic skin synthesis
│       ├── image_processor.py   # Image utilities
│       └── color_utils.py       # Color utilities
├── templates/                   # Web interface templates
│   └── index.html               # Main web interface
├── gui_app.py                   # Desktop GUI application
├── web_app.py                   # Web interface application
├── main.py                      # CLI interface
├── config.yaml                  # Configuration
├── requirements.txt             # Main dependencies
├── web_requirements.txt         # Web interface dependencies
├── WEB_INTERFACE_README.md      # Web interface documentation
└── README.md                    # This file
```

## 🎯 Best Practices

1. **Image Quality**: Use high-resolution images (at least 512x512)
2. **Lighting**: Natural lighting produces best skin tone detection
3. **Pose**: Full-body or upper-body shots work best
4. **Background**: Clear backgrounds improve detection
5. **Realistic Mode**: Keep enabled for authentic results

## 🔧 Troubleshooting

**GUI not opening?**
- Ensure tkinter is installed (usually comes with Python)
- On Linux: `sudo apt-get install python3-tk`

**Gender detection inaccurate?**
- Use manual override in GUI or API
- Ensure pose is clearly visible in image

**Skin looks unrealistic?**
- Adjust `blend_factor` (try 0.7-0.9)
- Ensure `realistic_mode: true` in config
- Check that body type and gender are correctly detected

**Processing slow?**
- Use GPU if available: set `device: "cuda"` in config
- Reduce image resolution for faster processing
- Disable `realistic_mode` for faster (but less realistic) results

## 📊 Output Format

### Complete Processing Result

```python
{
    "processed_image": np.ndarray,      # Processed image
    "clothes_color": {
        "primary_color": {
            "rgb": (255, 128, 64),
            "hex": "#ff8040",
            "name": "Orange",
            "percentage": 45.2
        },
        "dominant_colors": [...]
    },
    "body_size": {
        "measurements": {
            "shoulder_width_cm": 42.5,
            "hip_width_cm": 38.2,
            "body_type": "Inverted Triangle"
        },
        "gender": {
            "gender": "male",
            "confidence": 0.85,
            "method": "proportions"
        }
    },
    "output_path": "output.png"
}
```

## 🚀 Performance Tips

1. **GPU Acceleration**: Use CUDA for 5-10x speedup
2. **Batch Processing**: Process multiple images together
3. **Resolution**: Lower resolution = faster processing
4. **Realistic Mode**: Disable for faster results (less realistic)

## 📝 Examples

See `example_usage.py` and `QUICKSTART.md` for detailed examples.

## ⚖️ Legal and Ethical Notice

⚠️ **IMPORTANT**: This framework is for educational and research purposes only.

- Users must comply with local laws and regulations
- Obtain proper consent before processing images
- Respect privacy and dignity of individuals
- Use technology ethically and responsibly

## 📄 License

This project is provided as-is for educational purposes.

## 🔄 Version History

### v2.0 Enhanced (Current)
- ✅ Realistic skin generation with texture synthesis
- ✅ Gender detection from body proportions
- ✅ Body type-aware processing
- ✅ Modern GUI with logo and options
- ✅ Enhanced configuration system

### v1.0
- Basic clothing removal
- Color detection
- Body size estimation

---

**Version**: 2.0 Enhanced  
**Status**: Production Ready  
**Last Updated**: 2024
