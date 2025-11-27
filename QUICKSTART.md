# Quick Start Guide - Enhanced DeepFake Framework

## 🚀 Getting Started

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Using the GUI (Recommended)

Launch the graphical interface:

```bash
python gui_app.py
```

**Features:**
- ✅ Modern interface with logo
- ✅ Real-time image preview
- ✅ Advanced options panel
- ✅ Gender and body type controls
- ✅ Processing modes (complete, remove only, color, size)
- ✅ Live information display

### Using Command Line

```bash
# Complete processing (color + size + removal)
python main.py --input image.jpg --output result.png --mode complete

# Only clothing removal
python main.py --input image.jpg --output result.png --mode remove

# Only color detection
python main.py --input image.jpg --mode color

# Only body size estimation
python main.py --input image.jpg --mode size
```

### Using Python API

```python
from deepfake import DeepFakeFramework

# Initialize framework
framework = DeepFakeFramework()

# Complete processing with all enhancements
results = framework.process_complete(
    "input.jpg",
    output_path="output.png",
    return_metadata=True
)

# Access results
print(f"Gender: {results['body_size']['gender']['gender']}")
print(f"Body Type: {results['body_size']['measurements']['body_type']}")
print(f"Primary Color: {results['clothes_color']['primary_color']['name']}")
```

## 🎨 New Enhanced Features

### 1. Realistic Body-Aware Processing
- Automatic gender detection
- Body type classification (Rectangle, Pear, Inverted Triangle, Hourglass)
- Realistic skin texture generation
- Body-aware shadow and lighting

### 2. Gender Detection
```python
from deepfake.utils.gender_detector import GenderDetector

detector = GenderDetector()
gender_info = detector.detect_gender(image)
print(f"Gender: {gender_info['gender']} (confidence: {gender_info['confidence']:.2f})")
```

### 3. Realistic Skin Generation
```python
from deepfake.utils.skin_generator import RealisticSkinGenerator

generator = RealisticSkinGenerator()
skin = generator.generate_realistic_skin(
    base_tone=(220, 180, 150),
    region_shape=(512, 512),
    mask=body_mask,
    body_type="Rectangle",
    gender="female",
    add_texture=True,
    add_shadows=True
)
```

### 4. Advanced Clothing Removal
```python
# With gender and body type awareness
result = framework.remove_clothes(
    "image.jpg",
    output_path="output.png",
    gender="female",  # Optional override
    body_type="Pear",  # Optional override
    blend_factor=0.85  # Realism adjustment
)
```

## ⚙️ Configuration

Edit `config.yaml` for advanced settings:

```yaml
clothing_removal:
  realistic_mode: true      # Enable realistic skin generation
  gender_detection: true    # Auto-detect gender
  body_type_aware: true    # Use body type for processing
  blend_factor: 0.8        # Realism blending (0-1)
  confidence_threshold: 0.7
```

## 📊 Processing Modes

1. **complete** - Full pipeline (color + size + removal)
2. **remove_only** - Only clothing removal
3. **color_only** - Only color detection
4. **size_only** - Only body measurements

## 🔧 Advanced Usage

### Manual Gender/Body Type Override

```python
framework = DeepFakeFramework()

# Override automatic detection
result = framework.remove_clothes(
    "image.jpg",
    gender="female",
    body_type="Hourglass",
    blend_factor=0.9
)
```

### Batch Processing

```python
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = framework.batch_process(
    images,
    output_dir="outputs/",
    return_metadata=True
)
```

## 🎯 Tips for Best Results

1. **Image Quality**: Use high-resolution images with good lighting
2. **Pose**: Full-body or upper-body shots work best
3. **Background**: Clear backgrounds improve detection accuracy
4. **Lighting**: Natural lighting produces better skin tones
5. **Realistic Mode**: Keep enabled for best visual results

## 🐛 Troubleshooting

**Issue**: Gender detection not accurate
- Solution: Use manual override via GUI or API

**Issue**: Skin looks unrealistic
- Solution: Adjust `blend_factor` in config or use `realistic_mode: true`

**Issue**: GUI not opening
- Solution: Ensure tkinter is installed (usually comes with Python)

**Issue**: Processing is slow
- Solution: Use GPU if available (`device: "cuda"` in config)

## 📝 Examples

See `example_usage.py` for more code examples.

---

**Version**: 2.0 Enhanced  
**Last Updated**: 2024

