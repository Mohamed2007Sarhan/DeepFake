# ⭐ Clothing Removal Guide - FLAGSHIP FEATURE

## Overview

The **Clothing Removal** feature is the most advanced capability of the DeepFake Framework. It produces incredibly realistic, natural-looking results through:

- **Body-aware processing** with gender and body type detection
- **Realistic skin generation** with texture synthesis
- **Natural lighting and shadows** for authentic appearance
- **Multi-algorithm inpainting** for seamless results

---

## Quick Start

### GUI (Easiest)

1. Run: `python gui_app.py`
2. Click **"⭐ Remove Clothing"** tab (first tab)
3. Click **"Select Image to Process"**
4. Adjust settings (optional)
5. Click **"🔥 REMOVE CLOTHING NOW"**
6. Done!

### CLI

```bash
python main.py image --input photo.jpg --type remove --output result.png
```

### Interactive CLI

```bash
python main.py --interactive
# Select option 1: Remove Clothing
```

### Python API

```python
from deepfake import DeepFakeFramework

framework = DeepFakeFramework()

# Basic usage
result = framework.remove_clothes("input.jpg", "output.png")

# Advanced usage with options
result = framework.remove_clothes(
    "input.jpg",
    output_path="output.png",
    gender="female",  # or "male", or None for auto
    body_type="Hourglass",  # or "Rectangle", "Pear", "Inverted Triangle"
    blend_factor=0.9,  # Higher = more realistic (0-1)
    use_advanced=True  # Use AI model if available
)
```

---

## Advanced Options

### Gender Override

The framework automatically detects gender, but you can override:

```python
# Auto-detect (recommended)
framework.remove_clothes("input.jpg", "output.png")

# Manual override
framework.remove_clothes(
    "input.jpg", 
    "output.png",
    gender="female"  # Options: "male", "female", or None (auto)
)
```

**Why it matters:** Different skin tones and body shapes are used based on gender for more realistic results.

### Body Type Override

Body type affects skin texture and body proportions:

```python
framework.remove_clothes(
    "input.jpg",
    "output.png",
    body_type="Hourglass"  # Options: "Rectangle", "Pear", "Inverted Triangle", "Hourglass"
)
```

**Body Types:**
- **Rectangle**: Straight silhouette
- **Pear**: Narrow shoulders, wider hips
- **Inverted Triangle**: Broad shoulders, narrow hips
- **Hourglass**: Balanced proportions

### Realism Level (Blend Factor)

Control the realism level:

```python
framework.remove_clothes(
    "input.jpg",
    "output.png",
    blend_factor=0.9  # 0.5-1.0, higher = more realistic
)
```

- **0.5-0.7**: Faster processing, good results
- **0.8-0.9**: Excellent quality (recommended)
- **0.9-1.0**: Maximum realism, slower processing

---

## AI Model Support

### Using Advanced AI Models

For even better quality, you can use AI models:

1. **Download model** (optional):
   ```bash
   python download_model.py --auto
   ```

2. **Use in code**:
   ```python
   framework.remove_clothes(
       "input.jpg",
       "output.png",
       use_advanced=True  # Use AI model if available
   )
   ```

3. **Model location**: `models/nudify_model.onnx`

**Note:** The tool works excellently without models! Traditional inpainting produces great results. Models are optional for additional quality improvement.

### Manual Model Setup

1. Download an ONNX model for clothing removal/nudify
2. Place in `models/` folder
3. Rename to `nudify_model.onnx`
4. Tool will automatically detect and use it

---

## Configuration

Edit `config.yaml`:

```yaml
clothing_removal:
  realistic_mode: true          # Enable realistic skin generation
  blend_factor: 0.8            # Default realism level
  confidence_threshold: 0.7    # Detection sensitivity
  gender_detection: true       # Auto-detect gender
  body_type_aware: true        # Consider body type
  skin_tone_preservation: true # Preserve natural skin tones
```

---

## Tips for Best Results

### Image Quality
- **Resolution**: Higher resolution = better results (512x512+ recommended)
- **Lighting**: Good lighting produces better skin tone detection
- **Pose**: Front-facing or side poses work best
- **Clothing**: Clear contrast between clothing and skin helps detection

### Settings
- **Gender**: Specify if detection is uncertain
- **Body Type**: Specify for more accurate body proportions
- **Blend Factor**: Start with 0.85, increase for more realism
- **Realistic Mode**: Keep enabled for best results

### Processing
- First run may be slower (initialization)
- GPU acceleration (CUDA) speeds up processing
- Larger images take longer to process

---

## Troubleshooting

### No Clothing Detected

**Issue:** Warning: "No clothing detected in image"

**Solutions:**
- Ensure person is clearly visible
- Check lighting conditions
- Try adjusting `confidence_threshold` in config.yaml (lower value = more sensitive)
- Manually specify gender/body type

### Unrealistic Results

**Solutions:**
- Increase `blend_factor` (try 0.9)
- Enable `realistic_mode` in config
- Specify gender and body type
- Use higher quality input image

### Slow Processing

**Solutions:**
- Reduce image size (resize before processing)
- Use lower `blend_factor` (0.7 instead of 0.9)
- Enable GPU if available (set `device: "cuda"` in config)
- Close other applications

### Model Not Found

**Issue:** AI model not loading

**Solutions:**
- Tool works without models (traditional method)
- Check model path in config.yaml
- Download model: `python download_model.py --auto`
- Verify model file: `models/nudify_model.onnx`

---

## Examples

### Example 1: Basic Removal

```python
from deepfake import DeepFakeFramework

framework = DeepFakeFramework()
framework.remove_clothes("photo.jpg", "result.png")
```

### Example 2: Advanced with Options

```python
from deepfake import DeepFakeFramework

framework = DeepFakeFramework()

result = framework.remove_clothes(
    "input.jpg",
    output_path="output.png",
    gender="female",
    body_type="Hourglass",
    blend_factor=0.9,
    use_advanced=True
)

print(f"Processed image shape: {result.shape}")
```

### Example 3: Batch Processing

```python
from deepfake import DeepFakeFramework
from pathlib import Path

framework = DeepFakeFramework()

input_dir = Path("input_images")
output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob("*.jpg"):
    output_path = output_dir / f"processed_{img_path.name}"
    framework.remove_clothes(str(img_path), str(output_path))
    print(f"Processed: {img_path.name}")
```

---

## Technical Details

### How It Works

1. **Body Detection**: MediaPipe pose estimation for body landmarks
2. **Gender Detection**: Machine learning-based gender classification
3. **Body Type Classification**: Proportional analysis of body measurements
4. **Skin Tone Estimation**: Advanced color analysis of visible skin
5. **Clothing Segmentation**: Multi-algorithm clothing region detection
6. **Skin Generation**: Realistic texture synthesis with:
   - Natural gradients
   - Body-type specific textures
   - Gender-appropriate tones
   - Realistic shadows and lighting
7. **Inpainting**: Advanced inpainting algorithms for seamless results
8. **Blending**: Multi-stage blending for natural transitions

### Algorithms Used

- **MediaPipe**: Pose detection and body segmentation
- **K-Means Clustering**: Color analysis
- **OpenCV Inpainting**: Telea and Navier-Stokes algorithms
- **Gaussian Blur**: Edge feathering
- **Texture Synthesis**: Custom skin generation

---

## Legal & Ethical

⚠️ **IMPORTANT REMINDERS:**

- Always get **consent** before processing images
- Use **responsibly** and ethically
- Comply with **local laws**
- Respect **privacy** and rights
- **Educational/research use** only
- Do not **misuse** this technology

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review `HOW_TO_RUN.md` for setup
3. Check `README.md` for full documentation

---

**Version:** 2.1.0  
**Status:** Production Ready  
**Feature:** ⭐ FLAGSHIP - Most Advanced Feature
