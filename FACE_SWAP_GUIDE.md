# Face Swapping Guide

## Overview

The DeepFake Framework now includes powerful face swapping capabilities that support:
- **Image-to-Image**: Swap faces between two photos
- **Image-to-Video**: Swap a face from a photo onto faces in a video

## Quick Start

### Command Line

#### Swap Faces Between Two Images
```bash
python main.py swap --source source_face.jpg --target target_image.jpg --output result.png
```

#### Swap Face to Video
```bash
python main.py swap --source source_face.jpg --target video.mp4 --output output_video.mp4
```

### Python API

```python
from deepfake import DeepFakeFramework

framework = DeepFakeFramework()

# Swap between images
metadata = framework.swap_faces_image(
    source_image_path="person1.jpg",
    target_image_path="person2.jpg",
    output_path="result.png",
    blend_mode="seamless"
)

# Swap to video
metadata = framework.swap_face_to_video(
    source_image_path="person1.jpg",
    target_video_path="video.mp4",
    output_video_path="output.mp4"
)
```

### GUI Application

1. Launch GUI: `python gui_app.py`
2. Click "Face Swap" tab
3. Select source image (face to copy)
4. Select target image or video (face to replace)
5. Configure options (blend mode, face indices)
6. Click "🔄 Swap Faces"

## Advanced Options

### Blend Modes

1. **seamless** (Recommended)
   - Uses OpenCV seamless cloning
   - Best for natural blending
   - Preserves lighting and shadows

2. **feathered**
   - Soft edge blending with Gaussian blur
   - Good for smooth transitions
   - Fast processing

3. **linear**
   - Simple linear blending
   - Fastest method
   - Basic results

### Face Detection Methods

Configure in `config.yaml`:

```yaml
face_swap:
  method: "dlib"  # Options: "dlib", "mediapipe", "opencv"
  model_path: null  # Path to dlib shape predictor (optional)
```

**Methods:**
- **dlib**: Most accurate, requires model file
- **mediapipe**: Good accuracy, no model needed
- **opencv**: Fast but less accurate

### Multiple Faces

If an image contains multiple faces, specify which ones to use:

```python
# Use 2nd face from source, 1st face from target
metadata = framework.swap_faces_image(
    source_image_path="group_photo.jpg",
    target_image_path="target.jpg",
    output_path="result.png",
    source_face_idx=1,  # 0-indexed
    target_face_idx=0
)
```

## Tips for Best Results

### Image Quality
- Use high-resolution images (at least 512x512)
- Clear, well-lit photos work best
- Front-facing faces give better results

### Lighting
- Match lighting between source and target for best results
- The algorithm attempts color correction automatically

### Face Pose
- Front-facing faces work best
- Slight angles (up to 45°) are acceptable
- Profile views are not recommended

### Video Processing
- Processing time depends on video length and resolution
- For long videos, consider preprocessing to extract frames
- Ensure stable face tracking in video

## Configuration

### config.yaml

```yaml
face_swap:
  method: "dlib"
  model_path: "models/shape_predictor_68_face_landmarks.dat"
  blend_mode: "seamless"
  blend_factor: 0.8
```

### Download dlib Model (Optional)

For best accuracy with dlib, download the shape predictor:

```bash
# Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract and place in models/ directory
```

## API Reference

### `swap_faces_image()`

```python
metadata = framework.swap_faces_image(
    source_image_path: str,
    target_image_path: str,
    output_path: str,
    source_face_idx: int = 0,
    target_face_idx: int = 0,
    blend_mode: str = "seamless",
    blend_factor: float = 0.8
)
```

**Returns:**
```python
{
    "source_faces_detected": 1,
    "target_faces_detected": 1,
    "source_face_used": 0,
    "target_face_used": 0,
    "blend_mode": "seamless",
    "output_path": "result.png"
}
```

### `swap_face_to_video()`

```python
metadata = framework.swap_face_to_video(
    source_image_path: str,
    target_video_path: str,
    output_video_path: str,
    source_face_idx: int = 0,
    blend_mode: str = "seamless"
)
```

**Returns:**
```python
{
    "total_frames": 300,
    "faces_detected": 285,
    "fps": 30,
    "resolution": (1920, 1080)
}
```

## Troubleshooting

### No Face Detected
- Ensure face is clearly visible and front-facing
- Try different detection method in config
- Check image quality and lighting

### Poor Blending
- Try different blend modes
- Adjust blend_factor (0.6-0.9 range)
- Ensure source and target have similar lighting

### Video Processing Issues
- Ensure video codec is supported (MP4 recommended)
- Check available disk space
- Processing time scales with video length

### dlib Model Not Found
- Download model from dlib website
- Place in `models/` directory
- Or switch to "mediapipe" or "opencv" method

## Examples

### Example 1: Simple Face Swap
```python
from deepfake import DeepFakeFramework

framework = DeepFakeFramework()
framework.swap_faces_image(
    "person1.jpg",
    "person2.jpg",
    "swapped.jpg"
)
```

### Example 2: Batch Processing
```python
source_images = ["face1.jpg", "face2.jpg", "face3.jpg"]
target = "group_photo.jpg"

for i, source in enumerate(source_images):
    framework.swap_faces_image(
        source,
        target,
        f"output_{i}.jpg"
    )
```

### Example 3: Video Processing with Progress
```python
import time

start = time.time()
metadata = framework.swap_face_to_video(
    "face.jpg",
    "video.mp4",
    "output.mp4"
)
elapsed = time.time() - start

print(f"Processed {metadata['total_frames']} frames in {elapsed:.1f}s")
print(f"Average: {metadata['total_frames']/elapsed:.1f} fps")
```

---

**Note**: Always ensure you have proper rights and consent to process images/videos containing people's faces.

