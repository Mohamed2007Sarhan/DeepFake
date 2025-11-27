# ⚡ Quick Start: Clothing Removal in 60 Seconds

## The Fastest Way to Remove Clothing

### Method 1: GUI (Easiest) ⭐

```bash
python gui_app.py
```

1. Click **"Image Processing"** tab
2. Click **"Select Image"** → Choose your photo
3. Select **"Clothing Removal Only"** from dropdown
4. Click **"🚀 Process Image"**
5. **Done!** Result appears in preview

**Time: ~30 seconds**

---

### Method 2: One-Line Command

```bash
python main.py image --input your_photo.jpg --type remove --output result.png
```

**Time: ~10 seconds**

---

### Method 3: Python API (3 Lines)

```python
from deepfake import DeepFakeFramework

framework = DeepFakeFramework()
framework.remove_clothes("your_photo.jpg", "result.png")
```

**Time: ~5 seconds to write**

---

## 🎯 For Best Results

Add these settings for maximum realism:

```python
framework.remove_clothes(
    "photo.jpg",
    "result.png",
    blend_factor=0.9,      # Maximum realism (0.0-1.0)
    gender="female",       # Or "male", or None for auto
    body_type="Pear"       # Or "Rectangle", "Inverted Triangle", "Hourglass"
)
```

---

## 📚 Want More?

- **Full Guide**: [CLOTHING_REMOVAL_GUIDE.md](CLOTHING_REMOVAL_GUIDE.md)
- **Examples**: [CLOTHING_REMOVAL_EXAMPLES.py](CLOTHING_REMOVAL_EXAMPLES.py)
- **Main README**: [README.md](README.md)

---

**That's it! You're ready to use the most advanced clothing removal system. 🚀**

