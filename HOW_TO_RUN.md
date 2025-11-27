# 🚀 How to Run - Complete Setup Guide

## Quick Start (5 Minutes)

### Step 1: Installation

```bash
# Clone or download the project
cd deepfake

# Install all dependencies
pip install -r requirements.txt
```

**That's it!** The tool is ready to use.

---

## Step 2: Choose Your Method

### Method 1: GUI Application (Easiest) ⭐

```bash
python gui_app.py
```

**Features:**
- First tab: **⭐ Remove Clothing** (FLAGSHIP FEATURE)
- Click "Select Image to Process"
- Adjust settings (Gender, Body Type, Realism Level)
- Click "🔥 REMOVE CLOTHING NOW"
- Done!

### Method 2: Interactive CLI (Recommended for Beginners)

```bash
python main.py --interactive
# or
python cli.py
```

**Menu:**
1. Select "⭐ Remove Clothing (FLAGSHIP FEATURE)"
2. Enter image path
3. Configure options
4. Process!

### Method 3: Direct Command (Fast)

```bash
# Clothing removal
python main.py image --input photo.jpg --type remove --output result.png

# Face swapping
python main.py swap --source face1.jpg --target face2.jpg --output swapped.png

# Voice cloning
python main.py voice --sample voice.wav --text "Hello" --output output.wav
```

### Method 4: Python API

```python
from deepfake import DeepFakeFramework

framework = DeepFakeFramework()
framework.remove_clothes("input.jpg", "output.png")
```

---

## Step 3: Optional - Download AI Models (Better Quality)

For **even better** clothing removal results, download the AI model:

```bash
# Auto-download recommended model
python download_model.py --auto

# Or list available models
python download_model.py --list
```

**Model will be automatically placed in:** `models/nudify_model.onnx`

**Note:** The tool works great without models! Traditional inpainting produces excellent results.

---

## 📁 Project Structure

```
deepfake/
├── gui_app.py                    # GUI Application (RUN THIS)
├── main.py                       # CLI Interface
├── cli.py                        # Interactive CLI
├── download_model.py             # Model downloader
├── config.yaml                   # Configuration
├── requirements.txt              # Dependencies
│
├── deepfake/                     # Core framework
│   ├── core.py                   # Main API
│   ├── nudifier.py               # ⭐ Clothing removal (FLAGSHIP)
│   ├── face_swapper.py           # Face swapping
│   ├── voice_cloner.py           # Voice cloning
│   ├── color_detector.py         # Color detection
│   ├── body_estimator.py         # Body measurements
│   └── utils/                    # Utilities
│       ├── model_downloader.py   # Model downloader
│       ├── skin_generator.py     # Skin generation
│       ├── gender_detector.py    # Gender detection
│       └── ...
│
├── models/                       # AI models (auto-created)
│   └── nudify_model.onnx         # (Downloaded here)
│
├── HOW_TO_RUN.md                 # This file
├── CLOTHING_REMOVAL_GUIDE.md     # Detailed clothing removal guide
└── README.md                     # Full documentation
```

---

## 🎯 Main Feature: Clothing Removal

### Quick Usage

**GUI:**
1. `python gui_app.py`
2. Click "⭐ Remove Clothing" tab (first tab)
3. Select image → Click "🔥 REMOVE CLOTHING NOW"

**CLI:**
```bash
python main.py image --input photo.jpg --type remove --output result.png
```

**Python:**
```python
from deepfake import DeepFakeFramework
framework = DeepFakeFramework()
framework.remove_clothes("photo.jpg", "result.png")
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
clothing_removal:
  realistic_mode: true          # Enable realistic skin
  blend_factor: 0.8            # Realism level (0-1)
  confidence_threshold: 0.7    # Detection sensitivity

model:
  device: "cpu"                # Use "cuda" for GPU
```

---

## 🔧 Troubleshooting

### Issue: ModuleNotFoundError

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: GUI doesn't open

**Solution:**
- Windows: Usually works
- Linux: `sudo apt-get install python3-tk`
- macOS: Usually works

### Issue: Model download fails

**Solution:**
- Tool works without models (traditional method)
- Manual download: See `CLOTHING_REMOVAL_GUIDE.md`
- Place model in `models/` folder

### Issue: Processing is slow

**Solutions:**
- Use GPU: Set `device: "cuda"` in config.yaml
- Reduce image size
- Use lower blend_factor (0.7 instead of 0.9)

---

## 📚 Documentation Files

- **HOW_TO_RUN.md** (this file) - Quick start guide
- **CLOTHING_REMOVAL_GUIDE.md** - Complete clothing removal guide
- **CLI_GUIDE.md** - CLI usage guide
- **FACE_SWAP_GUIDE.md** - Face swapping guide
- **VOICE_CLONING_GUIDE.md** - Voice cloning guide
- **README.md** - Full documentation

---

## ✅ Verification Checklist

After installation, verify everything works:

```bash
# 1. Check installation
python -c "from deepfake import DeepFakeFramework; print('✓ Installed')"

# 2. Test GUI
python gui_app.py  # Should open GUI

# 3. Test CLI
python main.py --help  # Should show help

# 4. Test interactive CLI
python cli.py  # Should show menu
```

---

## 🎓 Example Workflow

### Complete Example: Remove Clothing

1. **Prepare your image** (JPG/PNG, preferably 512x512+)

2. **Run GUI:**
   ```bash
   python gui_app.py
   ```

3. **In GUI:**
   - Click "⭐ Remove Clothing" tab
   - Click "Select Image to Process"
   - Choose your image
   - Adjust settings (optional)
   - Click "🔥 REMOVE CLOTHING NOW"
   - Save result when prompted

4. **Done!** Check your output image.

---

## 🚨 Important Notes

### Legal & Ethical
- ⚠️ **Always get consent** before processing images
- ⚠️ **Use responsibly** - respect privacy and laws
- ⚠️ **Educational/research use** only
- ⚠️ **Do not misuse** this technology

### System Requirements
- **OS:** Windows/Linux/macOS
- **Python:** 3.8 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 2GB for installation, +1GB for models
- **GPU:** Optional (CUDA) for faster processing

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Run GUI | `python gui_app.py` |
| Run CLI | `python main.py --interactive` |
| Remove clothing | `python main.py image --input img.jpg --type remove --output out.png` |
| Download model | `python download_model.py --auto` |
| Check install | `python -c "from deepfake import DeepFakeFramework; print('OK')"` |

---

## 🎉 You're Ready!

The tool is **complete and ready to use**. Start with:

```bash
python gui_app.py
```

**For clothing removal (main feature):**
- GUI: First tab "⭐ Remove Clothing"
- CLI: Option 1 in menu
- Command: `python main.py image --input photo.jpg --type remove --output result.png`

---

**Version:** 2.1.0  
**Status:** Production Ready  
**Last Updated:** 2024

