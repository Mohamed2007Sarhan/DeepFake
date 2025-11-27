# CLI Interface Guide

## Overview

The DeepFake Framework provides two CLI modes:
1. **Interactive Mode** - Menu-driven interface with rich formatting (recommended)
2. **Command Mode** - Direct command-line arguments

## Interactive Mode

### Launch

```bash
# Method 1: Using main script
python main.py --interactive
python main.py -i

# Method 2: Using CLI launcher
python cli.py
```

### Features

- **Rich Formatting** - Colorful, formatted output with tables and panels
- **Progress Bars** - Visual feedback for long operations
- **Interactive Menus** - Easy navigation through options
- **File Validation** - Automatic file existence checking
- **Result Display** - Formatted tables for results
- **Error Handling** - Clear error messages

### Menu Structure

```
Main Menu:
  1. Image Processing (Color, Size, Clothing Removal)
  2. Face Swapping (Image-to-Image)
  3. Face Swapping (Image-to-Video)
  4. Voice Cloning & Synthesis
  5. Framework Information
  0. Exit
```

### Image Processing Submenu

```
1. Complete Processing (Color + Size + Removal)
2. Color Detection Only
3. Body Size Estimation Only
4. Clothing Removal Only
0. Back to Main Menu
```

## Command Mode

### General Syntax

```bash
python main.py <command> [options]
```

### Commands

#### Image Processing

```bash
python main.py image --input <path> [--output <path>] [--type <type>]

Types:
  - complete    : Full pipeline (default)
  - color       : Color detection only
  - size        : Body size estimation only
  - remove      : Clothing removal only
```

**Examples:**
```bash
python main.py image --input photo.jpg --type complete --output result.png
python main.py image --input photo.jpg --type color
```

#### Face Swapping

**Image-to-Image:**
```bash
python main.py swap --source <source_image> --target <target_image> --output <output>
```

**Image-to-Video:**
```bash
python main.py swap --source <source_image> --target <target_video> --output <output_video>
```

**Options:**
- `--blend` : Blend mode (seamless, linear, feathered) [default: seamless]
- `--source-face` : Source face index [default: 0]
- `--target-face` : Target face index [default: 0]

**Examples:**
```bash
python main.py swap --source person1.jpg --target person2.jpg --output swapped.png
python main.py swap --source face.jpg --target video.mp4 --output output.mp4 --blend seamless
```

#### Voice Cloning

```bash
python main.py voice --sample <audio_file> --text "<text>" --output <output_audio>
```

**Options:**
- `--method` : TTS method (auto, coqui, pyttsx3, gtts) [default: auto]

**Examples:**
```bash
python main.py voice --sample voice.wav --text "Hello world" --output output.wav
python main.py voice --sample voice.mp3 --text "Custom text" --output result.wav --method coqui
```

## CLI Enhancements

### Rich Library Support

For the best experience, install rich and colorama:

```bash
pip install rich colorama
```

**Benefits:**
- Colorful output
- Formatted tables
- Progress bars
- Better error display
- Interactive prompts

### Fallback Mode

If rich/colorama are not installed, the CLI falls back to:
- Basic text formatting
- Simple prompts
- Plain progress indicators

## Usage Examples

### Example 1: Interactive Mode

```bash
$ python main.py -i

============================================================
              🔥 DeepFake Framework v2.1
============================================================

DeepFake Framework - Main Menu
┌────────┬────────────────────────────────────────────┐
│ Option │ Description                                │
├────────┼────────────────────────────────────────────┤
│ 1      │ Image Processing                           │
│ 2      │ Face Swapping (Image-to-Image)            │
│ 3      │ Face Swapping (Image-to-Video)            │
│ 4      │ Voice Cloning & Synthesis                 │
│ 5      │ Framework Information                     │
│ 0      │ Exit                                       │
└────────┴────────────────────────────────────────────┘

Select an option: 1
```

### Example 2: Quick Command

```bash
# Quick color detection
python main.py image --input photo.jpg --type color

# Quick face swap
python main.py swap --source face1.jpg --target face2.jpg --output result.png
```

### Example 3: Batch Processing Script

```python
from deepfake import DeepFakeFramework

framework = DeepFakeFramework()

images = ["img1.jpg", "img2.jpg", "img3.jpg"]
for img in images:
    result = framework.process_complete(img, f"output_{img}")
    print(f"Processed: {img}")
```

## Keyboard Shortcuts

### Interactive Mode

- `Ctrl+C` - Cancel current operation / Exit confirmation
- `Enter` - Confirm selection / Continue
- `0` - Go back / Exit

## Tips

1. **Use Interactive Mode** for first-time users or complex operations
2. **Use Command Mode** for automation and scripts
3. **Install rich/colorama** for better visual experience
4. **Check file paths** - CLI validates file existence automatically
5. **Use tab completion** in terminal for faster path entry

## Troubleshooting

### "ModuleNotFoundError: No module named 'rich'"

**Solution:**
```bash
pip install rich colorama
```

Or use without rich (falls back to basic mode)

### Interactive mode not launching

**Solution:**
```bash
# Explicitly call interactive
python -m deepfake.cli.cli_interface
```

### File not found errors

**Tips:**
- Use absolute paths if relative paths don't work
- Remove quotes from paths if copied from file explorer
- Check file extensions match

---

**Version**: 2.1.0  
**Last Updated**: 2024

