# DeepFake Framework - Web Interface

A modern web interface for the DeepFake Framework that provides convenient access to all image and voice processing features.

## Features

- **Clothing Removal**: AI-powered clothing removal with realistic results
- **Face Swap**: Swap faces between images and videos
- **Voice Cloning**: Clone voices from audio samples
- **Image Analysis**: Extract color and body size information
- **Batch Processing**: Process multiple images at once

## Installation

1. Install the web interface requirements:
```bash
pip install -r web_requirements.txt
```

2. Make sure you have the main framework installed:
```bash
pip install -r requirements.txt
```

## Running the Web Interface

1. Start the web server:
```bash
python web_app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Clothing Removal**:
   - Upload an image using the "Clothing Removal" section
   - Adjust options like AI model, realism level, gender, and body type
   - Click "Remove Clothing" to process the image
   - Download the result when processing is complete

2. **Face Swap**:
   - Upload a source image (containing the face you want to use)
   - Upload a target image or video (where you want to place the face)
   - Select blending options
   - Click "Swap Faces" to process
   - Download the result when processing is complete

3. **Voice Cloning**:
   - Upload a voice sample (WAV, MP3, or FLAC)
   - Enter text you want to synthesize with the cloned voice
   - Click "Clone Voice" to process
   - Download the synthesized audio when processing is complete

4. **Image Analysis**:
   - Upload an image for analysis
   - Click "Analyze Image" to extract color and body size information
   - View the results in the analysis panel

## API Endpoints

The web interface uses the following API endpoints:

- `GET /` - Main web interface
- `GET /api/info` - Get framework information
- `POST /api/upload` - Upload files
- `POST /api/clothes_color` - Get clothes color information
- `POST /api/body_size` - Get body size information
- `POST /api/remove_clothes` - Remove clothes from image
- `POST /api/complete_process` - Complete processing pipeline
- `POST /api/swap_faces` - Swap faces between images
- `POST /api/clone_voice` - Clone voice from sample

## Configuration

The web interface uses the same configuration as the main framework (`config.yaml`).

## Troubleshooting

If you encounter issues:

1. Make sure all requirements are installed:
   ```bash
   pip install -r requirements.txt
   pip install -r web_requirements.txt
   ```

2. Check that the models are properly downloaded:
   ```bash
   python download_model.py
   ```

3. Ensure you have enough system resources (RAM, GPU) for processing

4. Check the console for error messages when running the web app

## Support

For support, please check the main README.md file or open an issue on the project repository.