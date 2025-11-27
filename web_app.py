"""
DeepFake Framework - Web Interface
Modern web application with sleek black theme for convenient use
"""

import os
import sys
import json
import base64
import io
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deepfake import DeepFakeFramework

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deepfake_framework_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the framework
framework = DeepFakeFramework()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'mp4', 'avi', 'mov', 'wav', 'mp3', 'flac'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image):
    """Convert numpy image array to base64 string for web display"""
    if image is None:
        return None
    
    # Convert to PIL Image
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/info')
def get_info():
    """Get framework information"""
    try:
        info = framework.get_framework_info()
        return jsonify({"status": "success", "data": info})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload file endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return jsonify({"status": "success", "filepath": filepath, "filename": filename})
        else:
            return jsonify({"status": "error", "message": "Invalid file type"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/clothes_color', methods=['POST'])
def get_clothes_color():
    """Get clothes color information"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({"status": "error", "message": "Invalid image path"})
        
        result = framework.get_clothes_color(image_path)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/body_size', methods=['POST'])
def get_body_size():
    """Get body size information"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({"status": "error", "message": "Invalid image path"})
        
        result = framework.get_body_size(image_path)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/remove_clothes', methods=['POST'])
def remove_clothes():
    """Remove clothes from image"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        use_advanced = data.get('use_advanced', False)
        blend_factor = data.get('blend_factor', 0.8)
        gender = data.get('gender', None)
        body_type = data.get('body_type', None)
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({"status": "error", "message": "Invalid image path"})
        
        # Process image
        result = framework.remove_clothes(
            image_path,
            use_advanced=use_advanced,
            blend_factor=blend_factor,
            gender=gender,
            body_type=body_type
        )
        
        # Convert to base64 for web display
        result_base64 = image_to_base64(result)
        
        return jsonify({
            "status": "success", 
            "processed_image": result_base64,
            "shape": result.shape if result is not None else None
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/complete_process', methods=['POST'])
def complete_process():
    """Complete processing pipeline"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({"status": "error", "message": "Invalid image path"})
        
        # Process image
        result = framework.process_complete(image_path, return_metadata=True)
        
        # Convert processed image to base64
        processed_image_base64 = image_to_base64(result["processed_image"])
        
        return jsonify({
            "status": "success",
            "clothes_color": result["clothes_color"],
            "body_size": result["body_size"],
            "processed_image": processed_image_base64
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/swap_faces', methods=['POST'])
def swap_faces():
    """Swap faces between images"""
    try:
        data = request.get_json()
        source_path = data.get('source_path')
        target_path = data.get('target_path')
        blend_mode = data.get('blend_mode', 'seamless')
        
        if not source_path or not os.path.exists(source_path):
            return jsonify({"status": "error", "message": "Invalid source image path"})
        
        if not target_path or not os.path.exists(target_path):
            return jsonify({"status": "error", "message": "Invalid target image path"})
        
        # Create output path
        output_filename = f"swapped_{os.path.basename(target_path)}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Process face swap
        framework.swap_faces_image(
            source_path,
            target_path,
            output_path,
            blend_mode=blend_mode
        )
        
        # Read result and convert to base64
        result_image = cv2.imread(output_path)
        if result_image is not None:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            result_base64 = image_to_base64(result_image)
            return jsonify({
                "status": "success",
                "result_image": result_base64,
                "output_path": output_path
            })
        else:
            return jsonify({"status": "error", "message": "Failed to process face swap"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/clone_voice', methods=['POST'])
def clone_voice():
    """Clone voice from sample"""
    try:
        data = request.get_json()
        voice_sample_path = data.get('voice_sample_path')
        text = data.get('text', '')
        
        if not voice_sample_path or not os.path.exists(voice_sample_path):
            return jsonify({"status": "error", "message": "Invalid voice sample path"})
        
        if not text:
            return jsonify({"status": "error", "message": "Text is required"})
        
        # Create output path
        output_filename = f"cloned_voice_{os.path.basename(voice_sample_path)}.wav"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Process voice cloning
        result = framework.clone_voice(voice_sample_path, text, output_path)
        
        return jsonify({
            "status": "success",
            "output_path": output_path,
            "message": "Voice cloning completed successfully"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)