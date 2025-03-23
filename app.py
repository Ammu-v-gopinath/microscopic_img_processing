import os
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import uuid

from image_processing.stitching import stitch
from image_processing.roi import roi_extract
from image_processing.zoom import zoom
from image_processing.focus import auto_foc

app = Flask(__name__)

from flask import render_template


@app.route('/')
def index():
    """
    Render the index page.
    """
    return render_template('index.html')

# Configuer
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# Creating folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_images', methods=['POST'])
def upload_images():
  
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    uploaded_files = []
    session_id = str(uuid.uuid4())
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(session_folder, filename)
            file.save(file_path)
            uploaded_files.append(file_path)
    
    return jsonify({
        'message': f'Successfully uploaded {len(uploaded_files)} files',
        'session_id': session_id,
        'uploaded_files': uploaded_files
    }), 201

@app.route('/stitch_images', methods=['GET'])
def stitch_images_endpoint():
   
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session_id provided'}), 400
    
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    if not os.path.exists(session_folder):
        return jsonify({'error': 'Session not found'}), 404
    
    
    image_files = [os.path.join(session_folder, f) for f in os.listdir(session_folder) 
                  if allowed_file(f)]
    
    if not image_files:
        return jsonify({'error': 'No images found in session'}), 404
    
    # Loading images
    images = [cv2.imread(img) for img in image_files]
    
    # Stitching images
    try:
        stitched_image = stitch(images)
        
        # Save image that stitched
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_stitched.jpg")
        cv2.imwrite(result_path, stitched_image)
        
        return jsonify({
            'message': 'Images stitched successfully',
            'result_path': result_path
        }), 200
    except Exception as e:
        return jsonify({'error': f'Stitching failed: {str(e)}'}), 500

@app.route('/roi_selection', methods=['POST'])
def roi_selection():
  
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    required_fields = ['image_path', 'x', 'y', 'width', 'height']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    image_path = data['image_path']
    x = int(data['x'])
    y = int(data['y'])
    width = int(data['width'])
    height = int(data['height'])
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    try:
        # Extract ROI
        roi_image = roi_extract(image_path, x, y, width, height)
        
        # Save ROI 
        filename = os.path.basename(image_path)
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"roi_{filename}")
        cv2.imwrite(result_path, roi_image)
        
        return jsonify({
            'message': 'ROI extracted successfully',
            'result_path': result_path
        }), 200
    except Exception as e:
        return jsonify({'error': f'ROI extraction failed: {str(e)}'}), 500

@app.route('/zoom', methods=['POST'])
def zoom_():
  
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    required_fields = ['image_path', 'zoom_factor']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    image_path = data['image_path']
    zoom_factor = int(data['zoom_factor'])
    
    if zoom_factor not in [10, 20]:
        return jsonify({'error': 'Zoom factor must be either 10 or 20'}), 400
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    try:
        # zoom
        zoomed_image = zoom(image_path, zoom_factor)
        
        # Save zoomed img
        filename = os.path.basename(image_path)
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"zoom_{zoom_factor}x_{filename}")
        cv2.imwrite(result_path, zoomed_image)
        
        return jsonify({
            'message': f'Image zoomed {zoom_factor}X successfully',
            'result_path': result_path
        }), 200
    except Exception as e:
        return jsonify({'error': f'Zooming failed: {str(e)}'}), 500

@app.route('/auto_focus', methods=['GET'])
def auto_focus_endpoint():
   
    image_path = request.args.get('image_path')
    if not image_path:
        return jsonify({'error': 'No image_path provided'}), 400
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    try:
        # auto-focus
        focused_image = auto_foc(image_path)
        
        # Save focused img
        filename = os.path.basename(image_path)
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"focused_{filename}")
        cv2.imwrite(result_path, focused_image)
        
        return jsonify({
            'message': 'Image auto-focused successfully',
            'result_path': result_path
        }), 200
    except Exception as e:
        return jsonify({'error': f'Auto-focus failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)