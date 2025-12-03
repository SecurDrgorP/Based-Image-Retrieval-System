"""
app.py - Flask Web Application for CBIR System
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.shape_features import extract_shape_features, process_all_shape_images
from src.texture_features import extract_texture_features, process_all_texture_images
from src.shape_retrieval import retrieve_similar_shapes
from src.texture_retrieval import retrieve_similar_textures
from src.utils import save_features_to_json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.template_folder = 'template'

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Home page
@app.route('/')
def index():
    return render_template('index.html')


# Get list of available shape images.
@app.route('/api/images/shapes')
def get_shape_images():
    images = []
    for ext in ['*.gif', '*.png', '*.jpg', '*.jpeg']:
        images.extend(Path('data/Formes').glob(ext))
    return jsonify([img.name for img in sorted(images)])


# Get list of available texture images.
@app.route('/api/images/textures')
def get_texture_images():
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(Path('data/Textures').glob(ext))
    return jsonify([img.name for img in sorted(images)])


# Extract features for all shape images.
@app.route('/api/extract/shapes', methods=['POST'])
def extract_shapes():
    try:
        process_all_shape_images('data/Formes', 'features/Formes')
        return jsonify({'success': True, 'message': 'Shape features extracted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Extract features for all texture images.
@app.route('/api/extract/textures', methods=['POST'])
def extract_textures():
    try:
        process_all_texture_images('data/Textures', 'features/Textures')
        return jsonify({'success': True, 'message': 'Texture features extracted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Search for similar shapes.
@app.route('/api/search/shapes', methods=['POST'])
def search_shapes():
    try:
        data = request.json
        query_image = data.get('image')
        top_k = data.get('top_k', 6)
        
        if not query_image:
            return jsonify({'success': False, 'error': 'No image specified'}), 400
        
        results = retrieve_similar_shapes(
            query_image,
            'features/Formes',
            'data/Formes',
            top_k
        )
        
        # Format results
        formatted_results = [
            {
                'name': name,
                'distance': float(dist),
                'similarity': max(0, 100 - dist * 10),
                'path': f'/images/Formes/{name}'
            }
            for name, dist, path in results
        ]
        
        return jsonify({
            'success': True,
            'query': query_image,
            'results': formatted_results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Search for similar textures.
@app.route('/api/search/textures', methods=['POST'])
def search_textures():
    try:
        data = request.json
        query_image = data.get('image')
        top_k = data.get('top_k', 6)
        
        if not query_image:
            return jsonify({'success': False, 'error': 'No image specified'}), 400
        
        results = retrieve_similar_textures(
            query_image,
            'features/Textures',
            'data/Textures',
            top_k
        )
        
        # Format results
        formatted_results = [
            {
                'name': name,
                'distance': float(dist),
                'similarity': max(0, 100 - dist * 20),
                'path': f'/images/Textures/{name}'
            }
            for name, dist, path in results
        ]
        
        return jsonify({
            'success': True,
            'query': query_image,
            'results': formatted_results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Upload a new image for search.
@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        search_type = request.form.get('type', 'shape')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract features
            if search_type == 'shape':
                features = extract_shape_features(filepath)
                json_path = os.path.join('features/Formes', Path(filename).stem + '.json')
                save_features_to_json(features, json_path)
                # Copy to data folder
                import shutil
                shutil.copy(filepath, os.path.join('data/Formes', filename))
            else:
                features = extract_texture_features(filepath)
                json_path = os.path.join('features/Textures', Path(filename).stem + '.json')
                save_features_to_json(features, json_path)
                # Copy to data folder
                import shutil
                shutil.copy(filepath, os.path.join('data/Textures', filename))
            
            return jsonify({
                'success': True,
                'filename': filename,
                'message': 'File uploaded and features extracted'
            })
        
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Serve images from data folder
@app.route('/images/<folder>/<filename>')
def serve_image(folder, filename):
    return send_from_directory(f'data/{folder}', filename)


# Serve uploaded images
@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)