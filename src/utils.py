"""
utils.py - Core utility functions
"""

import json
import os
import numpy as np
import cv2
from PIL import Image


def save_features_to_json(features, output_path):
    """Save feature vectors to JSON file."""
    json_features = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            json_features[key] = value.tolist()
        else:
            json_features[key] = value
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(json_features, f, indent=2)


def load_features_from_json(json_path):
    """Load feature vectors from JSON file."""
    with open(json_path, 'r') as f:
        features = json.load(f)
    
    for key, value in features.items():
        if isinstance(value, list):
            features[key] = np.array(value)
    
    return features


def euclidean_distance(vec1, vec2):
    """Compute Euclidean distance between two vectors."""
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


def cosine_distance(vec1, vec2):
    """Compute cosine distance between two vectors."""
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0
    
    similarity = dot_product / (norm1 * norm2)
    return 1 - similarity


def load_image(image_path):
    """
    Load image from file. Handles GIF, PNG, JPG formats.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        tuple: (grayscale, color) image arrays
        
    Raises:
        ValueError: If image cannot be loaded
    """
    # Try loading with OpenCV first (works for PNG, JPG)
    img = cv2.imread(image_path)
    
    # If OpenCV fails (e.g., for GIF), use PIL
    if img is None:
        try:
            # Load with PIL
            pil_image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL Image to numpy array (RGB format)
            img = np.array(pil_image)
            
            # Convert RGB to BGR (OpenCV format)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            raise ValueError(f"Cannot load image: {image_path} - {str(e)}")
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    return gray, img