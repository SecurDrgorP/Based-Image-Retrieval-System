import json
import os
import numpy as np
import cv2

def save_features_to_json(features, output_path):
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
    with open(json_path, 'r') as f:
        features = json.load(f)
    
    for key, value in features.items():
        if isinstance(value, list):
            features[key] = np.array(value)
    
    return features

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def cosine_distance(vec1, vec2):
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
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray, img
