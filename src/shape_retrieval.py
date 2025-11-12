import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def compute_shape_distance(features1, features2, weights=None):
    if weights is None:
        weights = {'fourier': 0.5, 'direction': 0.3, 'hu_moments': 0.2}
    
    fourier_dist = euclidean_distance(
        features1['fourier_descriptors'],
        features2['fourier_descriptors']
    )
    
    direction_dist = euclidean_distance(
        features1['direction_histogram'],
        features2['direction_histogram']
    )
    
    hu_dist = euclidean_distance(
        features1['hu_moments'],
        features2['hu_moments']
    )
    
    return (weights['fourier'] * fourier_dist +
            weights['direction'] * direction_dist +
            weights['hu_moments'] * hu_dist)

def retrieve_similar_shapes(query_image_name, features_folder, images_folder, top_k=6):
    query_json = os.path.join(features_folder, Path(query_image_name).stem + '.json')
    
    if not os.path.exists(query_json):
        raise ValueError(f"Feature file not found: {query_json}")
    
    query_features = load_features_from_json(query_json)
    distances = []
    
    for json_file in Path(features_folder).glob('*.json'):
        image_name = json_file.stem
        features = load_features_from_json(str(json_file))
        distance = compute_shape_distance(query_features, features)
        
        for ext in ['.gif', '.png', '.jpg', '.jpeg']:
            image_path = os.path.join(images_folder, image_name + ext)
            if os.path.exists(image_path):
                distances.append((os.path.basename(image_path), distance, image_path))
                break
    
    distances.sort(key=lambda x: x[1])
    
    results = []
    for img_name, dist, img_path in distances:
        if img_name != query_image_name:
            results.append((img_name, dist, img_path))
        if len(results) == top_k:
            break
    
    return results

def visualize_shape_results(query_image_path, results, output_path=None):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Shape-Based Image Retrieval', fontsize=14)
    
    query_img = cv2.imread(query_image_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title(f'Query: {os.path.basename(query_image_path)}')
    axes[0, 0].axis('off')
    
    for i in range(1, 4):
        axes[0, i].axis('off')
    
    for idx, (img_name, distance, img_path) in enumerate(results):
        row, col = 1, idx % 4
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'{img_name}\nDist: {distance:.4f}')
        axes[row, col].axis('off')
    
    for idx in range(len(results), 4):
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()
