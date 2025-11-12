import cv2
import numpy as np
import os
from pathlib import Path
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops

def gabor_filters(image, num_orientations=8, num_scales=5):
    features = []
    
    for scale in range(num_scales):
        lambd = 2 ** (scale + 2)
        
        for orientation in range(num_orientations):
            theta = orientation * np.pi / num_orientations
            
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=3.0,
                theta=theta,
                lambd=lambd,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            features.append(np.mean(filtered))
            features.append(np.std(filtered))
    
    return np.array(features)

def tamura_coarseness(image, k_max=5):
    image = image.astype(float)
    h, w = image.shape
    
    A = np.zeros((k_max, h, w))
    
    for k in range(k_max):
        window_size = 2 ** k
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        A[k] = ndimage.convolve(image, kernel, mode='reflect')
    
    E = np.zeros((k_max - 1, h, w))
    for k in range(k_max - 1):
        E[k] = np.abs(A[k] - A[k + 1])
    
    Sbest = np.argmax(E, axis=0)
    coarseness = np.mean(2 ** Sbest)
    
    return coarseness

def tamura_contrast(image):
    image = image.astype(float)
    
    mu4 = np.mean((image - np.mean(image)) ** 4)
    variance = np.var(image)
    
    if variance == 0:
        return 0
    
    alpha4 = mu4 / (variance ** 2)
    contrast = np.sqrt(variance) / (alpha4 ** 0.25) if alpha4 > 0 else 0
    
    return contrast

def tamura_directionality(image, num_bins=16):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle = (angle + 180) % 180
    
    threshold = np.mean(magnitude)
    significant = magnitude > threshold
    
    hist, _ = np.histogram(angle[significant], bins=num_bins, range=(0, 180))
    
    if hist.sum() > 0:
        hist = hist.astype(float) / hist.sum()
    
    uniform_dist = np.ones(num_bins) / num_bins
    directionality = np.sum((hist - uniform_dist) ** 2)
    
    return hist, directionality

def glcm_features(image, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    image_normalized = (image / 16).astype(np.uint8)
    
    glcm = graycomatrix(
        image_normalized,
        distances=distances,
        angles=angles,
        levels=16,
        symmetric=True,
        normed=True
    )
    
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    for prop in properties:
        values = graycoprops(glcm, prop)
        features.append(np.mean(values))
        features.append(np.std(values))
    
    return np.array(features)

def extract_texture_features(image_path):
    gray, _ = load_image(image_path)
    gray = cv2.resize(gray, (256, 256))
    
    gabor_feats = gabor_filters(gray, num_orientations=8, num_scales=4)
    coarseness = tamura_coarseness(gray)
    contrast = tamura_contrast(gray)
    direction_hist, directionality = tamura_directionality(gray)
    glcm_feats = glcm_features(gray)
    
    return {
        'image_name': os.path.basename(image_path),
        'gabor_features': gabor_feats,
        'tamura_coarseness': coarseness,
        'tamura_contrast': contrast,
        'tamura_directionality': directionality,
        'direction_histogram': direction_hist,
        'glcm_features': glcm_feats
    }

def process_all_texture_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
    
    print(f"Processing {len(image_files)} texture images...")
    
    for image_path in sorted(image_files):
        try:
            features = extract_texture_features(str(image_path))
            json_path = os.path.join(output_folder, image_path.stem + '.json')
            save_features_to_json(features, json_path)
            print(f"Processed: {image_path.name}")
        except Exception as e:
            print(f"Error with {image_path.name}: {str(e)}")

