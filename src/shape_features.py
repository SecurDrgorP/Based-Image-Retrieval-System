
import cv2
import numpy as np
import os
from pathlib import Path

def extract_contour(gray_image):
    _, binary = cv2.threshold(gray_image, 127, 255, 
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return None
    
    return max(contours, key=len)

def fourier_descriptors(contour, num_descriptors=20):
    if contour is None or len(contour) < 3:
        return np.zeros(num_descriptors)
    
    contour_complex = np.empty(contour.shape[:-1], dtype=complex)
    contour_complex.real = contour[:, 0, 0]
    contour_complex.imag = contour[:, 0, 1]
    
    fourier_result = np.fft.fft(contour_complex)
    fourier_magnitudes = np.abs(fourier_result)
    
    if fourier_magnitudes[0] != 0:
        fourier_magnitudes = fourier_magnitudes / fourier_magnitudes[0]
    
    descriptors = fourier_magnitudes[1:num_descriptors+1]
    
    if len(descriptors) < num_descriptors:
        descriptors = np.pad(descriptors, (0, num_descriptors - len(descriptors)))
    
    return descriptors

def edge_direction_histogram(contour, num_bins=36):
    if contour is None or len(contour) < 2:
        return np.zeros(num_bins)
    
    contour_points = contour[:, 0, :]
    dx = np.diff(contour_points[:, 0])
    dy = np.diff(contour_points[:, 1])
    
    angles = np.arctan2(dy, dx) * 180 / np.pi
    angles = (angles + 360) % 360
    
    histogram, _ = np.histogram(angles, bins=num_bins, range=(0, 360))
    
    if histogram.sum() > 0:
        histogram = histogram.astype(float) / histogram.sum()
    
    return histogram

def extract_shape_features(image_path, num_fourier=20, num_direction_bins=36):
    gray, _ = load_image(image_path)
    contour = extract_contour(gray)
    fourier_desc = fourier_descriptors(contour, num_fourier)
    direction_hist = edge_direction_histogram(contour, num_direction_bins)
    
    moments = cv2.moments(contour) if contour is not None else {}
    hu_moments = cv2.HuMoments(moments).flatten() if moments else np.zeros(7)
    
    return {
        'image_name': os.path.basename(image_path),
        'fourier_descriptors': fourier_desc,
        'direction_histogram': direction_hist,
        'hu_moments': np.log(np.abs(hu_moments) + 1e-10)
    }

def process_all_shape_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = []
    for ext in ['.gif', '.png', '.jpg', '.jpeg']:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
    
    print(f"Processing {len(image_files)} shape images...")
    
    for image_path in sorted(image_files):
        try:
            features = extract_shape_features(str(image_path))
            json_path = os.path.join(output_folder, image_path.stem + '.json')
            save_features_to_json(features, json_path)
            print(f"Processed: {image_path.name}")
        except Exception as e:
            print(f"Error with {image_path.name}: {str(e)}")
