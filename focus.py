import cv2
import numpy as np
from skimage import exposure

def auto_foc(image_path):
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale for sharpness measurement
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Measure initial sharpness using Laplacian variance
    initial_sharpness = measure_sharp(gray)
    
    # Apply sharpening
    sharpened_image = sharpen_images(image)
    
    # Measure final sharpness
    sharpened_gray = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
    final_sharpness = measure_sharp(sharpened_gray)
    
    # If sharpening improved the image, return it
    if final_sharpness > initial_sharpness:
        return sharpened_image
    
    # Otherwise, try contrast enhancement
    enhanced_image = contrast_enhance(image)
    
    # Measure enhanced sharpness
    enhanced_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    enhanced_sharpness = measure_sharp(enhanced_gray)
    
    # Return the best result
    if enhanced_sharpness > max(initial_sharpness, final_sharpness):
        return enhanced_image
    elif final_sharpness > initial_sharpness:
        return sharpened_image
    else:
        return image

def measure_sharp(gray_image):
    
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    
    # Calculate variance
    sharpness = laplacian.var()
    
    return sharpness

def sharpen_images(image):
   
    # Create a sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # Apply the kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened

def contrast_enhance(image):
   
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge the channels back
    lab = cv2.merge((l, a, b))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def stack_focus(image_paths):
    
    if not image_paths:
        raise ValueError("No images provided for focus stacking")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        raise ValueError(f"Could not read image at {image_paths[0]}")
    
    # Initialize result with zeros
    result = np.zeros_like(first_image, dtype=np.float32)
    
    # Initialize weights
    weight_sum = np.zeros(first_image.shape[:2], dtype=np.float32)
    
    # Process each image
    for path in image_paths:
        # Read image
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read image at {path}")
            continue
        
        # Resize if dimensions don't match
        if img.shape != first_image.shape:
            img = cv2.resize(img, (first_image.shape[1], first_image.shape[0]))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness map using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_map = np.abs(laplacian)
        
        # Normalize sharpness map
        sharpness_map = cv2.normalize(sharpness_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Expand dimensions for broadcasting
        sharpness_map = np.expand_dims(sharpness_map, axis=-1)
        
        # Accumulate the weighted image
        result += img.astype(np.float32) * sharpness_map
        
        # Accumulate the weights
        weight_sum += sharpness_map.squeeze()
    
    # Avoid division by zero
    weight_sum[weight_sum == 0] = 1
    
    # Normalize the result
    result = result / np.expand_dims(weight_sum, axis=-1)
    
    # Convert to uint8
    result = np.uint8(result)
    
    return result


