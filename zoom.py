import cv2
import numpy as np

def zoom(image_path, zoom_factor, interpolation=cv2.INTER_CUBIC):
    
    # Validate zoom factor
    if zoom_factor not in [10, 20]:
        raise ValueError("Zoom factor must be either 10 or 20")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Get original dimensions
    height, width = image.shape[:2]
    
    # Calculate new dimensions
    new_height = int(height * zoom_factor)
    new_width = int(width * zoom_factor)
    
    # Apply zoom using specified interpolation method
    zoomed_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    
    return zoomed_image

def selective_zoom(image_path, x, y, width, height, zoom_factor, interpolation=cv2.INTER_CUBIC):
   
    # Validate zoom factor
    if zoom_factor not in [10, 20]:
        raise ValueError("Zoom factor must be either 10 or 20")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Extract the region to zoom
    region = image[y:y+height, x:x+width]
    
    # Apply zoom to the region
    zoomed_region = cv2.resize(region, (width * zoom_factor, height * zoom_factor), interpolation=interpolation)
    
    return zoomed_region

def smooth_zoom(image_path, zoom_factor, steps=5):
    
    # Validate zoom factor
    if zoom_factor not in [10, 20]:
        raise ValueError("Zoom factor must be either 10 or 20")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # zoom step
    zoom_step = zoom_factor ** (1 / steps)
    
    # original dimensions
    height, width = image.shape[:2]
    
    # Current image
    current_image = image.copy()
    
    #  zoom in steps
    for _ in range(steps):
        # new dimensions
        new_height = int(current_image.shape[0] * zoom_step)
        new_width = int(current_image.shape[1] * zoom_step)
        
       
        current_image = cv2.resize(current_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return current_image