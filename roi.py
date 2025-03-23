import cv2
import numpy as np

def roi_extract(image_path, x, y, width, height):
 
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    #image dimensions
    img_height, img_width = image.shape[:2]
    
    # ROI coordinates
    if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
        # If coordinates are out of bounds, adjust them
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        width = min(width, img_width - x)
        height = min(height, img_height - y)
        
        if width <= 0 or height <= 0:
            raise ValueError("Invalid ROI dimensions after adjustment")
    
    # Extract ROI
    roi = image[y:y+height, x:x+width]
    
    #  visualization of the ROI selection
    visualization = image.copy()
    cv2.rectangle(visualization, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    
    
    return roi

def detect_roi(image_path, threshold=127):
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #  threshold to get binary image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # If no contours found, use the entire image
        return 0, 0, image.shape[1], image.shape[0]
    
    #  largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # bounding rectangle
    x, y, width, height = cv2.boundingRect(largest_contour)
    
    return x, y, width, height