import cv2
import numpy as np
import os

def load_images(folder):
    
    images = []
    filenames = sorted([f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ Error: Could not load {filename}")
        else:
            print(f"✅ Loaded: {filename}, Shape: {img.shape}")
            images.append(img)
    
    return images

def blend(image1, image2, H):
  
    # Warp using  homography matrix
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    warped_image2 = cv2.warpPerspective(image2, H, (w1 + w2, max(h1, h2)))

    # mask for the overlapping region
    mask = np.zeros((h1, w1 + w2), dtype=np.float32)
    mask[0:h2, 0:w2] = 1

    # Blend the images with alpha blending
    blended_image = np.zeros_like(warped_image2, dtype=np.float32)
    blended_image[0:h1, 0:w1] = image1.astype(np.float32)  
    blended_image += warped_image2.astype(np.float32) * mask[..., np.newaxis]

    # Normalize the image that blended
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

    return blended_image

def stitch(img_list):
    
    if len(img_list) < 2:
        print("❌ Error: Need at least 2 images for stitching")
        return None
    
    # built-in OpenCV stitcher 
    try:
        print("Attempting stitching with OpenCV Stitcher...")
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, stitched = stitcher.stitch(img_list)
        
        if status == cv2.Stitcher_OK:
            print("✅ Stitching successful using OpenCV Stitcher")
            return stitched
        else:
            print(f"⚠ OpenCV Stitcher failed with status {status}, trying manual method...")
    except Exception as e:
        print(f"⚠ Error with OpenCV Stitcher: {e}, trying manual method...")
    
    # If built-in stitcher fails,  feature matching and blending
    try:
        print("Attempting manual stitching with feature matching and blending...")
        #  images to grayscale
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        
        #  SIFT detector
        sift = cv2.SIFT_create()
        
        #  first image
        result = img_list[0].copy()
        
        for i in range(1, len(img_list)):
            # Detect keypoints and compute descriptors
            kp1, des1 = sift.detectAndCompute(gray_images[i-1], None)
            kp2, des2 = sift.detectAndCompute(gray_images[i], None)
            
            # Match features using FLANN
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Applyed ratio test 
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
            
            if len(good_matches) < 10:
                print(f"⚠ Not enough good matches between images {i-1} and {i}: {len(good_matches)}")
                continue
            
            # Extract locations of matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography matrix
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                print(f"⚠ Couldn't find homography between images {i-1} and {i}")
                continue
            
            # Blend  using alpha blending
            result = blend(result, img_list[i], H)
        
        print("✅ Manual stitching completed with blending")
        return result
    
    except Exception as e:
        print(f"❌ Manual stitching failed: {e}")
        return None

if __name__ == "__main__":
    image_folder = "sample_images"  # image folder
    images = load_images(image_folder)

    if len(images) > 1:
        result = stitch(images)

        if result is not None:
            # Show result
            cv2.imshow("Stitched Image", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Save result
            cv2.imwrite("images/stitched_output.jpg", result)
            print("✅ Stitched image saved as 'stitched_output.jpg'")
        else:
            print("❌ Stitching failed!")
    else:
        print("❌ Not enough images to stitch. Make sure you have at least 2 images.")