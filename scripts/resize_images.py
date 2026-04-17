import os
import cv2
import argparse
from tqdm import tqdm

# =================================================================
# SCRIPT: Image Standardization Utility
# PURPOSE: Resizes raw agricultural datasets to a fixed input size 
#          (default 224x224) for Neural Network compatibility.
# =================================================================

def resize_images(input_dir, output_dir, size=(224, 224)):
    """
    Batch processes images in a directory.
    - input_dir: Path to raw high-res images
    - output_dir: Path to save standardized images
    - size: Target resolution (Width, Height)
    """
    # Initialize the output directory structure if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Filter for standard image formats
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"PROCESS: Standardizing {len(image_files)} specimens...")
    
    # Progress-tracked loop for batch processing
    for filename in tqdm(image_files):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            # INTER_AREA interpolation is preferred for downsampling
            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            
            # Flush standardized specimen to the output path
            cv2.imwrite(os.path.join(output_dir, filename), resized_img)
        else:
            print(f"ERROR: Specimen {filename} is corrupt or unreadable.")

if __name__ == "__main__":
    # Command Line Interface Setup
    parser = argparse.ArgumentParser(description="AI Agronomist | Dataset Standardization Tool")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing raw images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed results")
    parser.add_argument("--width", type=int, default=224, help="Target pixel width")
    parser.add_argument("--height", type=int, default=224, help="Target pixel height")
    
    args = parser.parse_args()
    
    # Execute the pipeline
    resize_images(args.input, args.output, size=(args.width, args.height))
