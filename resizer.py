import os
import torch
import torch.nn.functional as F
from torchvision.io import write_png
import tifffile
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Configuration
INPUT_DIRS = ["path/to/input_dir1", "path/to/input_dir2"]  # Replace with your input directories
OUTPUT_DIRS = ["path/to/output_dir1", "path/to/output_dir2"]  # Replace with your output directories
TARGET_SIZE = (512, 512)
NUM_WORKERS = 10  # Adjust based on your system resources

# Set device (automatically uses GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_image(image_path):
    """Load image from file, supporting both TIF and PNG formats"""
    try:
        if image_path.lower().endswith(('.tif', '.tiff')):
            return tifffile.imread(image_path)
        else:  # For PNG files
            with Image.open(image_path) as img:
                return np.array(img)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def process_image_batch(batch_files, input_dir, output_dir):
    """Process a batch of images for better GPU utilization"""
    batch_tensors = []
    valid_files = []
    
    # Load and prepare batch
    for filename in batch_files:
        img = load_image(os.path.join(input_dir, filename))
        if img is not None:
            # Convert to tensor and move to GPU
            img_tensor = torch.tensor(img, dtype=torch.float32, device=device)
            
            # Handle grayscale vs color images
            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0)  # Shape: [1, H, W]
            else:
                img_tensor = img_tensor.permute(2, 0, 1)  # Shape: [C, H, W]
            
            batch_tensors.append(img_tensor)
            valid_files.append(filename)
    
    if not batch_tensors:
        return
    
    # Stack and resize all images in the batch at once
    batch = torch.stack(batch_tensors)
    resized_batch = F.interpolate(
        batch,
        size=TARGET_SIZE,
        mode="bilinear",
        align_corners=False
    )
    
    # Save each image in the batch
    for i, resized_img in enumerate(resized_batch):
        output_filename = os.path.splitext(valid_files[i])[0] + ".png"
        output_path = os.path.join(output_dir, output_filename)
        write_png(resized_img.byte().cpu(), output_path)

def process_directory(input_dir, output_dir, batch_size=32):
    """Process all images in a directory with batching"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files (TIF and PNG)
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.tif', '.tiff', '.png'))]
    
    print(f"Processing {len(image_files)} files from {input_dir}")
    
    # Process in batches
    for i in tqdm(range(0, len(image_files), batch_size),
                 desc=f"Processing {os.path.basename(input_dir)}"):
        batch_files = image_files[i:i + batch_size]
        process_image_batch(batch_files, input_dir, output_dir)

if __name__ == "__main__":
    # Process all directories
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(process_directory, INPUT_DIRS, OUTPUT_DIRS)
    
    print("All images processed successfully.")
