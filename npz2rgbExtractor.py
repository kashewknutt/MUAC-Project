import os
import numpy as np
import cv2


input_folder = "scans-20250225T092603Z-001/scans/34244ba6-8026-11ef-be7d-37d1ef72a9cf/104"
output_folder = "rgb_images"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".npz"):
        file_path = os.path.join(input_folder, filename)
        
        data = np.load(file_path)
        
        if "rgb_image" in data:
            rgb_image = data["rgb_image"]
            
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)
            
            cv2.imwrite(output_path, rgb_image)
            print(f"Saved: {output_path}")
        else:
            print(f"Skipping {filename} (No 'rgb_image' found)")

print("Processing complete! All images saved.")
