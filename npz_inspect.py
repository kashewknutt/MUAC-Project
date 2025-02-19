import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the .npz file
file_path = "images_86e198c6-802b-11ef-b3e7-8fe23563c53d_100_1.npz"  # Replace with your actual file path
data = np.load(file_path)

# Function to display images properly
def show_image(title, img, cmap=None):
    plt.figure(figsize=(6, 6))
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper color display
    plt.title(title)
    plt.axis("off")
    plt.show()

# Depth maps (grayscale)
depthmap = data["depthmap"].squeeze()  # Remove extra dimension
show_image("Depth Map", depthmap, cmap="viridis")

inpainted_depthmap = data["inpainted_depthmap"].squeeze()
show_image("Inpainted Depth Map", inpainted_depthmap, cmap="viridis")

# RGB Image
rgb_image = data["rgb_image"]  # Already in (H, W, 3) format
show_image("RGB Image", rgb_image)

# Segmentation Mask (grayscale)
segmentation_mask = data["segmentation_mask"].squeeze()  # Remove extra dimension
show_image("Segmentation Mask", segmentation_mask, cmap="gray")

# Connected Component (grayscale)
connected_component = data["connected_component"]  # Already in (H, W) format
show_image("Connected Component", connected_component, cmap="jet")

print("All images displayed successfully!")
