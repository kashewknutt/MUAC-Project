from PIL import Image
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from typing import Tuple, Optional



import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class MUACProcessor:
    def __init__(self, model_path: str, model_type: str = "vit_h"):
        import torch
        self.sam = sam_model_registry[model_type](checkpoint=model_path)
        
        # Move model to CUDA for GPU acceleration
        if torch.cuda.is_available():
            self.sam = self.sam.to(device="cuda")
        else:
            print("WARNING: CUDA not available. Running on CPU, which will be significantly slower.")
        
        self.predictor = SamPredictor(self.sam)
        
    def load_image(self, image_path: str) -> Tuple[np.ndarray, Image.Image]:
        """Load and prepare image for processing."""
        image = Image.open(image_path)
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image_array, image
    
    def segment_body(self, image: np.ndarray) -> np.ndarray:
        """Segment the body from the image using SAM."""
        self.predictor.set_image(image)
        
        # Get center point for initial prediction
        h, w = image.shape[:2]
        center_point = np.array([[w//2, h//2]])
        
        # Predict mask
        masks, _, _ = self.predictor.predict(
            point_coords=center_point,
            point_labels=np.array([1]),
            multimask_output=True
        )
        
        # Select the largest mask (likely to be the body)
        largest_mask = masks[np.argmax([mask.sum() for mask in masks])]
        
        # Apply mask to image
        masked_image = image.copy()
        masked_image[~largest_mask] = 0
        
        return masked_image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Additional preprocessing steps."""
        # Convert to grayscale for better edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to create binary image
        _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
        
        return thresh
    
    def process_image(self, image_path: str, save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Complete image processing pipeline."""
        # Load image
        image_array, original_image = self.load_image(image_path)
        
        # Segment body
        segmented = self.segment_body(image_array)
        
        # Preprocess
        preprocessed = self.preprocess_image(segmented)
        
        if save_path:
            self.save_results(segmented, preprocessed, save_path)
            
        return segmented, preprocessed
    
    def save_results(self, segmented: np.ndarray, preprocessed: np.ndarray, base_path: str):
        """Save processing results."""
        cv2.imwrite(f"{base_path}_segmented.png", segmented)
        cv2.imwrite(f"{base_path}_preprocessed.png", preprocessed)
    
    def visualize_results(self, original: np.ndarray, segmented: np.ndarray, preprocessed: np.ndarray):
        """Visualize processing steps."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original')
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        ax2.set_title('Segmented')
        ax2.axis('off')
        
        ax3.imshow(preprocessed, cmap='gray')
        ax3.set_title('Preprocessed')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Initialize processor
    processor = MUACProcessor(model_path="model/sam_vit_h_4b8939.pth")
    
    # Process image
    image_path = "rgb_images/images_342761ce-8026-11ef-be7f-6bffe21c74a1_102_1.png"
    segmented, preprocessed = processor.process_image(
        image_path, 
        save_path="output/result"
    )
    
    # Visualize results
    original, _ = processor.load_image(image_path)
    processor.visualize_results(original, segmented, preprocessed)

if __name__ == "__main__":
    main()