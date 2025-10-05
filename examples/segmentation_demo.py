#!/usr/bin/env python3
"""
Example script demonstrating the usage of MBD segmentation.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mc_mbd import segment_image

def create_seeds(image_shape, margin=15, object_size=20):
    """Create seeds for background and multiple objects."""
    height, width = image_shape
    seeds = np.zeros(image_shape, dtype=np.int32)
    
    # Create background seeds (label 1) as a frame
    seeds[margin:-margin, margin:margin*2] = 1  # Left
    seeds[margin:-margin, -margin*2:-margin] = 1  # Right
    seeds[margin:margin*2, margin:-margin] = 1  # Top
    seeds[-margin*2:-margin, margin:-margin] = 1  # Bottom
    
    # Create first object seed (label 2) in the center
    center_y, center_x = height // 2, width // 2
    half_size = object_size // 2
    seeds[center_y-half_size:center_y+half_size,
          center_x-half_size:center_x+half_size] = 2
    
    # Create second object seed (label 3) in the bottom-left quadrant
    quad_y = height * 3 // 4  # 3/4 down the image
    quad_x = width // 4      # 1/4 across the image
    seeds[quad_y-half_size:quad_y+half_size,
          quad_x-half_size:quad_x+half_size] = 3
    
    return seeds

def load_and_preprocess_image(image_path):
    """Load and preprocess an image for segmentation."""
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    
    image = np.array(img, dtype=np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    return image

def visualize_results(image, seeds, segmentation):
    """Visualize the input image, seeds, and segmentation result."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot seeds
    seed_vis = np.ma.masked_where(seeds == 0, seeds)
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(seed_vis, cmap='Set1', alpha=0.7)
    axes[1].set_title('Seeds\n(Red=Background, Green/Blue=Objects)')
    axes[1].axis('off')
    
    # Plot segmentation
    axes[2].imshow(segmentation, cmap='Set1')
    axes[2].set_title('Segmentation Result')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test MBD segmentation on an image')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--margin', type=int, default=15, 
                       help='Margin from border for background seeds (default: 15)')
    parser.add_argument('--object-size', type=int, default=20,
                       help='Size of the object seed region (default: 20)')
    args = parser.parse_args()

    # Load and preprocess the image
    print("Loading image...")
    image = load_and_preprocess_image(args.image_path)
    
    # Create seeds
    print("Creating seeds...")
    seeds = create_seeds(image.shape, margin=args.margin, object_size=args.object_size)
    
    # Run segmentation
    print("Running MBD segmentation...")
    labels = segment_image(image, seeds)
    
    # Verify the results maintain seed labels
    seed_positions = seeds > 0
    assert np.all(labels[seed_positions] == seeds[seed_positions]), \
           "Error: Segmentation did not preserve seed labels!"
    
    # Print some statistics
    print("\nSegmentation Statistics:")
    print(f"Image shape: {image.shape}")
    print(f"Number of regions: {len(np.unique(labels))}")
    for label in np.unique(labels):
        count = np.sum(labels == label)
        percentage = 100 * count / labels.size
        print(f"Label {label}: {count} pixels ({percentage:.1f}%)")
    
    # Visualize the results
    print("\nDisplaying visualization...")
    visualize_results(image, seeds, labels)

if __name__ == "__main__":
    main()