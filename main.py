import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Install dependencies
# pip install opencv-python numpy matplotlib

def convert_to_3d(image_path):
    # Read the 2D image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create a 3D surface plot using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate X and Y coordinates based on the image size
    x = np.arange(0, img.shape[1], 1)
    y = np.arange(0, img.shape[0], 1)
    x, y = np.meshgrid(x, y)

    # Normalize pixel values to the range [0, 1]
    z = img / 255.0

    # Extrude the image based on pixel values
    ax.plot_surface(x, y, z, cmap='viridis')

    # Show the 3D plot
    plt.show()

# Example usage
convert_to_3d('2d_Avatar.png')
