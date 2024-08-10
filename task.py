import matplotlib.pyplot as plt
from skimage import io

image = io.imread('image.jpeg')

plt.imshow(image)
plt.show()
import matplotlib.pyplot as plt
from skimage import io, color, filters

# Load the image
image = io.imread('image.jpeg')

# Convert the image to grayscale
gray_image = color.rgb2gray(image)

# Apply edge detection
edges = filters.sobel(gray_image)

# Display the images
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Grayscale image
plt.subplot(1, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Edge-detected image
plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

plt.show()
