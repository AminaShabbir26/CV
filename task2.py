import matplotlib.pyplot as plt
from skimage import io

image = io.imread('image.jpg')

plt.imshow(image)
plt.show()

import matplotlib.pyplot as plt
from skimage import io, color, filters, feature, measure, morphology

# Load the image
image = io.imread('image.jpg')

# Step 1: Convert to Grayscale
gray_image = color.rgb2gray(image)

# Step 2: Apply Gaussian Blur
blurred_image = filters.gaussian(gray_image, sigma=1)

# Step 3: Edge Detection using Sobel
edges = filters.sobel(blurred_image)

# Step 4: Image Thresholding
threshold_value = filters.threshold_otsu(blurred_image)
binary_image = blurred_image > threshold_value

# Step 5: Contour Detection
contours = measure.find_contours(binary_image, level=0.8)

# Plotting the results
plt.figure(figsize=(20, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Grayscale Image
plt.subplot(2, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Blurred Image
plt.subplot(2, 3, 3)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

# Edge-detected Image
plt.subplot(2, 3, 4)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

# Thresholded Image
plt.subplot(2, 3, 5)
plt.imshow(binary_image, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')

# Contour Detection
plt.subplot(2, 3, 6)
plt.imshow(binary_image, cmap='gray')
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
plt.title('Contour Detection')
plt.axis('off')

plt.show()
