import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_low_pass_filter(img, radius):
    # Apply FFT
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a low-pass filter mask
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1

    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

    return img_back

def apply_high_pass_filter(img, radius):
    # Apply FFT
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a high-pass filter mask
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0

    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

    return img_back

# Load the images
img1 = cv2.imread('monkey.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('tiger.jpg', cv2.IMREAD_GRAYSCALE)

# Ensure images are the same size
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Apply filters
low_pass_img = apply_low_pass_filter(img1, 30)  # Adjust the radius as needed
high_pass_img = apply_high_pass_filter(img2, 10)  # Adjust the radius as needed

# Normalize the images for better visualization
# low_pass_img = cv2.normalize(low_pass_img, None, 0, 255, cv2.NORM_MINMAX)
# high_pass_img = cv2.normalize(high_pass_img, None, 0, 255, cv2.NORM_MINMAX)

# Combine the images
combined_img = cv2.addWeighted(low_pass_img, 0.5, high_pass_img, 0.5, 0)

# Display the results
plt.figure(figsize=(10, 10))
plt.subplot(2,2,1),plt.imshow(img1, cmap = 'gray'),plt.title('Original Image 1')
plt.subplot(2,2,2),plt.imshow(img2, cmap = 'gray'),plt.title('Original Image 2')
plt.subplot(2,2,3),plt.imshow(low_pass_img, cmap = 'gray'),plt.title('Low Pass Filtered Image')
plt.subplot(2,2,4),plt.imshow(high_pass_img, cmap = 'gray'),plt.title('High Pass Filtered Image')
plt.figure(figsize=(5, 5))
plt.imshow(combined_img, cmap = 'gray'),plt.title('Combined Image')
plt.show()
