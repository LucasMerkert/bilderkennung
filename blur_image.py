import cv2

# Step 1: Read the image
image = cv2.imread('streifen.png')

# Check if the image was successfully read
if image is None:
    print("Error: Could not read the image.")
    exit()

# Step 2: Apply Gaussian blur
# You can adjust the kernel size (e.g., (15, 15)) and the sigma (e.g., 0) to control the blur intensity
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

# Step 3: Save the blurred image
cv2.imwrite('blurred_image.jpg', blurred_image)

print("Blurred image saved successfully.")