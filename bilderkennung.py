import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("streifen.JPG", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# # numpy fourier transform
# f = np.fft.fft2(img)

# # shift zero frequency point from top left corner to center
# fshift = np.fft.fftshift(f)

# #magnitide spectrum?
# magnitude_spectrum = 20*np.log(np.abs(fshift))

# plt.subplots
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

# Compute the discrete Fourier Transform of the image
fourier = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
 
# Shift the zero-frequency component to the center of the spectrum
fourier_shift = np.fft.fftshift(fourier)
 
# calculate the magnitude of the Fourier Transform
magnitude = 20*np.log(cv.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))
 
# Scale the magnitude for display
magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
 
# Display the magnitude of the Fourier Transform
cv.imshow('Fourier Transform', magnitude)
cv.waitKey(0)
cv.destroyAllWindows()