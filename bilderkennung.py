import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_name:str):
    img = cv.imread(image_name, cv.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    else:
        print("file could not be read, check with os.path.exists()")
        return None

def process(img):
    # numpy fourier transform
    f = np.fft.fft2(img)

    # shift zero frequency point from top left corner to center
    fshift = np.fft.fftshift(f)

    #magnitide spectrum?
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    plt.subplot(152),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    fshift[crow-30:crow+31, ccol-30:ccol+31] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    magnitude_spectrum = 20*np.log(np.abs(f_ishift))
    
    plt.subplot(151),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(154),plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(155),plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
    plt.subplot(153),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    
    plt.show()

def process_open_cv(img):
    offset = 10
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    
    # create a mask first, center square is 1, remaining all zeros
    mask = np.ones((rows,cols,2),np.uint8)
    mask[crow-offset:crow+offset, ccol-offset:ccol+offset] = 0

    # x = cv.getGaussianKernel(5,10)

    dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
   

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
    magnitude_spectrum = 20*np.log(cv.magnitude(fshift[:,:,0],fshift[:,:,1]))
    
    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img_back, cmap = 'gray')
    plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

    
    # magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    
    # plt.subplot(121),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()



if __name__ == "__main__":
    # process(read_image("streifen.JPG"))
    process_open_cv(read_image("streifen.JPG"))
    process_open_cv(read_image("einstein.JPG"))



# # Compute the discrete Fourier Transform of the image
# fourier = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
 
# # Shift the zero-frequency component to the center of the spectrum
# fourier_shift = np.fft.fftshift(fourier)
 
# # calculate the magnitude of the Fourier Transform
# magnitude = 20*np.log(cv.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))
 
# # Scale the magnitude for display
# magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
 
# # Display the magnitude of the Fourier Transform
# cv.imshow('Fourier Transform', magnitude)
# cv.waitKey(0)
# cv.destroyAllWindows()