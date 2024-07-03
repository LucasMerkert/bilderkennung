import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import os

class FilterType(Enum):
    HIGH = 1
    LOW = 2
    BAND = 3
    INVERSEBAND = 4

def read_image(image_name:str):
    img = cv.imread(image_name, cv.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    else:
        print("file could not be read, check with os.path.exists()")
        return None

def fft_numpy(img):
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

def get_low_pass_mask(img, radius):
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2 <= radius*radius
    mask[mask_area] = 1

    return mask

def get_high_pass_mask(img, radius):
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    mask = np.ones((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2 <= radius*radius
    mask[mask_area] = 0
    
    return mask

def get_band_pass_mask(img, inner_radius, outer_radius):
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and((x - crow)**2 + (y - ccol)**2 >= inner_radius**2,
                            (x - crow)**2 + (y - ccol)**2 <= outer_radius**2)
    mask[mask_area] = 1
    
    return mask

def get_inverse_band_pass_mask(img, inner_radius, outer_radius):
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    mask = np.ones((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and((x - crow)**2 + (y - ccol)**2 >= inner_radius**2,
                            (x - crow)**2 + (y - ccol)**2 <= outer_radius**2)
    mask[mask_area] = 0
    
    return mask

def process_open_cv(img_name, filter_type):

    img = read_image(img_name)

    if filter_type == FilterType.HIGH:
        print("Applying high-pass filter")
        mask = get_high_pass_mask(img,50)
    elif filter_type == FilterType.LOW:
        print("Applying low-pass filter")
        mask = get_low_pass_mask(img,20)
    elif filter_type == FilterType.BAND:
        print("Applying band-pass filter")
        mask = get_band_pass_mask(img,200, 250)
    elif filter_type == FilterType.INVERSEBAND:
        print("Applying band-pass filter")
        mask = get_inverse_band_pass_mask(img,100, 250)
    
    dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
   

    fshift = dft_shift * mask

    # Get the magnitude spectrum of the filtered image
    filtered_magnitude_spectrum = 20 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1)

    # Inverse DFT to get the image back
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize the filtered image for display
    cv.normalize(img_back, img_back, 0, 255, cv.NORM_MINMAX)
    img_back = np.uint8(img_back)

    # Display the results in a 2x2 grid
    plt.figure(figsize=(10, 10))

    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.axis('off')

    plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.axis('off')

    plt.subplot(223), plt.imshow(img_back, cmap='gray')
    plt.title('Filtered Image'), plt.axis('off')

    plt.subplot(224), plt.imshow(filtered_magnitude_spectrum, cmap='gray')
    plt.title('Filtered Magnitude Spectrum'), plt.axis('off')

    image_name = os.path.splitext(os.path.basename(img_name))[0]
    plt.tight_layout()

    if filter_type == FilterType.HIGH:
        output_filename = f'{image_name}_filtered_high.png'
        plt.savefig(output_filename)
    elif filter_type == FilterType.LOW:
        output_filename = f'{image_name}_filtered_low.png'
        plt.savefig(output_filename)
    elif filter_type == FilterType.BAND:
        output_filename = f'{image_name}_filtered_band.png'
        plt.savefig(output_filename)
    elif filter_type == FilterType.INVERSEBAND:
        output_filename = f'{image_name}_filtered_inverse_band.png'
        plt.savefig(output_filename)

    # Save the figure
    output_filename = f'{image_name}_filtered.png'
    plt.savefig(output_filename)

    plt.show()


if __name__ == "__main__":
    # process_open_cv(read_image("blurred_image.JPG"),FilterType.HIGH)
    # process_open_cv(read_image("blurred_image.JPG"),FilterType.LOW)
    # process_open_cv(read_image("blurred_image.JPG"),FilterType.BAND)

    # process_open_cv(read_image("einstein.JPG"),FilterType.HIGH)
    # process_open_cv(read_image("einstein.JPG"),FilterType.LOW)
    # process_open_cv(read_image("einstein.JPG"),FilterType.BAND)

    process_open_cv("Joseph_Fourier.JPG",FilterType.HIGH)
    process_open_cv("Joseph_Fourier.JPG",FilterType.LOW)
    process_open_cv("Joseph_Fourier.JPG",FilterType.BAND)
    process_open_cv("Joseph_Fourier.JPG",FilterType.INVERSEBAND)
