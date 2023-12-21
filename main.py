import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image, showIntermeditaeSteps):
    # Apply Gaussian blur to the input image, to reduce image noise and details
    # https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    blurred_image = cv2.GaussianBlur(image, (5, 7), 0)

    if showIntermeditaeSteps:
        plt.imshow(blurred_image, cmap='gray')
        plt.title('Blurred Image')
        plt.show()



    # calculates the gradient of the image in x and y directions, highlighting areas of rapid intensity change (edges) by computing the approximation of the derivative in each direction
    # https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
    gradient_x = cv2.Sobel(blurred_image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)     # Calculate the gradient in the x direction
    gradient_y = cv2.Sobel(blurred_image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)     # Calculate the gradient in the y direction

    gradient_magnitude = cv2.subtract(gradient_x, gradient_y)                          # Compute the difference between x and y gradients
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)                       # Convert the resulting gradient to an unsigned 8-bit image

    if showIntermeditaeSteps:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title('Gradient Magnitude')
        plt.show()



    # Blur the image to smooth out the image and reduce noise or high-frequency variations
    # https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    blurred_gradient_magnitude = cv2.blur(gradient_magnitude, (7, 9))
    blurred_gradient_magnitude = cv2.medianBlur(blurred_gradient_magnitude, 3)

    if showIntermeditaeSteps:
        plt.imshow(blurred_gradient_magnitude, cmap='gray')
        plt.title('Blurred Gradient Magnitude')
        plt.show()



    # Apply a threshold to create a binary image, where each value above 230 becomes a binary 1 and each value below becomes a 0
    # https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    (_, threshold_image) = cv2.threshold(blurred_gradient_magnitude, 230, 255, cv2.THRESH_BINARY)

    if showIntermeditaeSteps:
        plt.imshow(threshold_image, cmap='gray')
        plt.title('Thresholded Image')
        plt.show()

    return threshold_image


def find_contours(binary_image):
    # identifies the outer contours in a binary image by filling gaps, shrinking foreground objects, enlarging them, and then detecting the contours
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))                    # Define a structuring element for morphological operations (rectangular kernel of size 29x29)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, morph_kernel)                # Perform morphological closing operation to fill gaps in the contours

    closed_image = cv2.erode(closed_image, None, iterations=3)                           # Erode the image to reduce the size of foreground objects
    closed_image = cv2.dilate(closed_image, None, iterations=3)                          # Dilate the eroded image to enlarge the objects

    contours = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]    # Find contours in the processed binary image; RETR_EXTERNAL retrieves only the external contours; CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points

    return contours


def draw_bounding_box(image, contours):
    # https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html

    if contours:
        largest_contour = sorted(contours, key=cv2.contourArea)[len(contours)-1]            # Select the largest contour among the given contours
        min_area_rect = cv2.minAreaRect(largest_contour)                                    # Find the minimum area bounding rectangle for the largest contour
        bounding_box = np.intp(cv2.boxPoints(min_area_rect))                                # Get the bounding box coordinates as integers

        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)                               # Convert the grayscale image to color, to draw Rectangle in Color

        final_image = cv2.drawContours(color_image, [bounding_box], -1, (0, 256, 0), 15)        # Draw the bounding box around the detected object

        cv2.imshow('Detected Barcodes', final_image)
        cv2.waitKey(0)
    else:
        print("No contours found.")

def detect_barcode(image_path):
    showIntermediateStep = False

    image = tifffile.imread(image_path)
    preprocessed_image = preprocess_image(image, showIntermediateStep)
    contours = find_contours(preprocessed_image)
    draw_bounding_box(image, contours)

# insert image Paths
image_paths = ['images/barcode-0.tif', 'images/barcode-1.tif', 'images/barcode-2.tif', 'images/barcode-3.tif']
for path in image_paths:
    detect_barcode(path)
