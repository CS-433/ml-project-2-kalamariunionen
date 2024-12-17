import cv2
import numpy as np

"""
Code to detect object and crop according to outlines of object
https://medium.com/@mdmarjanhossain0/object-detection-and-cropped-objects-4ec0a4816561

Used to get head, thorax and abdomen of image
"""


def crop_image(image_paths,input_dir,output_dir):
    """
    The function processes images in the input directory, crops them to 
    include only the object, and saves the cropped images in the output directory.

    Parameters:
        image_paths (list): A list of image file names to process.
        input_dir (str): The directory containing the image files.
        output_dir (str): The directory where the cropped images will be saved.

    """
    for image_path in image_paths:
        input_dir_image = input_dir + image_path
        input_image = cv2.imread(input_dir_image, cv2.IMREAD_UNCHANGED)

        # Checking image is grayscale or not. If image shape is 2 then gray scale otherwise not
        if len(input_image.shape) == 2:
            gray_input_image = input_image.copy()
        else:
            # Converting BGR image to grayscale image
            gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        upper_threshold, thresh_input_image = cv2.threshold(
            gray_input_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Calculate lower threshold
        lower_threshold = 0.5 * upper_threshold

        # Apply canny edge detection
        canny = cv2.Canny(input_image, lower_threshold, upper_threshold)
        # Finding the non-zero points of canny
        pts = np.argwhere(canny > 0)

        # Finding the min and max points
        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)

        # Crop ROI from the givn image
        output_image = input_image[y1:y2, x1:x2]

        output_dir_image = output_dir + image_path

        cv2.imwrite(output_dir_image, output_image)