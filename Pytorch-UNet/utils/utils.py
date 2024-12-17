import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask):
    """
    Plot an input image alongside its corresponding segmentation masks for each class.

    Args:
        img (numpy.ndarray or torch.Tensor): The input image to display. Shape: (H, W, C) or (H, W).
        mask (numpy.ndarray or torch.Tensor): The segmentation mask, where each pixel's value corresponds 
                                              to a class label. Shape: (H, W).

    The function displays:
    - The original input image.
    - Separate binary masks for each class present in the segmentation mask.

    Example:
        If the mask contains class labels 0, 1, and 2, the function will display the image 
        followed by binary masks for classes 0, 1, and 2.
    """
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()
