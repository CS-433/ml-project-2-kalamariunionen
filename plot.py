
import matplotlib.pyplot as plt
import numpy as np

def plot_data_example(image, label):
    """
    
    """
    
    img_np = image.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_np)
    ax[0].set_title("Image")
    ax[0].axis("off")

    rgb_image = np.ones((10, 10, 3), dtype=int) * label.numpy() / 255 #For normalization

    ax[1].imshow(rgb_image)
    ax[1].set_title("Color")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_color_result(output_colors,target_colors):

    for i in range(0,1):

        # Create a blank image (100x100 pixels) with the RGB values
        output_image = np.zeros((100, 100, 3), dtype=np.uint8)
        output_image[:, :, 0] = output_colors[i][0,0]
        output_image[:, :, 1] = output_colors[i][0,1]
        output_image[:, :, 2] = output_colors[i][0,2]

        target_image = np.zeros((100, 100, 3), dtype=np.uint8)
        target_image[:, :, 0] = target_colors[i][0,0]
        target_image[:, :, 1] = target_colors[i][0,1]
        target_image[:, :, 2] = target_colors[i][0,2]

        # Plot the images
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot predicted RGB image
        axes[0].imshow(output_image)
        axes[0].set_title("Predicted RGB Image")
        axes[0].axis('off')  # Hide axes

        # Plot target RGB image
        axes[1].imshow(target_image)
        axes[1].set_title("Target RGB Image")
        axes[1].axis('off')  # Hide axes

        plt.show()