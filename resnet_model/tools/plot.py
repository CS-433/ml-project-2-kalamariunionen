
import matplotlib.pyplot as plt
import numpy as np

"""
Update this code to plots for ResNET
"""

def plot_rgb_blocks(color_list, title, max_colors=None):
    '''
    Plot specified number of blocks of RGB values
    '''
    if max_colors is not None:
        color_list = color_list[:max_colors]  # Slice the list to include only the specified number of colors

    plt.figure(figsize=(max(32, len(color_list)), 4)) 
    for i, color in enumerate(color_list):
        plt.fill_between([i, i + 1], 0, 1, color=color)
        
    plt.xlim(0, len(color_list))
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title(title, fontsize=30)
    plt.show()


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

def plot_ant_colors(df):

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    colors = list(zip(df['r_thorax']/255, df['g_thorax']/255, df['b_thorax']/255))

    
    # Creating plot
    ax.scatter3D(df['r_thorax'], df['g_thorax'], df['b_thorax'], color = colors)
    plt.title("simple 3D scatter plot")
    
    # show plot
    plt.show()

def plot_results(color_array):
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    colors = list(zip(color_array[:,0], color_array[:,1], color_array[:,2]))

    # Creating plot
    ax.scatter3D(color_array[:,0], color_array[:,1], color_array[:,2], color = colors)
    plt.title("simple 3D scatter plot")
    
    # show plot
    plt.show()

def plot_colors_with_subplots(target_colors_np, output_colors_np, indices):
    # Initialize lists for target and output color rows
    all_target_colors = []
    all_output_colors = []

    for idx in indices:
        target_block = np.ones((10, 10, 3), dtype=float) * target_colors_np[idx]
        output_block = np.ones((10, 10, 3), dtype=float) * output_colors_np[idx]

        all_target_colors.append(target_block)
        all_output_colors.append(output_block)

    # Combine all blocks for each row into horizontal arrays
    target_row = np.hstack(all_target_colors) 
    output_row = np.hstack(all_output_colors) 

    # Create the figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(len(indices), 4))
    # Plot target colors in the first subplot
    axs[0].imshow(target_row)
    axs[0].axis("off")
    axs[0].set_title("Target Colors")
    # Plot output colors in the second subplot
    axs[1].imshow(output_row)
    axs[1].axis("off")
    axs[1].set_title("Output Colors")

    plt.tight_layout()
    plt.show()