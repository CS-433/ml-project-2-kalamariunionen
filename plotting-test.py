

import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv

#Plot R vs G vs 
path_df = '/Volumes/T7 Shield/AntProject/colour_ants.csv'

def open_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


def convert_values_to_numpy(output_tensor):
    output_np = []
    for batch in output_tensor:
        for value in batch:
            output_np.append(value.numpy())

    return np.array(output_np)

file_path = "output_training/output_resnet_1layer_10epochs/target_colors.pkl"
target_colors = open_pickle(file_path)

file_path = "output_training/output_resnet_1layer_10epochs/output_colors.pkl"
output_colors = open_pickle(file_path)

target_colors_np = convert_values_to_numpy(target_colors)
output_colors_np = convert_values_to_numpy(output_colors)

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

colors = list(zip(target_colors_np[:,0], target_colors_np[:,1], target_colors_np[:,2]))

# Creating plot
ax.scatter3D(target_colors_np[:,0], target_colors_np[:,1], target_colors_np[:,2], color = colors)
plt.title("simple 3D scatter plot")
 
# show plot
plt.show()


out_of_bounds = (output_colors_np < 0) | (output_colors_np > 1)

if np.any(out_of_bounds):
    print("Some values are out of bounds. Clipping them to the range [0, 1].")
    output_colors_np = np.clip(output_colors_np, 0, 1)
else:
    print("All values are within the range [0, 1].")


fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

colors = list(zip(output_colors_np[:,0], output_colors_np[:,1], output_colors_np[:,2]))

# Creating plot
ax.scatter3D(output_colors_np[:,0], output_colors_np[:,1], output_colors_np[:,2], color = colors)
plt.title("simple 3D scatter plot")
 
# show plot
plt.show()

"""

# Read the Excel file
df = pd.read_csv(path_df)
df.head()

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

colors = list(zip(df['r_thorax']/255, df['g_thorax']/255, df['b_thorax']/255))

 
# Creating plot
ax.scatter3D(df['r_thorax'], df['g_thorax'], df['b_thorax'], color = colors)
plt.title("simple 3D scatter plot")
 
# show plot
plt.show()


rgb_normalized = np.array(list(zip(df['r_thorax']/255, df['g_thorax']/255, df['b_thorax']/255)))
hsv_normalized = rgb_to_hsv(rgb_normalized)

# Re-normalize HSV values to RGB for plotting
hsv_colors = [tuple(c) for c in hsv_normalized]

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

ax.scatter3D(df['r_thorax'], df['g_thorax'], df['b_thorax'], color=hsv_colors)
plt.title("3D Scatter Plot with HSV Colors")

# Show plot
plt.show()

"""