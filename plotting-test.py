

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv

#Plot R vs G vs 
path_df = '/Volumes/T7 Shield/AntProject/colour_ants.csv'

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