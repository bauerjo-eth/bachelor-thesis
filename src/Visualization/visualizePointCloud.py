import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================
# === Configuration Values ===
# ============================

# Set the file path directly here (update this to your file location)
FILE_PATH = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\Scan\MeWithIphone\MeWithIphone2_14_27_37.xyz"

def visualize_xyz(file_path):
    """
    Load an .xyz point cloud file with comma-separated values and display it in a 3D scatter plot.
    
    If the file contains more than 3 columns, it assumes that columns 4-6 represent color data.
    """
    try:
        # Use delimiter=',' since your file uses commas
        data = np.loadtxt(file_path, delimiter=',')
    except Exception as e:
        print("Error reading the file:", e)
        return

    if data.ndim != 2 or data.shape[1] < 3:
        print("The file does not appear to contain valid 3D point cloud data.")
        return

    # Extract x, y, and z coordinates (assumes first three columns are x, y, z)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # If there are at least 6 columns, assume the next three are r, g, b.
    if data.shape[1] >= 6:
        colors = data[:, 3:6]
        # Normalize colors if needed (assumes values >1 imply 0-255 range)
        if colors.max() > 1.0:
            colors = colors / 255.0
    else:
        colors = 'b'  # default to blue if no color data

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=colors, s=0.2, marker='.')
    ax.set_title("3D Point Cloud Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Optionally, set equal scaling for all axes.
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()

if __name__ == '__main__':
    visualize_xyz(FILE_PATH)
