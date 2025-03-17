import numpy as np

# ================================
# === Configuration Parameters ===
# ================================
INPUT_XYZ = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\ML\Scan\MeWithIphone\MeWithIphone2_14_27_37.xyz"  # Path to your input .xyz file
OUTPUT_NPY = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\ML\smplx_big_test\presampled_meshes\64_points_from_point_cloud\iphone_without_background.npy"  # Path where the .npy file will be saved
Z_THRESHOLD = -1.82                                   # Remove points with z < -1.82
NUM_DOWNSAMPLED_POINTS = 64                           # Number of points after downsampling

def main():
    # Load the .xyz file (assuming comma-separated values)
    try:
        data = np.loadtxt(INPUT_XYZ, delimiter=',')
    except Exception as e:
        print("Error reading the file:", e)
        return

    # Check that we have at least three columns (x, y, z)
    if data.ndim != 2 or data.shape[1] < 3:
        print("The file does not contain valid 3D point cloud data.")
        return

    # Remove any additional columns (e.g., RGB) and keep only x, y, z
    coords = data[:, :3]

    # Filter out points with z < Z_THRESHOLD
    filtered_coords = coords[coords[:, 2] >= Z_THRESHOLD]
    print(f"Number of points after filtering: {filtered_coords.shape[0]}")

    # Downsample to NUM_DOWNSAMPLED_POINTS points if necessary
    num_points = filtered_coords.shape[0]
    if num_points >= NUM_DOWNSAMPLED_POINTS:
        indices = np.random.choice(num_points, NUM_DOWNSAMPLED_POINTS, replace=False)
        downsampled_coords = filtered_coords[indices]
    else:
        print("Warning: Not enough points after filtering; using all available points.")
        downsampled_coords = filtered_coords

    # Save the downsampled point cloud to a .npy file
    try:
        np.save(OUTPUT_NPY, downsampled_coords)
        print(f"Downsampled point cloud (shape {downsampled_coords.shape}) saved to: {OUTPUT_NPY}")
    except Exception as e:
        print("Error saving the .npy file:", e)

if __name__ == '__main__':
    main()
