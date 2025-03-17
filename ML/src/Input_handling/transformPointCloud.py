import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================
# === Configuration Values ===
# ============================

# Path to the input_frame point cloud (.xyz, comma-separated)
FILE_PATH = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\Scan\MeWithIphone2_14_27_37.xyz"

# Output file path to save the transformed SMPL_frame point cloud
OUTPUT_FILE_PATH = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\Scan\transformed_point_cloud.xyz"

# Z cutoff in the input_frame: remove points with z < -1.82 before transformation
Z_THRESHOLD = -1.82

# After transformation, delete points with x > 1 or z > 1.1.
X_CUTOFF = 1.0
Z_CUTOFF_TRANSFORMED = 1.1

# ===== Manual Inputs for the Transformation =====
# Define the mapping from the input_frame to the SMPL_frame.

# In the input_frame (from your iPhone scan):
p_input = np.array([-0.7, 0.9, -1.75])           # Hip center in input_frame
p_input_spine_top = np.array([-1.35, 1.28, -1.75]) # Top of spine in input_frame
v1_input = p_input_spine_top - p_input            # Spine vector in input_frame

# Choose a second vector in the input_frame (must be non-collinear with v1_input).
v2_input = np.array([0.0, 0.0, 1.0])             

# In the SMPL_frame (target frame):
p_SMPL = np.array([0.0, -0.35, 0.0])              # Hip center in SMPL_frame
v1_SMPL = np.array([0.0, 0.0, 1.0])               # Desired spine vector (points along positive z)
v2_SMPL = np.array([0.0, -1.0, 0.0])              # Second vector in SMPL_frame

# ================================
# === Utility Functions        ===
# ================================

def compute_rotation_translation(v1_in, v2_in, p_in, v1_SMPL, v2_SMPL, p_SMPL):
    """
    Computes the rotation matrix R and translation vector t mapping the input_frame to the SMPL_frame.
    
    Constructs an orthonormal basis for each frame from two non-collinear vectors,
    then computes R = B * A^T and t = p_SMPL - R * p_in.
    """
    # Normalize the vectors
    v1_in = v1_in / np.linalg.norm(v1_in)
    v2_in = v2_in / np.linalg.norm(v2_in)
    v1_SMPL = v1_SMPL / np.linalg.norm(v1_SMPL)
    v2_SMPL = v2_SMPL / np.linalg.norm(v2_SMPL)
    
    # Construct orthonormal basis for input_frame:
    a1 = v1_in
    a2 = v2_in - np.dot(v2_in, a1) * a1
    a2 = a2 / np.linalg.norm(a2)
    a3 = np.cross(a1, a2)
    A = np.column_stack((a1, a2, a3))
    
    # Construct orthonormal basis for SMPL_frame:
    b1 = v1_SMPL
    b2 = v2_SMPL - np.dot(v2_SMPL, b1) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    B = np.column_stack((b1, b2, b3))
    
    # Rotation matrix:
    R = B @ A.T
    # Translation:
    t = p_SMPL - R @ p_in
    return R, t

def compute_transformation_matrix(R, t):
    """
    Constructs a 4x4 homogeneous transformation matrix T from R and t.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def transform_point_cloud(points, T):
    """
    Applies the 4x4 transformation T to the point cloud.
    """
    N = points.shape[0]
    points_h = np.hstack((points, np.ones((N, 1))))
    transformed_h = (T @ points_h.T).T
    return transformed_h[:, :3]

# ================================
# === Main Processing Function ===
# ================================

def main():
    # --- Load the .xyz point cloud (comma-separated) ---
    try:
        data = np.loadtxt(FILE_PATH, delimiter=',')
    except Exception as e:
        print("Error reading the file:", e)
        return
    
    if data.ndim != 2 or data.shape[1] < 3:
        print("The file does not contain valid 3D point cloud data.")
        return
    
    # Extract coordinates (first three columns) from the input_frame
    coords = data[:, :3]
    
    # --- Filter: Remove points with z < Z_THRESHOLD in input_frame ---
    valid_idx = coords[:, 2] >= Z_THRESHOLD
    filtered_coords = coords[valid_idx]
    
    # --- Compute the transformation from input_frame to SMPL_frame ---
    R, t = compute_rotation_translation(v1_input, v2_input, p_input, v1_SMPL, v2_SMPL, p_SMPL)
    T = compute_transformation_matrix(R, t)
    print("Rotation Matrix R:\n", R)
    print("Translation Vector t:\n", t)
    print("Transformation Matrix T:\n", T)
    
    # --- Apply transformation ---
    transformed_coords = transform_point_cloud(filtered_coords, T)
    
    # --- Additional Filtering in SMPL_frame ---
    # Remove points with x > 1 or z > 1.1.
    valid_idx2 = (transformed_coords[:, 0] <= 1) & (transformed_coords[:, 2] <= 1.1)
    transformed_coords = transformed_coords[valid_idx2]
    
    # --- Save the transformed point cloud for NN prediction ---
    try:
        np.savetxt(OUTPUT_FILE_PATH, transformed_coords, delimiter=',', fmt="%.6f")
        print(f"Transformed point cloud saved to: {OUTPUT_FILE_PATH}")
    except Exception as e:
        print("Error saving transformed point cloud:", e)
    
    # --- Visualization ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(transformed_coords[:, 0], transformed_coords[:, 1], transformed_coords[:, 2],
               c='g', s=0.1, marker='.')
    ax.set_title("Transformed Point Cloud (SMPL_frame)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

if __name__ == '__main__':
    main()
