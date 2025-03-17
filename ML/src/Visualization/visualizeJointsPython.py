import os
import csv
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# --- Configuration ---
mesh_output_folder = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_final\smplx_big_meshes"
joints_csv_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_final\smplx_big_joints\joints.csv"
# Set the mesh number you want to visualise (1-indexed)
selected_mesh_number = 1

# --- Load CSV Data ---
# Read all rows from the CSV file.
with open(joints_csv_path, 'r', newline='') as csvfile:
    reader = list(csv.DictReader(csvfile))
    if selected_mesh_number < 1 or selected_mesh_number > len(reader):
        raise ValueError("Selected mesh number is out of range.")
    # Select the row corresponding to the chosen mesh.
    row = reader[selected_mesh_number - 1]

# --- Construct the Mesh Path ---
# The CSV contains only the basename of the mesh file.
mesh_filename = row["filename"]
mesh_path = os.path.join(mesh_output_folder, mesh_filename)

# --- Load the Mesh ---
mesh = trimesh.load(mesh_path)
vertices = mesh.vertices  # (N, 3) array of vertex coordinates

# --- Extract Joint Coordinates from CSV ---
# The CSV header has "filename", "gender", then "joint1_x", "joint1_y", "joint1_z", "joint2_x", etc.
# We extract the joint coordinates by iterating over the joint numbers.
joint_keys = [key for key in row.keys() if key.startswith("joint") and key.endswith("_x")]
n_joints = len(joint_keys)

joints_list = []
for j in range(1, n_joints + 1):
    x = float(row[f"joint{j}_x"])
    y = float(row[f"joint{j}_y"])
    z = float(row[f"joint{j}_z"])
    joints_list.append([x, y, z])
joints = np.array(joints_list)

# --- Visualisation ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the mesh vertices in light grey.
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
           s=0.1, color="gray", label="Mesh Vertices")

# Plot the joints in red with a larger marker.
ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
           s=50, color="red", marker="o", label="Joints")

# Annotate each joint with its index.
for j, (x, y, z) in enumerate(joints_list):
    ax.text(x, y, z, str(j), color="blue", fontsize=10)

# Optionally, set the axes limits (adjust as needed).
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# Set labels and title.
ax.set_title(f"SMPL-X Mesh and Joint Coordinates (Mesh {selected_mesh_number})")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.show()
