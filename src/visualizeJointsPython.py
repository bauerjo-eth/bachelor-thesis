import trimesh
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# --- Configuration ---
# Adjust these paths to where your mesh and joint CSV are saved.
mesh_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_output\smplx_mesh.obj"      # Change to your actual path.
joints_csv_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_output\smplx_joints.csv"  # Change to your actual path.

# --- Load the Mesh ---
mesh = trimesh.load(mesh_path)
vertices = mesh.vertices  # (N, 3) array of vertex coordinates

# --- Load Joint Coordinates (only joints 0 to 24) ---
joints = []
joint_indices = []
with open(joints_csv_path, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        idx = int(row["joint_index"])
        if 0 <= idx <= 24:
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            joints.append([x, y, z])
            joint_indices.append(idx)
joints = np.array(joints)

# --- Visualization ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the mesh vertices (in light grey)
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
           s=0.1, color="gray", label="Mesh Vertices")

# Plot the joints (in red) with a larger marker size
ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
           s=50, color="red", marker="o", label="Joints")

# Annotate each joint with its index (from the CSV)
for idx, (x, y, z) in zip(joint_indices, joints):
    ax.text(x, y, z, str(idx), color="blue", fontsize=10)

# Set all axes from -1 to 1.
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

ax.set_title("SMPL-X Mesh and Joint Coordinates (Joints 0 to 24)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()
