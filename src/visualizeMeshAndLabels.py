import trimesh
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# --- Configuration ---
mesh_folder = r"smplx_meshes_test"           # Folder with the mesh OBJ files.
labels_csv_path = r"smplx_labels_test\labels.csv"  # CSV file with label points.
selected_id = 2  # Change this value (0-indexed) to visualize a different mesh/labels.

# --- Read the labels CSV ---
with open(labels_csv_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Read header
    rows = list(reader)

# Validate selected_id.
if selected_id < 0 or selected_id >= len(rows):
    raise ValueError(f"Selected ID ({selected_id}) is out of range. The CSV file contains {len(rows)} data rows.")

row = rows[selected_id]

# The first column is the mesh filename.
mesh_filename = row[0]
mesh_path = os.path.join(mesh_folder, mesh_filename)

# --- Load the Mesh ---
mesh = trimesh.load(mesh_path)
vertices = mesh.vertices  # (N, 3) array

# --- Extract Label Points ---
# We assume each row contains: Filename, then 6 points (each point: x, y, z).
num_points = 6
label_points = []
for i in range(num_points):
    base = 1 + i * 3  # skip the filename column
    x = float(row[base])
    y = float(row[base + 1])
    z = float(row[base + 2])
    label_points.append([x, y, z])
label_points = np.array(label_points)

# --- Visualization using matplotlib ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot mesh vertices in light grey.
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
           s=0.1, color="gray", label="Mesh Vertices")

# Plot the label points in red.
ax.scatter(label_points[:, 0], label_points[:, 1], label_points[:, 2],
           s=50, color="red", marker="o", label="Label Points")

# Annotate each label point with its index.
for i, (x, y, z) in enumerate(label_points):
    ax.text(x, y, z, f"P{i+1}", color="blue", fontsize=10)

# Set all axes from -1 to 1.
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

ax.set_title(f"Mesh and Label Points for ID {selected_id}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()
