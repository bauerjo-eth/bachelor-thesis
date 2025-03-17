import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# --- Configuration ---
mesh_folder = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\ML\smplx_big_test\pipeline\final_dataset"  # Folder with the mesh files (.obj or .npy).
labels_csv_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\ML\smplx_big_test\pipeline\final_dataset\labels_transformed.csv"  # CSV file with label points.
selected_id = 5  # Change this value (0-indexed) to visualize a different mesh/labels.

def load_all_vertices_from_obj(obj_path):
    """
    Manually parse the OBJ file to load all vertices (lines starting with 'v ').
    """
    vertices = []
    with open(obj_path, "r") as file:
        for line in file:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x, y, z])
                    except ValueError:
                        continue  # Skip lines that don't parse correctly
    return np.array(vertices)

def load_mesh_vertices(mesh_path):
    """
    Loads the mesh vertices from the given file.
    Supports .obj files (by manual parsing) and .npy files.
    """
    if mesh_path.lower().endswith(".obj"):
        vertices = load_all_vertices_from_obj(mesh_path)
    elif mesh_path.lower().endswith(".npy"):
        vertices = np.load(mesh_path)
    else:
        raise ValueError("Unsupported mesh file format. Please use .obj or .npy.")
    return vertices

# --- Read the labels CSV ---
with open(labels_csv_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Read header
    rows = list(reader)

# Validate selected_id.
if selected_id < 0 or selected_id >= len(rows):
    raise ValueError(f"Selected ID ({selected_id}) is out of range. The CSV file contains {len(rows)} data rows.")

row = rows[selected_id]

# The first column is assumed to be the mesh filename.
mesh_filename = row[0]
mesh_path = os.path.join(mesh_folder, mesh_filename)

# If the file doesn't exist, try replacing .obj with .npy
if not os.path.exists(mesh_path):
    if mesh_filename.lower().endswith(".obj"):
        alt_mesh_filename = mesh_filename.replace(".obj", ".npy")
        alt_mesh_path = os.path.join(mesh_folder, alt_mesh_filename)
        if os.path.exists(alt_mesh_path):
            print(f"File {mesh_filename} not found, loading {alt_mesh_filename} instead.")
            mesh_filename = alt_mesh_filename
            mesh_path = alt_mesh_path
        else:
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}. Also, {alt_mesh_filename} was not found.")
    else:
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

# --- Load the Mesh Vertices ---
vertices = load_mesh_vertices(mesh_path)
print("Number of vertices loaded:", vertices.shape[0])

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
           s=0.1, color="grey", label="Mesh Vertices")

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
