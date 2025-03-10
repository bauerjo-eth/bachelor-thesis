import bpy
import csv
import os

# --- Configuration ---
# Set the absolute paths to the OBJ mesh and the CSV file with joint coordinates.
mesh_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_output\smplx_mesh.obj"      # Change to your actual path.
joints_csv_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_output\smplx_joints.csv"  # Change to your actual path.

# --- Import the OBJ Mesh ---
bpy.ops.wm.obj_import(filepath=mesh_path)

# --- Read Joint Coordinates from CSV and apply transformation ---
# Transformation: (x, y, z) from CSV (matplotlib coordinate system) is converted to:
# new_x = x, new_y = -z, new_z = y (to match Blender's coordinate system).
joint_coords = []
with open(joints_csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        idx = int(row["joint_index"])
        x = float(row["x"])
        y = float(row["y"])
        z = float(row["z"])
        # Apply transformation
        transformed = (x, -z, y)
        joint_coords.append((idx, transformed))

# --- Create Empties (or text labels) for each joint ---
for idx, coord in joint_coords:
    empty = bpy.data.objects.new(name=f"Joint_{idx}", object_data=None)
    empty.location = coord
    bpy.context.collection.objects.link(empty)
    
    # Optionally, create a text object for a visual label:
    """
    bpy.ops.object.text_add(location=coord)
    text_obj = bpy.context.object
    text_obj.data.body = f"Joint {idx}"
    text_obj.name = f"JointLabel_{idx}"
    """

print("Mesh imported and joint labels created with corrected coordinates.")
