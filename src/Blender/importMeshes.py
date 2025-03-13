import bpy
import os

# -- 1) Update the path to your .obj folder --
mesh_folder = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_meshes_test"

# -- 2) (Optional) Clear the current scene --
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# -- 3) Find all .obj files in the folder --
mesh_files = [f for f in os.listdir(mesh_folder) if f.lower().endswith('.obj')]
print(f"Found {len(mesh_files)} OBJ file(s).")

# Define the offset distance (adjust as needed)
offset_distance = 2.0

# -- 4) Import each .obj and offset them --
for i, file_name in enumerate(mesh_files):
    file_path = os.path.join(mesh_folder, file_name)
    print("Importing:", file_path)
    
    # Import the .obj file using the new importer
    bpy.ops.wm.obj_import(filepath=file_path)
    
    # Retrieve the newly imported objects (they are selected automatically)
    imported_objs = bpy.context.selected_objects
    
    # Move the objects by an offset based on the index to avoid collision
    for obj in imported_objs:
        obj.location.x += i * offset_distance
        # Optionally, adjust y or z coordinates too:
        # obj.location.y += i * offset_distance

    # Deselect all objects to prepare for the next import
    bpy.ops.object.select_all(action='DESELECT')

print("Import completed!")
