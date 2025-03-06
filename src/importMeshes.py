#this is to import meshes in blender
#just open a scripting window in blender and run the script from the text editor
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

# -- 4) Import each .obj using the NEW importer --
for file_name in mesh_files:
    file_path = os.path.join(mesh_folder, file_name)
    print("Importing:", file_path)
    # This is the new C++ OBJ Importer
    bpy.ops.wm.obj_import(filepath=file_path)

print("Import completed!")