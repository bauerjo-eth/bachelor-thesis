import bpy
import os
import csv
from bpy_extras import view3d_utils
import mathutils

def get_3d_view_context():
    """Find a 3D view context override."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        rv3d = area.spaces.active.region_3d
                        return {'window': window,
                                'screen': window.screen,
                                'area': area,
                                'region': region,
                                'region_data': rv3d,
                                'scene': bpy.context.scene}
    return None

class OBJECT_OT_LabelPoints(bpy.types.Operator):
    """Right-click on 6 points on the mesh to label it.
After 6 points, they are saved to a CSV file and the next file is loaded."""
    bl_idname = "object.label_points"
    bl_label = "Label Points on Mesh"

    def __init__(self):
        self.obj_files = []         # List of full paths to OBJ files
        self.current_index = 0      # Index of the current OBJ file
        self.points = []            # List to store 6 clicked points (mathutils.Vector)
        self.folder_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_meshes"  # Folder with OBJ files
        self.output_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_labels\labels.csv"
        self.override_context = None

    def load_obj_files(self):
        """Load all .obj file paths from the folder and sort them."""
        self.obj_files = [os.path.join(self.folder_path, f) 
                          for f in os.listdir(self.folder_path) 
                          if f.lower().endswith('.obj')]
        self.obj_files.sort()

    def clear_scene(self):
        """Delete all objects in the current scene."""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def load_current_file(self, context):
        """Clear the scene and load the OBJ file at current_index."""
        self.clear_scene()
        if self.current_index < len(self.obj_files):
            file_path = self.obj_files[self.current_index]
            self.report({'INFO'}, f"Loading: {os.path.basename(file_path)}")
            bpy.ops.wm.obj_import(filepath=file_path)
        else:
            self.report({'INFO'}, "No more files to load.")

    def invoke(self, context, event):
        # Try to get a 3D view context override.
        if context.area.type == 'VIEW_3D':
            self.override_context = context.copy()
        else:
            self.override_context = get_3d_view_context()

        if not self.override_context:
            self.report({'ERROR'}, "Could not find a 3D view context.")
            return {'CANCELLED'}

        # Delete the existing labels CSV file if it exists.
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
            self.report({'INFO'}, "Existing labels.csv deleted. Creating a new one.")

        self.load_obj_files()
        if not self.obj_files:
            self.report({'ERROR'}, "No OBJ files found in the folder.")
            return {'CANCELLED'}

        self.current_index = 0
        self.points = []
        self.load_current_file(context)
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Right-click on the mesh to mark a point (6 points per file).")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # Only process right mouse button presses for labeling.
        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            # Use the override context's region and region_data.
            region = self.override_context['region']
            rv3d = self.override_context['region_data']
            # Map global mouse coordinates to local region coordinates.
            local_coord = (event.mouse_x - region.x, event.mouse_y - region.y)

            # Compute the ray from the view using the local coordinates.
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, local_coord)
            direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, local_coord)
            depsgraph = self.override_context['scene'].view_layers[0].depsgraph
            hit, location, normal, face_index, obj, matrix = self.override_context['scene'].ray_cast(depsgraph, origin, direction, distance=1000)
            
            if hit:
                self.points.append(location.copy())
                self.report({'INFO'}, f"Point {len(self.points)} recorded at {location}")
                # (Optional) Visualize the point by adding a small sphere.
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.005, location=location)
            else:
                self.report({'WARNING'}, "No mesh hit. Right-click again.")

            # When 6 points have been marked, save them and load the next file.
            if len(self.points) >= 6:
                current_file = os.path.basename(self.obj_files[self.current_index])
                self.save_points_to_csv(current_file, self.points)
                self.report({'INFO'}, f"Saved 6 points for {current_file}")
                self.current_index += 1
                if self.current_index < len(self.obj_files):
                    self.points = []
                    self.load_current_file(context)
                    self.report({'INFO'}, "Loaded next file. Right-click to label points.")
                else:
                    self.report({'INFO'}, "All files have been labeled.")
                    return {'FINISHED'}
            return {'RUNNING_MODAL'}

        elif event.type in {'ESC'}:
            self.report({'INFO'}, "Labeling cancelled.")
            return {'CANCELLED'}

        # Pass through all other events so normal navigation works.
        return {'PASS_THROUGH'}

    def save_points_to_csv(self, filename, points):
        """Append the filename and the six transformed (x,y,z) coordinates to the CSV output file.
           The transformation converts Blender coordinates to the SMPL-X coordinate system:
           SMPL-X: x = B_x, y = B_z, z = -B_y
        """
        row = [filename]
        for pt in points:
            # Apply the transformation: (x, y, z) -> (x, z, -y)
            transformed_x = pt.x
            transformed_y = pt.z
            transformed_z = -pt.y
            row.extend([transformed_x, transformed_y, transformed_z])
        file_exists = os.path.exists(self.output_path)
        with open(self.output_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                header = ['Filename'] + [f'P{i+1}_{axis}' for i in range(6) for axis in ['x', 'y', 'z']]
                writer.writerow(header)
            writer.writerow(row)

def register():
    bpy.utils.register_class(OBJECT_OT_LabelPoints)

def unregister():
    bpy.utils.unregister_class(OBJECT_OT_LabelPoints)

if __name__ == "__main__":
    register()
    # Start the operator (it now uses right-click to label).
    bpy.ops.object.label_points('INVOKE_DEFAULT')
