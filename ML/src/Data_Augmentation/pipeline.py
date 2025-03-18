import os
import shutil
import numpy as np
import csv

# Import transformation and noise functions from your modules.
from addTransformation import transform_meshes_and_labels
from addNoise import process_obj_file

# --- Configuration ---
# Folders with the original dataset.
ORIGINAL_MESH_FOLDER = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\ML\smplx_big_test\smplx_meshes"
ORIGINAL_LABELS_CSV = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\ML\smplx_big_test\smplx_labels\labels_reordered.csv"

# Base folders for final outputs.
FINAL_DATASET_FOLDER = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\pipeline\final_dataset"
FINAL_DOWNSAMPLED_FOLDER = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\pipeline\final_downsampled"

# Temporary folders (will be re-created each iteration).
TEMP_TRANSFORM_FOLDER = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\pipeline\temp_transform"
TEMP_OBJ_FOLDER = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\pipeline\temp_obj"

# Set number of downsampled points.
NUM_SAMPLED_POINTS = 512

# Set multiplication factor.
MULTIPLICATION_FACTOR = 3

# Clear and recreate final output folders.
for folder in [FINAL_DATASET_FOLDER, FINAL_DOWNSAMPLED_FOLDER]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Ensure temporary folders are cleared.
for folder in [TEMP_TRANSFORM_FOLDER, TEMP_OBJ_FOLDER]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Global counter for naming final meshes.
global_counter = 1

# This list will store label rows from all iterations.
master_labels = []

# --- Helper Functions ---
def npy_to_obj(npy_path, original_obj_path, output_obj_path):
    """
    Converts a transformed mesh (saved as .npy with vertices only)
    to a valid OBJ file by combining the transformed vertices with
    the non-vertex lines (faces, etc.) from the original OBJ file.
    """
    vertices = np.load(npy_path)
    with open(original_obj_path, "r") as f:
        lines = f.readlines()
    non_vertex_lines = [line for line in lines if not line.startswith("v ")]
    with open(output_obj_path, "w") as f:
        for v in vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for line in non_vertex_lines:
            f.write(line)

def load_all_vertices(obj_path):
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
                        continue
    return np.array(vertices)

def downsample_obj_file(input_obj_path, output_npy_path, num_points):
    vertices = load_all_vertices(input_obj_path)
    if vertices.shape[0] > num_points:
        indices = np.random.choice(vertices.shape[0], num_points, replace=False)
        downsampled = vertices[indices]
    else:
        downsampled = vertices
    np.save(output_npy_path, downsampled)

def append_labels(master_list, labels_csv_path, new_names_ordered):
    """
    Reads the labels CSV from one run and replaces the filename field with the new names.
    It assumes the CSV rows are in the same order as sorted filenames.
    """
    with open(labels_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Replace the filename with the corresponding new name.
            row["filename"] = new_names_ordered[i]
            master_list.append(row)
    # Return header (fieldnames) from the CSV.
    with open(labels_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames

# --- Main Pipeline Loop ---
for run in range(1, MULTIPLICATION_FACTOR + 1):
    print(f"\n=== Pipeline iteration {run} ===")
    # Clear temporary folders for this run.
    for folder in [TEMP_TRANSFORM_FOLDER, TEMP_OBJ_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
    
    # --- Step 1: Apply Transformation ---
    print("Step 1: Transformation")
    transform_meshes_and_labels(
        labels_csv_path=ORIGINAL_LABELS_CSV,
        mesh_folder=ORIGINAL_MESH_FOLDER,
        output_folder=TEMP_TRANSFORM_FOLDER,
        input_type="obj"  # assuming original meshes are OBJ files.
    )
    transformed_labels_csv = os.path.join(TEMP_TRANSFORM_FOLDER, "labels_transformed.csv")
    if not os.path.exists(transformed_labels_csv):
        raise RuntimeError("Transformed labels CSV was not created.")
    print("Transformation complete.")
    
    # --- Step 2 & 3: Convert .npy to OBJ and Add Noise ---
    print("Step 2 & 3: Converting .npy to OBJ and Adding Noise")
    # Get sorted list of transformed .npy files (assumes order matches the labels CSV order)
    npy_files = sorted([f for f in os.listdir(TEMP_TRANSFORM_FOLDER) if f.lower().endswith(".npy")])
    if not npy_files:
        raise RuntimeError("No transformed .npy files found in the temp transform folder.")
    
    # This list will store the new file names for this run (to later update the labels)
    new_names_this_run = []
    
    for filename in npy_files:
        npy_path = os.path.join(TEMP_TRANSFORM_FOLDER, filename)
        # Determine the original OBJ file (assumes same base name as the npy file)
        original_obj_filename = filename.replace(".npy", ".obj")
        original_obj_path = os.path.join(ORIGINAL_MESH_FOLDER, original_obj_filename)
        if not os.path.exists(original_obj_path):
            print(f"Warning: Original OBJ file {original_obj_path} not found; skipping {filename}.")
            continue
        
        # Generate new sequential file name for the final noisy mesh.
        new_name = f"smplx_mesh_{global_counter:03d}.obj"
        new_names_this_run.append(new_name)
        temp_obj_path = os.path.join(TEMP_OBJ_FOLDER, new_name)
        
        # Convert .npy to OBJ (using the original OBJ for non-vertex data) into temporary OBJ folder.
        npy_to_obj(npy_path, original_obj_path, temp_obj_path)
        
        # Apply noise: read the temporary OBJ and output the noisy OBJ into the FINAL_DATASET_FOLDER.
        final_obj_path = os.path.join(FINAL_DATASET_FOLDER, new_name)
        process_obj_file(temp_obj_path, final_obj_path)
        print(f"Processed and saved {new_name}")
        
        global_counter += 1
    print("Conversion and noise addition complete.")
    
    # --- Step 4: Process Labels CSV ---
    print("Step 4: Processing Labels CSV")
    # Append labels for this run (assumes order in CSV matches sorted npy_files)
    header = append_labels(master_labels, transformed_labels_csv, new_names_this_run)
    print("Labels processed for this iteration.")

# --- Step 5: Downsample Noisy Meshes ---
print("\n=== Step 5: Downsampling Noisy Meshes ===")
# Process each noisy OBJ file in the final dataset folder.
noisy_obj_files = sorted([f for f in os.listdir(FINAL_DATASET_FOLDER) if f.lower().endswith(".obj")])
if not noisy_obj_files:
    raise RuntimeError("No noisy OBJ files found in the final dataset folder.")

for filename in noisy_obj_files:
    input_obj_path = os.path.join(FINAL_DATASET_FOLDER, filename)
    # Save downsampled point cloud with same base name but .npy extension.
    output_npy_path = os.path.join(FINAL_DOWNSAMPLED_FOLDER, filename.replace(".obj", ".npy"))
    downsample_obj_file(input_obj_path, output_npy_path, NUM_SAMPLED_POINTS)
    print(f"Downsampled {filename} to {NUM_SAMPLED_POINTS} points: {output_npy_path}")
print("Downsampling complete.")

# --- Write Combined Labels CSV ---
final_labels_csv = os.path.join(FINAL_DATASET_FOLDER, "labels_final.csv")
print("\n=== Writing combined labels CSV ===")
with open(final_labels_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    for row in master_labels:
        writer.writerow(row)
print(f"Combined labels CSV written to: {final_labels_csv}")

print("\n=== Pipeline Complete ===")
print(f"Final noisy meshes (OBJ) are in: {FINAL_DATASET_FOLDER}")
print(f"Final downsampled point clouds (.npy) are in: {FINAL_DOWNSAMPLED_FOLDER}")
print(f"Combined labels CSV is at: {final_labels_csv}")
