import os
import numpy as np
import pandas as pd
import trimesh

def presample_meshes(labels_csv_path, mesh_folder, output_folder, num_sampled_points=1024, input_type="obj", delimiter=","):
    """
    Reads the labels CSV, loads each mesh file (either .obj or .xyz),
    samples a fixed number of points, and saves the sampled point cloud as a .npy file in the output folder.
    
    Parameters:
      labels_csv_path: Path to the CSV file containing at least a "filename" column.
      mesh_folder: Folder where the mesh files are stored.
      output_folder: Folder where the output .npy files will be saved.
      num_sampled_points: Number of points to sample from each mesh.
      input_type: "obj" to process OBJ files using trimesh, or "xyz" to process .xyz point cloud files.
      delimiter: Delimiter to use when loading .xyz files.
    """
    df = pd.read_csv(labels_csv_path)
    # Rename "Filename" to "filename" for consistency if necessary.
    if "Filename" in df.columns:
        df.rename(columns={"Filename": "filename"}, inplace=True)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for idx, row in df.iterrows():
        filename = row["filename"]
        mesh_path = os.path.join(mesh_folder, filename)
        try:
            print(f"Processing {filename} ...")
            if input_type.lower() == "obj":
                # Load the mesh using trimesh and sample the points.
                mesh = trimesh.load(mesh_path, force='mesh')
                points = mesh.sample(num_sampled_points)  # shape: (num_sampled_points, 3)
            elif input_type.lower() == "xyz":
                # Load the point cloud using numpy.
                points_full = np.loadtxt(mesh_path, delimiter=delimiter)
                # If there are more points than required, sample uniformly.
                if points_full.shape[0] > num_sampled_points:
                    indices = np.random.choice(points_full.shape[0], num_sampled_points, replace=False)
                    points = points_full[indices]
                else:
                    points = points_full
            else:
                raise ValueError("Invalid input_type. Use 'obj' or 'xyz'.")
            
            # Save the sampled points as a .npy file.
            # Replace the extension accordingly.
            if input_type.lower() == "obj":
                output_file = os.path.join(output_folder, filename.replace(".obj", ".npy"))
            else:
                output_file = os.path.join(output_folder, filename.replace(".xyz", ".npy"))
            np.save(output_file, points)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    # --- Configuration (update these paths) ---
    MESH_FOLDER = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\Scan\transformed_meshes"
    LABELS_CSV_PATH = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_labels\labels.csv"
    OUTPUT_FOLDER = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\presampled_meshes\32_points"
    NUM_SAMPLED_POINTS = 32

    # Set the input type: either "obj" or "xyz"
    INPUT_TYPE = "xyz"  # Change to "xyz" if processing .xyz point cloud files.

    presample_meshes(LABELS_CSV_PATH, MESH_FOLDER, OUTPUT_FOLDER, NUM_SAMPLED_POINTS, input_type=INPUT_TYPE)
