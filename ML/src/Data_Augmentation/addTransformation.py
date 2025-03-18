import os
import numpy as np
import pandas as pd

def load_all_vertices(obj_path):
    """
    Manually parse the OBJ file to load all vertices.
    
    Parameters:
      obj_path: Path to the .obj file.
      
    Returns:
      A NumPy array of shape (N, 3) containing all vertex coordinates.
    """
    vertices = []
    with open(obj_path, "r") as file:
        for line in file:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        vertices.append([x, y, z])
                    except ValueError:
                        continue
    return np.array(vertices)

def random_rotation_matrix(max_rotation_deg):
    """
    Generate a random rotation matrix with each axis rotated by an angle
    uniformly sampled from [-max_rotation_deg, max_rotation_deg].
    """
    max_rot_rad = np.deg2rad(max_rotation_deg)
    theta_x = np.random.uniform(-max_rot_rad, max_rot_rad)
    theta_y = np.random.uniform(-max_rot_rad, max_rot_rad)
    theta_z = np.random.uniform(-max_rot_rad, max_rot_rad)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x),  np.cos(theta_x)]])
    Ry = np.array([[ np.cos(theta_y), 0, np.sin(theta_y)],
                   [0,                1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z),  np.cos(theta_z), 0],
                   [0,                0,               1]])
    R = Rz @ Ry @ Rx
    return R

def random_translation_vector(max_translation):
    """
    Generate a random translation vector with each component sampled uniformly
    from [-max_translation, max_translation].
    """
    return np.random.uniform(-max_translation, max_translation, size=(3,))

def transform_labels(row, R, t, num_label_points=6):
    """
    Apply the same transformation (R, t) to the label points in a CSV row.
    
    Assumes the CSV row format is:
      [filename, x1, y1, z1, x2, y2, z2, ..., xN, yN, zN]
      
    Parameters:
      row: A pandas Series representing one row.
      R: 3x3 rotation matrix.
      t: 3-element translation vector.
      num_label_points: Number of label points (each with x, y, z).
      
    Returns:
      A list: [filename, new_x1, new_y1, new_z1, ..., new_xN, new_yN, new_zN]
    """
    transformed = []
    # row[0] is assumed to be the filename.
    for i in range(num_label_points):
        base = 1 + i * 3
        x = float(row.iloc[base])
        y = float(row.iloc[base+1])
        z = float(row.iloc[base+2])
        new_point = R @ np.array([x, y, z]) + t
        transformed.extend(new_point.tolist())
    return [row.iloc[0]] + transformed

def transform_meshes_and_labels(labels_csv_path, mesh_folder, output_folder,
                                input_type="obj", delimiter=",", max_translation=5,
                                max_rotation_deg=90, num_label_points=6):
    """
    Reads the labels CSV, loads each mesh file (OBJ or XYZ), applies a random transformation
    (rotation and translation) to all mesh vertices and label points, and saves the transformed
    mesh as a .npy file. Also writes the transformed labels to a new CSV file.
    
    The new CSV file is saved as "labels_transformed.csv" in the output folder (overwriting any existing file).
    Note: No downsampling is performed; the full set of vertices is transformed.
    """
    df = pd.read_csv(labels_csv_path)
    # Rename "Filename" to "filename" if needed.
    if "Filename" in df.columns:
        df.rename(columns={"Filename": "filename"}, inplace=True)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    transformed_labels = []
    
    for idx, row in df.iterrows():
        filename = row["filename"]
        mesh_path = os.path.join(mesh_folder, filename)
        try:
            print(f"Processing {filename} ...")
            # Generate a random transformation.
            R = random_rotation_matrix(max_rotation_deg)
            t = random_translation_vector(max_translation)
            
            if input_type.lower() == "obj":
                # Load the mesh vertices manually.
                vertices = load_all_vertices(mesh_path)
                # Apply the transformation to all vertices.
                transformed_vertices = (R @ vertices.T).T + t  # shape: (N, 3)
                points = transformed_vertices  # Use all points without downsampling.
            elif input_type.lower() == "xyz":
                points_full = np.loadtxt(mesh_path, delimiter=delimiter)
                transformed_points_full = (R @ points_full.T).T + t
                points = transformed_points_full  # Use all points.
            else:
                raise ValueError("Invalid input_type. Use 'obj' or 'xyz'.")
            
            # Save the full, transformed point cloud as a .npy file.
            if input_type.lower() == "obj":
                output_file = os.path.join(output_folder, filename.replace(".obj", ".npy"))
            else:
                output_file = os.path.join(output_folder, filename.replace(".xyz", ".npy"))
            np.save(output_file, points)
            
            # Transform the labels using the same transformation.
            transformed_row = transform_labels(row, R, t, num_label_points=num_label_points)
            transformed_labels.append(transformed_row)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Create a new DataFrame for transformed labels.
    columns = ["filename"] + [f"{coord}{i+1}" for i in range(num_label_points) for coord in ["x", "y", "z"]]
    transformed_df = pd.DataFrame(transformed_labels, columns=columns)
    
    # Remove any existing labels_transformed.csv and write the new one.
    transformed_csv_path = os.path.join(output_folder, "labels_transformed.csv")
    if os.path.exists(transformed_csv_path):
        os.remove(transformed_csv_path)
    transformed_df.to_csv(transformed_csv_path, index=False)
    print(f"Transformed labels saved to {transformed_csv_path}")

if __name__ == '__main__':
    # --- Configuration (update these paths) ---
    MESH_FOLDER = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_small_test\smplx_meshes_test"
    LABELS_CSV_PATH = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_small_test\smplx_labels_test\labels.csv"
    OUTPUT_FOLDER = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_small_test\smplx_meshes_with_transformations"

    # Set the input type: either "obj" or "xyz"
    INPUT_TYPE = "obj"  # Change to "xyz" if processing .xyz point cloud files.

    transform_meshes_and_labels(LABELS_CSV_PATH, MESH_FOLDER, OUTPUT_FOLDER,
                                input_type=INPUT_TYPE)
