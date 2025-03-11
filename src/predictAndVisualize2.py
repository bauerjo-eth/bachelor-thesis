import os
import numpy as np
import pandas as pd
import torch
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the classes from your training script.
# Adjust the module name if your training script is named differently.
from trainEFAST import MeshToEFastNet, MeshToEFastDataset

# ============================
# === Configuration Values ===
# ============================

# Paths (update these to match your environment)
MESH_FOLDER = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_meshes"
LABELS_CSV_PATH = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_labels\labels_reordered.csv"
# If you pre-sampled the meshes, set PRESAMPLED_DIR to the folder with the .npy files; otherwise None.
PRESAMPLED_DIR = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\presampled_meshes\32_points"

# Number of points used in training (must match training configuration)
NUM_SAMPLED_POINTS = 32

# Checkpoint file (update the path to the checkpoint you want to use)
CHECKPOINT_PATH = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\model_checkpoints\checkpoint_20250311_131923.pth"

# Choose which mesh to visualize:
# If True, the script will load and display the original full-resolution OBJ mesh vertices.
# If False, it will visualize the downsampled mesh (the points used as input to the network).
VISUALIZE_ORIGINAL = True

# Sample index to predict (you can change this)
SAMPLE_INDEX = 225

# ============================
# === Prediction Function  ===
# ============================

def predict_and_visualize():
    # --- Device configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Load labels CSV ---
    labels_df = pd.read_csv(LABELS_CSV_PATH)
    if "Filename" in labels_df.columns:
        labels_df.rename(columns={"Filename": "filename"}, inplace=True)
    
    # Determine output dimension (all columns except "filename")
    label_cols = [col for col in labels_df.columns if col != "filename"]
    output_dim = len(label_cols)
    
    # --- Create Dataset ---
    dataset = MeshToEFastDataset(
        labels_df, 
        MESH_FOLDER, 
        num_sampled_points=NUM_SAMPLED_POINTS, 
        presample_dir=PRESAMPLED_DIR
    )
    
    # Get a sample from the dataset.
    # Each sample is (X, Y) where X is a flattened vector of sampled points.
    sample_X, sample_Y = dataset[SAMPLE_INDEX]
    sample_X = sample_X.to(device)
    
    # --- Initialize the model ---
    # The input dimension is number of points * 3.
    input_dim = NUM_SAMPLED_POINTS * 3
    model = MeshToEFastNet(input_dim, output_dim).to(device)
    
    # --- Load the checkpoint ---
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    else:
        print("Checkpoint file not found!")
        return

    model.eval()
    with torch.no_grad():
        prediction = model(sample_X.unsqueeze(0))  # Add batch dimension
    # Reshape prediction into (-1, 3) assuming each keypoint is 3 coordinates.
    predicted_keypoints = prediction.cpu().numpy().reshape(-1, 3)
    print("Predicted keypoints:\n", predicted_keypoints)
    
    # --- Visualization ---
    if VISUALIZE_ORIGINAL:
        # Option 1: Visualize original full mesh vertices using matplotlib.
        # Get the mesh filename for the sample.
        mesh_filename = labels_df.iloc[SAMPLE_INDEX]["filename"]
        mesh_path = os.path.join(MESH_FOLDER, mesh_filename)
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
        except Exception as e:
            print("Error loading mesh:", e)
            return

        vertices = mesh.vertices  # Use vertices only, not faces.
        
        # Create a matplotlib 3D plot.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the mesh vertices as blue dots.
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   c='b', label='Mesh Vertices', s=0.25, zorder=1)
        
        # Plot the predicted keypoints as red dots.
        ax.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], predicted_keypoints[:, 2],
                   c='r', label='Predicted Keypoints', s=40, zorder=10)
        
        ax.set_title("Original Mesh Vertices with Predicted Keypoints")
        ax.legend()
        
        # Set axes limits based on the mesh bounds.
        bounds = vertices.max(axis=0) - vertices.min(axis=0)
        max_range = bounds.max() / 2.0
        mid = vertices.mean(axis=0)
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
        plt.show()

    else:
        # Option 2: Visualize the downsampled mesh points (the input data) with matplotlib.
        downsampled_points = sample_X.cpu().numpy().reshape(-1, 3)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot the downsampled mesh points in blue.
        ax.scatter(downsampled_points[:, 0], downsampled_points[:, 1], downsampled_points[:, 2],
                   c='b', label='Downsampled Points', s=2.5)
        # Plot the predicted keypoints in red.
        ax.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], predicted_keypoints[:, 2],
                   c='r', label='Predicted Keypoints', s=40)
        ax.set_title("Downsampled Mesh and Predicted Keypoints")
        ax.legend()
        # Set axes limits (adjust as necessary for your data).
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.show()

if __name__ == '__main__':
    predict_and_visualize()
