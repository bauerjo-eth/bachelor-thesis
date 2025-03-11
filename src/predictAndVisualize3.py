import os
import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import your trained model class.
# Adjust the module name if your training script is named differently.
from trainEFAST import MeshToEFastNet

# ============================
# === Configuration Values ===
# ============================

# Choose input format for the downsampled file: "npy", "obj", or "xyz"
# (Prediction always uses the downsampled version.)
INPUT_FORMAT = "npy"

# File path for the downsampled point cloud (used for prediction)
DOWNSAMPLED_FILE_PATH = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\presampled_meshes\32_points_from_point_cloud\transformed_point_cloud_sample_2.npy"

# File path for the full (original) point cloud used for visualization
FULL_CLOUD_FILE_PATH = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\Scan\transformed_meshes\transformed_point_cloud.xyz"

# Number of points used in training (must match training configuration)
NUM_SAMPLED_POINTS = 32

# Checkpoint file for the trained NN model.
CHECKPOINT_PATH = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\model_checkpoints\checkpoint_20250311_131923.pth"

# Output dimension of your NN model (e.g. if it predicts 18 values, set OUTPUT_DIM = 18)
OUTPUT_DIM = 18

# Visualization flag: if True, use the full point cloud for visualization; otherwise, use the downsampled sample.
VISUALIZE_FULL_CLOUD = True

# ============================
# === Loading Functions      ===
# ============================

def load_downsampled_points():
    """
    Loads the downsampled point cloud used for prediction.
    Returns an (N x 3) numpy array.
    """
    if INPUT_FORMAT.lower() == "npy":
        points = np.load(DOWNSAMPLED_FILE_PATH)
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        return points
    elif INPUT_FORMAT.lower() == "obj":
        mesh = trimesh.load(DOWNSAMPLED_FILE_PATH, force='mesh')
        points = mesh.sample(NUM_SAMPLED_POINTS)
        return points
    elif INPUT_FORMAT.lower() == "xyz":
        data = np.loadtxt(DOWNSAMPLED_FILE_PATH, delimiter=',')
        total_points = data.shape[0]
        if total_points >= NUM_SAMPLED_POINTS:
            indices = np.random.choice(total_points, NUM_SAMPLED_POINTS, replace=False)
            points = data[indices, :3]
        else:
            points = data[:, :3]
        return points
    else:
        raise ValueError("Invalid INPUT_FORMAT. Use 'npy', 'obj', or 'xyz'.")

def load_full_cloud():
    """
    Loads the full (original) point cloud for visualization.
    Returns an (N x 3) numpy array.
    """
    # We assume the full cloud is stored as an .xyz file (comma-separated).
    # Adjust this function if your full cloud is stored differently.
    data = np.loadtxt(FULL_CLOUD_FILE_PATH, delimiter=',')
    return data[:, :3]

# ============================
# === Prediction & Visualization ===
# ============================

def predict_and_visualize():
    # Device configuration.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load the downsampled point cloud (for prediction).
    points = load_downsampled_points()  # shape: (N, 3)
    
    # Ensure exactly NUM_SAMPLED_POINTS are used for prediction.
    if points.shape[0] > NUM_SAMPLED_POINTS:
        indices = np.random.choice(points.shape[0], NUM_SAMPLED_POINTS, replace=False)
        points = points[indices]
    elif points.shape[0] < NUM_SAMPLED_POINTS:
        extra = np.random.choice(points.shape[0], NUM_SAMPLED_POINTS - points.shape[0], replace=True)
        points = np.concatenate([points, points[extra]], axis=0)
    
    # Flatten the points into a 1D vector.
    input_vector = points.flatten().astype(np.float32)
    input_tensor = torch.from_numpy(input_vector).to(device)
    
    # Initialize the model.
    input_dim = NUM_SAMPLED_POINTS * 3
    model = MeshToEFastNet(input_dim, OUTPUT_DIM).to(device)
    
    # Load the trained model checkpoint.
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    else:
        print("Checkpoint file not found!")
        return
    
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor.unsqueeze(0))  # shape: (1, OUTPUT_DIM)
    
    # Reshape prediction to (-1, 3) assuming each keypoint has 3 coordinates.
    predicted_keypoints = prediction.cpu().numpy().reshape(-1, 3)
    print("Predicted keypoints:\n", predicted_keypoints)
    
    # Visualization.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if VISUALIZE_FULL_CLOUD:
        # Load and display the full point cloud.
        full_points = load_full_cloud()
        ax.scatter(full_points[:, 0], full_points[:, 1], full_points[:, 2],
                   c='c', label='Full Point Cloud', s=0.2)
    else:
        # Display the downsampled points used for prediction.
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c='b', label='Downsampled Points', s=5)
    
    # Overlay predicted keypoints in red.
    ax.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], predicted_keypoints[:, 2],
               c='r', label='Predicted eFAST Points', s=40)
    
    ax.set_title("Prediction and Visualization")
    # Adjust axis limits as needed.
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.legend()
    plt.show()

if __name__ == '__main__':
    predict_and_visualize()
