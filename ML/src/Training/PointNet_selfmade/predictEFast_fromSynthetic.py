import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from model import LandmarkSegmentationNet
from dataloader import farthest_point_sampling  # Import the FPS function

# =======================================
# File Paths and Parameters (Update as needed)
# =======================================
INPUT_OBJ = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\ML\smplx_small_test\smplx_meshes_test\smplx_mesh_003.obj"  # Path to an original .obj file.
CHECKPOINT_FILE = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\ML\src\Training\PointNet\Checkpoints\smplx_big_test\best_model_20250318_133257.pth"  # Path to the checkpoint file.
NUM_POINTS = 2048  # Desired number of points per sample.

def load_obj_points(filename):
    """
    Loads an .obj file using trimesh and returns its vertices as a numpy array.
    
    Args:
        filename (str): Path to the .obj file.
    
    Returns:
        points: np.ndarray of shape (N, 3)
    """
    try:
        mesh = trimesh.load(filename, force='mesh')
        points = mesh.vertices.astype(np.float32)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        raise e
    return points

def random_transform(points, max_rotation_deg=90, max_translation=1.0):
    """
    Applies a random transformation to the point cloud.
    Each axis is rotated by an angle uniformly sampled from [-max_rotation_deg, max_rotation_deg]
    and the entire cloud is translated by a vector with components in [-max_translation, max_translation].
    
    Args:
        points (np.ndarray): Input point cloud of shape (N, 3).
        max_rotation_deg (float): Maximum rotation angle in degrees.
        max_translation (float): Maximum translation in meters.
    
    Returns:
        transformed_points (np.ndarray): Transformed point cloud of shape (N, 3).
    """
    # Generate random rotation angles in radians.
    angles = np.deg2rad(np.random.uniform(-max_rotation_deg, max_rotation_deg, size=3))
    theta_x, theta_y, theta_z = angles

    # Rotation matrix around x-axis.
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    # Rotation matrix around y-axis.
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    # Rotation matrix around z-axis.
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix.
    R = Rz @ Ry @ Rx

    # Generate random translation vector.
    t = np.random.uniform(-max_translation, max_translation, size=(3,))
    
    # Apply transformation.
    transformed_points = (R @ points.T).T + t
    return points

def visualize_prediction(points, pred_labels):
    """
    Visualizes the point cloud colored by the predicted labels.
    
    Args:
        points: np.ndarray of shape (N, 3)
        pred_labels: np.ndarray of shape (N,) with integer labels.
    """
    # Define a color map: background (label 0) in gray; landmarks (1-6) in distinct colors.
    color_map = {
        0: 'gray',
        1: 'red',
        2: 'green',
        3: 'blue',
        4: 'orange',
        5: 'purple',
        6: 'cyan'
    }
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(pred_labels)
    for lab in unique_labels:
        idx = np.where(pred_labels == lab)[0]
        marker_size = 2 if lab == 0 else 20
        ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2],
                   c=color_map.get(lab, 'black'),
                   label=f"Label {lab}",
                   s=marker_size)
    
    ax.set_title("Predicted eFAST Landmarks")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_zlabel("Z")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.legend()
    plt.show()

def main():
    print(f"Loading .obj file from: {INPUT_OBJ}")
    points = load_obj_points(INPUT_OBJ)
    print(f"Original point cloud shape: {points.shape}")
    
    # Apply random transformation.
    transformed_points = random_transform(points, max_rotation_deg=90, max_translation=1.0)
    print("Applied random transformation (rotation up to ±90° and translation up to ±1m).")
    
    # Use farthest point sampling to get exactly NUM_POINTS.
    sampled_indices = farthest_point_sampling(transformed_points, NUM_POINTS)
    sampled_points = transformed_points[sampled_indices]
    print(f"Sampled point cloud shape: {sampled_points.shape}")
    
    # Convert to torch tensor and add a batch dimension: (1, NUM_POINTS, 3).
    points_tensor = torch.tensor(sampled_points, dtype=torch.float32).unsqueeze(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    points_tensor = points_tensor.to(device)
    
    # Load the model and checkpoint.
    model = LandmarkSegmentationNet(num_classes=7).to(device)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=device))
        print(f"Loaded checkpoint from: {CHECKPOINT_FILE}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        seg_logits = model(points_tensor)  # Expected shape: (1, NUM_POINTS, 7)
        pred_labels = torch.argmax(seg_logits, dim=2)  # Shape: (1, NUM_POINTS)
    prediction_time = time.time() - start_time
    print(f"Prediction completed in {prediction_time:.4f} seconds")
    
    pred_labels = pred_labels.squeeze(0).cpu().numpy()  # (NUM_POINTS,)
    
    # Visualize the predictions.
    visualize_prediction(sampled_points, pred_labels)

if __name__ == '__main__':
    main()
