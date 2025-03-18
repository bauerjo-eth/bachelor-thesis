# dataloader.py Farthest Point Sampling and random transformations.
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import trimesh
from scipy.spatial import KDTree
import pandas as pd

def farthest_point_sampling(points, num_samples):
    """
    Performs farthest point sampling on a point cloud.
    
    Parameters:
        points (np.ndarray): Input point cloud of shape (N, 3).
        num_samples (int): Number of points to sample.
    
    Returns:
        np.ndarray: Indices of the sampled points (shape: (num_samples,)).
    """
    N = points.shape[0]
    if num_samples >= N:
        indices = np.arange(N)
        extra_indices = np.random.choice(N, num_samples - N, replace=True)
        return np.concatenate([indices, extra_indices])
    
    selected_indices = np.empty(num_samples, dtype=np.int64)
    # Start with a random point.
    selected_indices[0] = np.random.randint(0, N)
    distances = np.full(N, np.inf)
    
    for i in range(1, num_samples):
        last_selected = points[selected_indices[i - 1]]
        # Compute distances from the last selected point to all points.
        dists = np.linalg.norm(points - last_selected, axis=1)
        # Update the distances: keep the minimum distance to any selected point.
        distances = np.minimum(distances, dists)
        # Select the point with the maximum distance.
        selected_indices[i] = np.argmax(distances)
    return selected_indices

def get_random_transformation(max_rotation_deg=90, max_translation=1.5):
    """
    Generates a random transformation with rotations and translations.
    
    Returns:
        R (np.ndarray): Rotation matrix of shape (3, 3).
        t (np.ndarray): Translation vector of shape (3,).
    """
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
    
    R = Rz @ Ry @ Rx
    t = np.random.uniform(-max_translation, max_translation, size=(3,))
    return R, t

class EFASTDataset(Dataset):
    """
    Dataset for eFAST landmark segmentation.

    Each sample:
      - Loads a .obj mesh and extracts its vertices as a point cloud.
      - Downsamples (or duplicates points if needed) to a fixed number of points using farthest point sampling.
      - Applies a random transformation (rotation up to ±90° around each axis and translation up to ±1.5m)
        to both the point cloud and its corresponding landmarks.
      - Loads the corresponding eFAST landmark annotations from a CSV file.
      - For each of the 6 landmarks, finds the N nearest vertices (using a KDTree)
        and assigns them labels 1 to 6.
      - All other vertices are assigned label 0 (background).

    CSV Format:
      Must contain columns: 'filename' and for each landmark the columns:
      '<landmark>_x', '<landmark>_y', '<landmark>_z'.
      The expected landmarks are:
         right_chest, left_chest, right_side, left_side, middle_above_lungs, below_belly.
    """
    def __init__(self, obj_dir, csv_file, num_points=2048, nearest_count=20, transform=None):
        """
        obj_dir: Directory containing .obj mesh files.
        csv_file: Path to CSV file with eFAST landmark coordinates.
        num_points: Fixed number of points to output per sample.
        nearest_count: Number of nearest vertices to assign per landmark.
        transform: Optional transformation applied to the point cloud.
        """
        self.obj_dir = obj_dir
        self.csv_data = pd.read_csv(csv_file)
        self.num_points = num_points
        self.nearest_count = nearest_count
        self.transform = transform
        
        self.samples = []
        for idx, row in self.csv_data.iterrows():
            filename = row['filename']
            obj_path = os.path.join(self.obj_dir, filename)
            if os.path.exists(obj_path):
                self.samples.append((filename, obj_path))
        if len(self.samples) == 0:
            raise ValueError("No matching .obj files found in the provided directory.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, obj_path = self.samples[idx]
        # Load mesh and extract vertices.
        mesh = trimesh.load(obj_path, force='mesh')
        points = mesh.vertices  # shape: (num_points_in_mesh, 3)
        orig_num_points = points.shape[0]
        
        # Downsample (or duplicate) points using farthest point sampling.
        points = points.astype(np.float32)
        indices = farthest_point_sampling(points, self.num_points)
        points = points[indices]
        num_points = points.shape[0]  # should equal self.num_points

        # Apply random transformation to the point cloud.
        R, t = get_random_transformation(max_rotation_deg=90, max_translation=1.5)
        points = (R @ points.T).T + t

        # Retrieve corresponding landmarks from CSV.
        csv_row = self.csv_data[self.csv_data['filename'] == filename]
        if csv_row.empty:
            raise ValueError(f"Landmark information for file {filename} not found in CSV.")
        csv_row = csv_row.iloc[0]
        
        # Define the expected landmark names.
        landmark_names = [
            "right_chest",
            "left_chest",
            "right_side",
            "left_side",
            "middle_above_lungs",
            "below_belly"
        ]
        
        # Extract landmarks.
        landmarks = []
        for name in landmark_names:
            x = csv_row[f'{name}_x']
            y = csv_row[f'{name}_y']
            z = csv_row[f'{name}_z']
            landmarks.append(np.array([x, y, z]))
        landmarks = np.stack(landmarks, axis=0)  # shape: (6, 3)
        # Apply the same transformation to the landmarks.
        landmarks = (R @ landmarks.T).T + t
        
        # Initialize segmentation labels for each point as background (0).
        seg_labels = np.zeros(num_points, dtype=np.int64)
        
        # Build a KDTree on the transformed point cloud.
        tree = KDTree(points)
        # For each landmark, assign the nearest 'nearest_count' points the label (i+1).
        for i in range(6):
            landmark = landmarks[i]
            distances, nearest_idxs = tree.query(landmark, k=self.nearest_count)
            nearest_idxs = np.atleast_1d(nearest_idxs)
            seg_labels[nearest_idxs] = i + 1  # Labels 1 to 6
        
        # Optionally apply an additional transformation.
        if self.transform:
            points = self.transform(torch.tensor(points, dtype=torch.float32)).numpy()
        
        # Convert to torch tensors.
        points = torch.tensor(points, dtype=torch.float32)
        seg_labels = torch.tensor(seg_labels, dtype=torch.long)
        return points, seg_labels

"""
Output of the dataloader:
Each sample returned by the dataloader is a tuple (points, seg_labels).
- 'points' is a torch tensor of shape (num_points, 3), representing one mesh's point cloud
  (after downsampling, random transformation, and optional additional transformations).
- 'seg_labels' is a torch tensor of shape (num_points,) containing the segmentation labels
  (0 for background and 1–6 for the corresponding eFAST landmarks).
Thus, each sample corresponds to one processed mesh, which is what the PointNet network expects.
"""

if __name__ == '__main__':
    # Example usage:
    obj_dir = r"path/to/obj_files"  # Update with your .obj directory
    csv_file = r"path/to/labels.csv"  # Update with your CSV file
    dataset = EFASTDataset(obj_dir, csv_file, num_points=2048, nearest_count=20)
    print("Dataset length:", len(dataset))
    sample_points, sample_labels = dataset[0]
    print("Points shape:", sample_points.shape)
    print("Unique labels in sample:", torch.unique(sample_labels))

