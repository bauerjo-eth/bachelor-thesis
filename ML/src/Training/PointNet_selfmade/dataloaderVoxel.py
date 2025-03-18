# dataloader.py Voxel
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import trimesh
from scipy.spatial import KDTree
import pandas as pd

def voxel_downsampling(points, voxel_size):
    """
    Downsamples a point cloud using a voxel grid approach.
    Each voxel's centroid is computed and returned.
    
    Parameters:
        points (np.ndarray): Input point cloud of shape (N, 3).
        voxel_size (float): The size of each voxel.
        
    Returns:
        np.ndarray: The downsampled point cloud (M, 3), where M <= N.
    """
    if points.shape[0] == 0:
        return points
    # Shift points so that the minimum coordinate is at the origin.
    min_coords = np.min(points, axis=0)
    shifted_points = points - min_coords
    # Compute voxel indices for each point.
    voxel_indices = np.floor(shifted_points / voxel_size).astype(np.int32)
    # Find unique voxel indices and get the inverse mapping.
    unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    
    centroids = []
    # Compute the centroid for each voxel.
    for i in range(len(unique_voxels)):
        pts_in_voxel = points[inverse == i]
        centroid = np.mean(pts_in_voxel, axis=0)
        centroids.append(centroid)
    centroids = np.stack(centroids, axis=0)
    return centroids

class EFASTDataset(Dataset):
    """
    Dataset for eFAST landmark segmentation.

    Each sample:
      - Loads a .obj mesh and extracts its vertices as a point cloud.
      - Downsamples (or duplicates points if needed) to a fixed number of points using voxel downsampling.
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
    def __init__(self, obj_dir, csv_file, num_points=2048, nearest_count=20, voxel_size=0.005, transform=None):
        """
        obj_dir: Directory containing .obj mesh files.
        csv_file: Path to CSV file with eFAST landmark coordinates.
        num_points: Fixed number of points to output per sample.
        nearest_count: Number of nearest vertices to assign per landmark.
        voxel_size: The size of each voxel for downsampling.
        transform: Optional transformation applied to the point cloud.
        """
        self.obj_dir = obj_dir
        self.csv_data = pd.read_csv(csv_file)
        self.num_points = num_points
        self.nearest_count = nearest_count
        self.voxel_size = voxel_size
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
        points = mesh.vertices.astype(np.float32)  # shape: (N, 3)
        
        # Downsample points using voxel downsampling.
        points = voxel_downsampling(points, self.voxel_size)
        downsampled_count = points.shape[0]
        
        # Adjust the point cloud to have exactly self.num_points.
        if downsampled_count >= self.num_points:
            indices = np.random.choice(downsampled_count, self.num_points, replace=False)
            points = points[indices]
        else:
            extra_indices = np.random.choice(downsampled_count, self.num_points - downsampled_count, replace=True)
            points = np.concatenate([points, points[extra_indices]], axis=0)
        num_points = points.shape[0]  # should equal self.num_points

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
        
        # Initialize segmentation labels for each point as background (0).
        seg_labels = np.zeros(num_points, dtype=np.int64)
        
        # Build a KDTree on the fixed-size point cloud.
        tree = KDTree(points)
        # For each landmark, assign the nearest 'nearest_count' points the label (i+1).
        for i in range(6):
            landmark = landmarks[i]
            distances, nearest_idxs = tree.query(landmark, k=self.nearest_count)
            nearest_idxs = np.atleast_1d(nearest_idxs)
            seg_labels[nearest_idxs] = i + 1  # Labels 1 to 6
        
        # Optionally apply a transformation to the points.
        if self.transform:
            points = self.transform(torch.tensor(points, dtype=torch.float32)).numpy()
        
        # Convert to torch tensors.
        points = torch.tensor(points, dtype=torch.float32)
        seg_labels = torch.tensor(seg_labels, dtype=torch.long)
        return points, seg_labels

if __name__ == '__main__':
    # Example usage:
    obj_dir = r"path/to/obj_files"  # Update with your .obj directory
    csv_file = r"path/to/labels.csv"  # Update with your CSV file
    dataset = EFASTDataset(obj_dir, csv_file, num_points=2048, nearest_count=20, voxel_size=0.05)
    print("Dataset length:", len(dataset))
    sample_points, sample_labels = dataset[0]
    print("Points shape:", sample_points.shape)
    print("Unique labels in sample:", torch.unique(sample_labels))
