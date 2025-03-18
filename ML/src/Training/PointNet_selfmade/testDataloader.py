import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time  # For timing
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataloader import EFASTDataset

# -------------------------------
# Configuration
# -------------------------------
OBJ_DIR = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\ML\smplx_big_test\smplx_meshes"  # Directory containing your .obj meshes.
CSV_FILE = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\ML\smplx_big_test\smplx_labels\labels_reordered.csv"  # CSV file with ground truth eFAST landmark coordinates.
NEAREST_COUNT = 3                    # Number of nearest vertices to assign per landmark.
NUM_POINTS = 4096                    # Number of points per sample (if your dataloader supports it)

def visualize_sample(points, labels):
    """
    Visualizes a 3D point cloud with colors based on segmentation labels.
    points: (N, 3) numpy array.
    labels: (N,) numpy array with integer labels.
    """
    # Define colors for each label: background=gray, landmarks 1-6 get different colors.
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
    
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        # Set marker sizes: background points smaller, others bigger.
        marker_size = 2 if lab == 0 else 20
        ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2],
                   c=color_map.get(lab, 'black'),
                   label=f"Label {lab}",
                   s=marker_size)
    ax.set_title("EFAST Dataset Sample Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.legend()
    plt.show()

def main():
    # Create the dataset.
    dataset = EFASTDataset(obj_dir=OBJ_DIR, csv_file=CSV_FILE, num_points=NUM_POINTS, nearest_count=NEAREST_COUNT)
    print("Dataset length:", len(dataset))
    
    # Measure the load time for the first sample.
    start_time = time.time()
    points, seg_labels = dataset[0]
    load_time = time.time() - start_time
    print(f"Loaded sample in {load_time:.4f} seconds")
    
    print("Points shape:", points.shape)
    print("Unique labels:", torch.unique(seg_labels))
    
    # Convert tensors to numpy arrays if necessary.
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(seg_labels):
        seg_labels = seg_labels.cpu().numpy()
    
    # Visualize the sample.
    visualize_sample(points, seg_labels)

if __name__ == '__main__':
    main()
