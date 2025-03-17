import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Define the Neural Network Model (same architecture as used during training) ---
class JointToEFastNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(JointToEFastNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# --- Configuration: File Paths and Settings ---
mesh_folder = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_meshes"   # Folder containing mesh OBJ files.
joints_csv_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_joints\joints.csv"
labels_csv_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_labels\labels.csv"
checkpoint_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\model_checkpoints\checkpoint_20250310_163120.pth"  # Path to the saved checkpoint.
selected_id = 27  # Change this (0-indexed) to select a different sample.

# --- Load CSV Data ---
joints_df = pd.read_csv(joints_csv_path)
labels_df = pd.read_csv(labels_csv_path)
labels_df.rename(columns={"Filename": "filename"}, inplace=True)

# Merge the dataframes on filename.
data_df = pd.merge(joints_df, labels_df, on="filename")

# --- Determine Input and Output Columns ---
input_columns = list(joints_df.columns)
input_columns.remove("filename")
input_columns.remove("gender")
output_columns = list(labels_df.columns)
output_columns.remove("filename")

print("Number of input features:", len(input_columns))   # e.g., 381
print("Number of output features:", len(output_columns))   # e.g., 18

# --- Select Sample ---
if selected_id < 0 or selected_id >= len(data_df):
    raise ValueError("Selected ID is out of range!")
sample_row = data_df.iloc[selected_id]
mesh_filename = sample_row["filename"]
mesh_path = os.path.join(mesh_folder, mesh_filename)

# --- Load Mesh ---
mesh = trimesh.load(mesh_path)
vertices = mesh.vertices

# --- Prepare Input for Prediction ---
joint_values = sample_row[input_columns].values.astype(np.float32)
joint_tensor = torch.from_numpy(joint_values).unsqueeze(0)  # shape (1, input_dim)

input_dim = len(input_columns)   # e.g., 381
output_dim = len(output_columns) # e.g., 18

# --- Load the Trained Model from Checkpoint ---
model = JointToEFastNet(input_dim, output_dim)
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print("Loaded model checkpoint from", checkpoint_path)
else:
    raise FileNotFoundError("Checkpoint file not found!")
model.eval()

# --- Predict eFAST Points ---
with torch.no_grad():
    prediction = model(joint_tensor)  # shape (1, output_dim)
prediction = prediction.squeeze(0).numpy()  # shape (output_dim,)
predicted_points = prediction.reshape(-1, 3)  # (6, 3)

# --- Get Ground Truth eFAST Points ---
ground_truth = sample_row[output_columns].values.astype(np.float32)
ground_truth_points = ground_truth.reshape(-1, 3)

# --- Visualization ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot mesh vertices.
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
           s=0.1, color='gray', label='Mesh Vertices')

# Plot ground truth eFAST points (red circles).
ax.scatter(ground_truth_points[:, 0], ground_truth_points[:, 1], ground_truth_points[:, 2],
           s=100, color='red', marker='o', label='Ground Truth eFAST Points')

# Plot predicted eFAST points (blue squares).
ax.scatter(predicted_points[:, 0], predicted_points[:, 1], predicted_points[:, 2],
           s=100, color='blue', marker='s', label='Predicted eFAST Points')

# Annotate each point.
for i, (gt, pred) in enumerate(zip(ground_truth_points, predicted_points)):
    ax.text(gt[0], gt[1], gt[2], f'GT{i+1}', color='darkred', fontsize=10)
    ax.text(pred[0], pred[1], pred[2], f'P{i+1}', color='darkblue', fontsize=10)

# Set all axes from -1 to 1.
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

ax.set_title(f"Mesh and eFAST Points for {mesh_filename}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()
