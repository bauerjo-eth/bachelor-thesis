import smplx
import torch
import numpy as np
import trimesh
import os
import csv
import random

# --- Configuration ---
mesh_output_folder = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_final\smplx_big_meshes"
joints_output_folder = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_final\smplx_big_joints"
os.makedirs(mesh_output_folder, exist_ok=True)
os.makedirs(joints_output_folder, exist_ok=True)

# CSV file path in the joints output folder.
joints_csv_path = os.path.join(joints_output_folder, "joints.csv")

# Define the model folder and genders.
model_folder = r"models_smplx_v1_1\models"  # Adjust the path as needed.
genders = ['neutral', 'male', 'female']

# Load SMPL-X models for each gender once.
models = {}
for gender in genders:
    models[gender] = smplx.create(model_folder, model_type='smplx', gender=gender, use_pca=False)

num_meshes = 250  # Number of meshes to generate

# Define the joint indices for the shoulders (0-indexed):
# In SMPL-X, joint 16 is the left shoulder and joint 17 is the right shoulder.
left_shoulder_idx = 16 - 1   # 15 in 0-index
right_shoulder_idx = 17 - 1  # 16 in 0-index

# --- Prepare a reference pose for a lying down position with arms at the side ---
# Global orientation: rotate the entire body 90° about the x-axis (lying down).
global_orient = torch.tensor([[np.pi/2, 0, 0]], dtype=torch.float32)
# Start with a zero body pose (63 dims: 21 joints * 3 values) and then adjust the shoulders.
body_pose_ref = torch.zeros(1, 63, dtype=torch.float32)
# Rotate left shoulder (joint 16) by -60° about the z-axis.
body_pose_ref[:, left_shoulder_idx*3:(left_shoulder_idx*3+3)] = torch.tensor([[0, 0, -np.deg2rad(60)]], dtype=torch.float32)
# Rotate right shoulder (joint 17) by +60° about the z-axis.
body_pose_ref[:, right_shoulder_idx*3:(right_shoulder_idx*3+3)] = torch.tensor([[0, 0, np.deg2rad(60)]], dtype=torch.float32)

# Define the noise standard deviation for the body pose (in radians).
pose_noise_std = 0.05  # Adjust as needed

# --- Run a forward pass once to determine the number of joints ---
# (Using the 'neutral' model here for reference.)
betas = torch.randn(1, 10)  # Standard normal; typical range ~[-2,2] for most subjects.
output = models['neutral'].forward(betas=betas, body_pose=body_pose_ref, global_orient=global_orient)
joints = output.joints.detach().cpu().numpy().squeeze()  # Expected shape: (num_joints, 3)
n_joints = joints.shape[0]

# Write CSV header: add a column for gender.
with open(joints_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = ["filename", "gender"]
    for j in range(n_joints):
        header.extend([f"joint{j+1}_x", f"joint{j+1}_y", f"joint{j+1}_z"])
    writer.writerow(header)

# --- Generate meshes and save corresponding joint positions ---
for i in range(num_meshes):
    # Randomly choose a gender for this mesh.
    random_gender = random.choice(genders)
    model = models[random_gender]
    
    # Sample betas from a standard normal distribution.
    betas = torch.randn(1, 10) * 1  # Adjust multiplier if more variation is desired.
    
    # Create a noisy body pose by adding Gaussian noise to the reference pose.
    body_pose = body_pose_ref.clone() + torch.randn_like(body_pose_ref) * pose_noise_std
    
    # Global orientation remains fixed.
    global_orient = torch.tensor([[np.pi/2, 0, 0]], dtype=torch.float32)
    
    # Forward pass through the model.
    output = model.forward(betas=betas, body_pose=body_pose, global_orient=global_orient)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = model.faces  # Faces are fixed for SMPL-X.
    
    # Save the mesh as an OBJ file with three-digit numbering.
    mesh_file = os.path.join(mesh_output_folder, f"smplx_mesh_{i+1:03d}.obj")
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(mesh_file)
    
    # Extract joint positions.
    joints = output.joints.detach().cpu().numpy().squeeze()  # shape: (num_joints, 3)
    row = [os.path.basename(mesh_file), random_gender]
    row.extend(joints.flatten().tolist())
    
    # Append joint positions to CSV.
    with open(joints_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

print(f"Generated {num_meshes} SMPL-X meshes in '{mesh_output_folder}' and saved joint positions to '{joints_csv_path}'")
