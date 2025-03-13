import smplx
import torch
import numpy as np
import trimesh
import os
import csv

# --- Configuration ---
model_folder = r"models_smplx_v1_1\models"  # Update as needed.
output_folder = "smplx_output"
os.makedirs(output_folder, exist_ok=True)

# --- Create the SMPL-X model ---
model = smplx.create(model_folder, model_type='smplx', gender='neutral', use_pca=False)

# --- Set parameters for a lying down pose with arms at the side ---
# Random shape parameters.
betas = torch.randn(1, 10)

# Use a controlled body pose: start with zeros.
body_pose = torch.zeros(1, 63, dtype=torch.float32)
# Adjust shoulder rotations (joints are 0-indexed):
# Joint 16 is the left shoulder: rotate by +75° about the z-axis.
body_pose[:, 15*3:15*3+3] = torch.tensor([[0, 0, -np.deg2rad(60)]], dtype=torch.float32)
# Joint 17 is the right shoulder: rotate by -75° about the z-axis.
body_pose[:, 16*3:16*3+3] = torch.tensor([[0, 0, np.deg2rad(60)]], dtype=torch.float32)

# Set the global orientation to place the body lying down.
global_orient = torch.tensor([[np.pi/2, 0, 0]], dtype=torch.float32)

# --- Forward pass through the model ---
output = model.forward(betas=betas, body_pose=body_pose, global_orient=global_orient)
vertices = output.vertices.detach().cpu().numpy().squeeze()
faces = model.faces  # Faces remain fixed.
joints = output.joints.detach().cpu().numpy().squeeze()  # (num_joints, 3)

# --- Save the mesh as an OBJ file ---
mesh_filename = os.path.join(output_folder, "smplx_mesh.obj")
mesh = trimesh.Trimesh(vertices, faces)
mesh.export(mesh_filename)
print(f"Mesh saved to {mesh_filename}")

# --- Save joint coordinates to a CSV file ---
joints_filename = os.path.join(output_folder, "smplx_joints.csv")
with open(joints_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header: joint_index, x, y, z.
    writer.writerow(["joint_index", "x", "y", "z"])
    for i, joint in enumerate(joints):
        writer.writerow([i] + joint.tolist())
print(f"Joint coordinates saved to {joints_filename}")
