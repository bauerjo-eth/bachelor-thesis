import smplx
import torch
import numpy as np
import trimesh
import os

# Define output folder
output_folder = "smplx_meshes_test"
os.makedirs(output_folder, exist_ok=True)

# Load SMPL-X model
# Adjust the path to point to the parent folder that contains the "smplx" subfolder.
model_folder = r"models_smplx_v1_1\models"  # Using a raw string for Windows paths
model = smplx.create(model_folder, model_type='smplx', gender='neutral', use_pca=False)

# Generate 10 different random SMPL-X meshes
num_meshes = 10

for i in range(num_meshes):
    # Generate random shape and pose parameters
    betas = torch.randn(1, 10) * 0.03  # Shape variations
    body_pose = torch.randn(1, 63) * 0.05  # Small pose variations
    global_orient = torch.randn(1, 3) * 0.1  # Global orientation variations

    # Generate the mesh
    output = model.forward(betas=betas, body_pose=body_pose, global_orient=global_orient)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = model.faces  # Faces are fixed for SMPL-X

    # Save as .obj file
    mesh = trimesh.Trimesh(vertices, faces)
    mesh_file = os.path.join(output_folder, f"smplx_mesh_{i+1}.obj")
    mesh.export(mesh_file)

print(f"Generated {num_meshes} SMPL-X meshes in '{output_folder}'")
