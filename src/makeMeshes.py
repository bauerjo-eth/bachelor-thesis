import smplx
import torch
import numpy as np
import trimesh
import os
import csv

# Define output folders for meshes and joints CSV.
mesh_output_folder = "smplx_meshes_test"
joints_output_folder = "smplx_joints_test"
os.makedirs(mesh_output_folder, exist_ok=True)
os.makedirs(joints_output_folder, exist_ok=True)

# CSV file path in the joints output folder.
joints_csv_path = os.path.join(joints_output_folder, "joints.csv")

# Load SMPL-X model
model_folder = r"models_smplx_v1_1\models"  # Adjust the path as needed.
model = smplx.create(model_folder, model_type='smplx', gender='neutral', use_pca=False)

num_meshes = 10  # Number of meshes to generate

# Open the CSV file for writing joint positions
with open(joints_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Run a forward pass once to determine the number of joints
    betas = torch.randn(1, 10) * 1
    body_pose = torch.randn(1, 63) * 0.05
    global_orient = torch.randn(1, 3) * 0.1
    output = model.forward(betas=betas, body_pose=body_pose, global_orient=global_orient)
    joints = output.joints.detach().cpu().numpy().squeeze()  # Expected shape: (num_joints, 3)
    n_joints = joints.shape[0]
    
    # Write CSV header: "filename, joint1_x, joint1_y, joint1_z, joint2_x, ..."
    header = ["filename"]
    for j in range(n_joints):
        header.extend([f"joint{j+1}_x", f"joint{j+1}_y", f"joint{j+1}_z"])
    writer.writerow(header)
    
    # Generate meshes and save corresponding joint positions.
    for i in range(num_meshes):
        # Generate random parameters for SMPL-X.
        betas = torch.randn(1, 10) 
        body_pose = torch.randn(1, 63) * 0.2
        global_orient = torch.randn(1, 3) * 0.1
        
        # Forward pass through the model.
        output = model.forward(betas=betas, body_pose=body_pose, global_orient=global_orient)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        faces = model.faces  # Faces are fixed for SMPL-X.
        
        # Save the mesh as an OBJ file with three-digit numbering.
        mesh_file = os.path.join(mesh_output_folder, f"smplx_mesh_{i+1:03d}.obj")
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(mesh_file)
        
        # Extract joint positions and flatten them.
        joints = output.joints.detach().cpu().numpy().squeeze()  # shape: (num_joints, 3)
        row = [os.path.basename(mesh_file)]
        row.extend(joints.flatten().tolist())
        writer.writerow(row)

print(f"Generated {num_meshes} SMPL-X meshes in '{mesh_output_folder}' and saved joint positions to '{joints_csv_path}'")
