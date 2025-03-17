import os
import numpy as np

# Parameters
gaussian_std = 0.005  # Standard deviation for Gaussian noise on mesh vertices
background_noise_ratio = 0.2  # Fraction of the original vertex count to add as background noise

# Cube dimensions for generating cluster centers (e.g., within a 2x2x2 m cube)
cube_min = -1.0
cube_max = 1.0

# Parameters for clustered background noise
num_clusters = 5   # Number of clusters for background noise
cluster_std = 0.3  # Standard deviation for points within each cluster

# Input and output folders
input_folder = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_small_test\smplx_meshes_test"
output_folder = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_small_test\smplx_meshes_with_noise"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def generate_clustered_background_noise(num_points, num_clusters=8, cluster_std=0.05):
    """
    Generate background noise as clusters of points.
    
    Args:
        num_points (int): Total number of background noise points to generate.
        num_clusters (int): Number of clusters.
        cluster_std (float): Standard deviation for points in each cluster.
        
    Returns:
        numpy.ndarray: Array of shape (num_points, 3) containing noise points.
    """
    # Calculate points per cluster and distribute any remainder
    points_per_cluster = num_points // num_clusters
    remainder = num_points % num_clusters
    noise_points = []
    for i in range(num_clusters):
        # Random center for the cluster within the defined cube
        center = np.random.uniform(low=cube_min, high=cube_max, size=(3,))
        # Determine the number of points for this cluster
        num_points_this_cluster = points_per_cluster + (1 if i < remainder else 0)
        # Generate points around the center using a Gaussian distribution
        cluster_points = np.random.normal(loc=center, scale=cluster_std, size=(num_points_this_cluster, 3))
        noise_points.append(cluster_points)
    return np.concatenate(noise_points, axis=0)

def process_obj_file(file_path, output_path):
    """
    Reads an .obj file, applies Gaussian noise to its vertex coordinates,
    adds clustered background noise vertices, and writes the result to a new file.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Separate vertex lines from other lines
    vertex_lines = []
    other_lines = []
    for line in lines:
        if line.startswith("v "):
            vertex_lines.append(line)
        else:
            other_lines.append(line)
    
    # Parse vertices from vertex lines
    vertices = []
    for v_line in vertex_lines:
        parts = v_line.split()
        if len(parts) >= 4:
            x, y, z = map(float, parts[1:4])
            vertices.append([x, y, z])
    vertices = np.array(vertices)

    # Apply Gaussian noise to the mesh vertices
    noisy_vertices = vertices + np.random.normal(loc=0.0, scale=gaussian_std, size=vertices.shape)

    # Calculate the number of background noise points to add
    num_bg_points = int(background_noise_ratio * len(noisy_vertices))
    # Generate clustered background noise vertices
    background_noise = generate_clustered_background_noise(num_bg_points, num_clusters=num_clusters, cluster_std=cluster_std)

    # Write the new .obj file with the noisy vertices and background noise
    with open(output_path, "w") as out_file:
        # Write the noisy vertices from the mesh
        for v in noisy_vertices:
            out_file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        # Append the background noise vertices
        for v in background_noise:
            out_file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        # Write the remaining lines (faces, texture coordinates, etc.)
        for line in other_lines:
            out_file.write(line)

def main():
    # Process all .obj files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".obj"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            print("Processing:", filename)
            process_obj_file(input_path, output_path)
    print("Processing complete. Noisy files are in '{}'.".format(output_folder))

if __name__ == "__main__":
    main()
