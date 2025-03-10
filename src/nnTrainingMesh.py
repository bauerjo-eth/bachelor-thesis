import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
import os
import datetime
import trimesh

# --- Define the Dataset class at module level so it can be pickled ---
class MeshToEFastDataset(Dataset):
    def __init__(self, df, mesh_folder, num_sampled_points=1024):
        """
        df: DataFrame with at least a 'filename' column and eFAST label columns.
        mesh_folder: folder path containing the mesh OBJ files.
        num_sampled_points: number of points to sample from each mesh.
        """
        self.df = df.reset_index(drop=True)
        self.mesh_folder = mesh_folder
        self.num_sampled_points = num_sampled_points
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        mesh_path = os.path.join(self.mesh_folder, filename)
        # Load the mesh using trimesh.
        mesh = trimesh.load(mesh_path, force='mesh')
        # Sample a fixed number of points from the mesh surface.
        points = mesh.sample(self.num_sampled_points)  # shape: (num_sampled_points, 3)
        # Flatten to a 1D vector.
        points_flat = points.flatten().astype(np.float32)
        # Get the labels (all columns except "filename")
        labels = row[self.df.columns.difference(["filename"])].values.astype(np.float32)
        # Convert to tensors.
        X = torch.from_numpy(points_flat)
        Y = torch.from_numpy(labels)
        return X, Y

# --- Define the Neural Network Model ---
class MeshToEFastNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MeshToEFastNet, self).__init__()
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

def main():
    # --- Configuration: File paths ---
    mesh_folder = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_meshes"
    labels_csv_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_labels\labels.csv"
    checkpoint_dir = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\model_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # --- Hyperparameters for mesh processing ---
    num_sampled_points = 1024   # Number of points sampled per mesh.
    input_dim = num_sampled_points * 3  # Each point has 3 coordinates.
    
    # --- Device configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # --- Load Labels CSV using pandas ---
    labels_df = pd.read_csv(labels_csv_path)
    # Rename "Filename" to "filename" for consistency.
    labels_df.rename(columns={"Filename": "filename"}, inplace=True)
    
    # For outputs, assume all columns except "filename" are labels.
    output_columns = list(labels_df.columns)
    output_columns.remove("filename")
    output_dim = len(output_columns)
    print("Output columns count:", output_dim)  # Expected: 18
    
    # --- Create the full dataset ---
    full_dataset = MeshToEFastDataset(labels_df, mesh_folder, num_sampled_points=num_sampled_points)
    total_samples = len(full_dataset)
    print("Total samples in dataset:", total_samples)
    
    # --- Hold out Test Set ---
    indices = np.arange(total_samples)
    train_val_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_val_dataset = Subset(full_dataset, train_val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # --- K-Fold Cross Validation on Train/Val (k=5) ---
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # --- Training Hyperparameters ---
    epochs = 100
    learning_rate = 0.001
    batch_size = 32
    criterion = nn.MSELoss()
    
    fold_train_losses = []
    fold_val_losses = []
    
    print("Starting K-Fold Cross Validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
        print(f"Fold {fold+1}/{k}")
        train_subset = Subset(train_val_dataset, train_idx)
        val_subset = Subset(train_val_dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        model = MeshToEFastNet(input_dim, output_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch_X, batch_Y in train_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_Y = batch_Y.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_X.size(0)
            epoch_train_loss = running_loss / len(train_subset)
            if (epoch+1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}")
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_Y = batch_Y.to(device, non_blocking=True)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item() * batch_X.size(0)
        epoch_val_loss = val_loss / len(val_subset)
        print(f"Fold {fold+1} Validation Loss: {epoch_val_loss:.4f}")
        fold_train_losses.append(epoch_train_loss)
        fold_val_losses.append(epoch_val_loss)
    
    avg_train_loss = np.mean(fold_train_losses)
    avg_val_loss = np.mean(fold_val_losses)
    print(f"\nAverage Train Loss across folds: {avg_train_loss:.4f}")
    print(f"Average Validation Loss across folds: {avg_val_loss:.4f}")
    
    # --- Train Final Model on All Training+Validation Data ---
    print("\nTraining final model on all training+validation data...")
    final_model = MeshToEFastNet(input_dim, output_dim).to(device)
    final_optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)
    final_loader = DataLoader(
        train_val_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    for epoch in range(epochs):
        final_model.train()
        running_loss = 0.0
        for batch_X, batch_Y in final_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_Y = batch_Y.to(device, non_blocking=True)
            final_optimizer.zero_grad()
            outputs = final_model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            final_optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        final_epoch_loss = running_loss / len(train_val_dataset)
        if (epoch+1) % 20 == 0:
            print(f"Final Model Epoch {epoch+1}/{epochs} - Loss: {final_epoch_loss:.4f}")
    
    # Save the final model checkpoint with date and time in the filename.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f"checkpoint_{timestamp}.pth"
    final_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    torch.save(final_model.state_dict(), final_checkpoint_path)
    print(f"Final model checkpoint saved to {final_checkpoint_path}")
    
    # --- Evaluate on the Test Set ---
    final_model.eval()
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_Y = batch_Y.to(device, non_blocking=True)
            outputs = final_model(batch_X)
            loss = criterion(outputs, batch_Y)
            test_loss += loss.item() * batch_X.size(0)
    test_loss /= len(test_dataset)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    
    # --- Example Prediction on Test Samples ---
    final_model.eval()
    with torch.no_grad():
        sample_loader = DataLoader(
            test_dataset, 
            batch_size=5, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        sample_inputs, sample_labels = next(iter(sample_loader))
        sample_inputs = sample_inputs.to(device, non_blocking=True)
        predictions = final_model(sample_inputs)
        print("\nSample Predictions for first 5 test samples:")
        print(predictions.cpu())

if __name__ == '__main__':
    main()
