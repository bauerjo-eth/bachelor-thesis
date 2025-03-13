import os
import datetime
import numpy as np
import pandas as pd
import trimesh
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold

# =====================
# === Configuration ===
# =====================

# Paths (update these to match your setup)
MESH_FOLDER = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_meshes"
LABELS_CSV_PATH = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_labels\labels_reordered.csv"
CHECKPOINT_DIR = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\model_checkpoints"
# If you pre-sampled the meshes, set PRESAMPLED_DIR to the folder with the .npy files;
# otherwise set it to None to sample on the fly.
PRESAMPLED_DIR = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\presampled_meshes\32_points"

# Sampling configuration: choose the number of points to sample from each mesh.
NUM_SAMPLED_POINTS = 32

# Training configuration
USE_KFOLD = False        # Set to False to use a simple train/validation split.
K_FOLDS = 5              # Only used if USE_KFOLD is True.
TEST_SIZE = 0.2          # Fraction of samples held out for testing.
EPOCHS = 60
LEARNING_RATE = 0.001
BATCH_SIZE = 10

# ================================
# === Dataset and Model Classes ===
# ================================

class MeshToEFastDataset(Dataset):
    def __init__(self, df, mesh_folder, num_sampled_points=NUM_SAMPLED_POINTS, presample_dir=None):
        """
        df: DataFrame with at least a 'filename' column and eFAST label columns.
        mesh_folder: folder path containing the mesh OBJ files.
        num_sampled_points: number of points to sample from each mesh.
        presample_dir: if provided, loads pre-sampled numpy arrays instead of sampling on the fly.
        """
        self.df = df.reset_index(drop=True)
        self.mesh_folder = mesh_folder
        self.num_sampled_points = num_sampled_points
        self.presample_dir = presample_dir

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        
        if self.presample_dir is not None:
            # Try to load pre-sampled points
            pre_file = os.path.join(self.presample_dir, filename.replace(".obj", ".npy"))
            if os.path.exists(pre_file):
                points = np.load(pre_file)
            else:
                # Fall back to on-the-fly sampling if file is not found
                mesh_path = os.path.join(self.mesh_folder, filename)
                mesh = trimesh.load(mesh_path, force='mesh')
                points = mesh.sample(self.num_sampled_points)
        else:
            # Sample points from the mesh on the fly.
            mesh_path = os.path.join(self.mesh_folder, filename)
            mesh = trimesh.load(mesh_path, force='mesh')
            points = mesh.sample(self.num_sampled_points)

        # Flatten the points to a 1D vector.
        points_flat = points.flatten().astype(np.float32)
        # Get the label values (assume all columns except "filename" are labels).
        label_cols = [col for col in self.df.columns if col != "filename"]
        labels = row[label_cols].values.astype(np.float32)
        
        X = torch.from_numpy(points_flat)
        Y = torch.from_numpy(labels)
        return X, Y

class MeshToEFastNet(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(MeshToEFastNet, self).__init__()
        # Reduced network size with dropout for regularization.
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ======================
# === Training Logic ===
# ======================

def train_model(model, optimizer, criterion, loader, device):
    model.train()
    running_loss = 0.0
    for batch_X, batch_Y in loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_Y = batch_Y.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    return running_loss

def evaluate_model(model, criterion, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_Y = batch_Y.to(device, non_blocking=True)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            total_loss += loss.item() * batch_X.size(0)
    return total_loss

def main():
    # --- Device configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # --- Ensure checkpoint directory exists ---
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    # --- Load the labels CSV ---
    labels_df = pd.read_csv(LABELS_CSV_PATH)
    # Rename "Filename" to "filename" if needed.
    if "Filename" in labels_df.columns:
        labels_df.rename(columns={"Filename": "filename"}, inplace=True)
    
    # Determine output dimensions (all columns except "filename")
    output_columns = [col for col in labels_df.columns if col != "filename"]
    output_dim = len(output_columns)
    print("Output dimensions (number of labels):", output_dim)
    
    # --- Create the full dataset ---
    full_dataset = MeshToEFastDataset(
        labels_df, 
        MESH_FOLDER, 
        num_sampled_points=NUM_SAMPLED_POINTS, 
        presample_dir=PRESAMPLED_DIR
    )
    total_samples = len(full_dataset)
    print("Total samples in dataset:", total_samples)
    
    # --- Split into training+validation and test sets ---
    indices = np.arange(total_samples)
    train_val_indices, test_indices = train_test_split(indices, test_size=TEST_SIZE, random_state=42)
    train_val_dataset = Subset(full_dataset, train_val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # --- Define loss criterion ---
    criterion = nn.MSELoss()
    input_dim = NUM_SAMPLED_POINTS * 3  # Each point has 3 coordinates.
    
    # --- Training Phase ---
    if USE_KFOLD:
        # ----- K-Fold Cross Validation Training -----
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        best_val_loss = float('inf')
        best_model_state = None
        
        print("Starting K-Fold Cross Validation...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
            print(f"\nFold {fold+1}/{K_FOLDS}")
            train_subset = Subset(train_val_dataset, train_idx)
            val_subset = Subset(train_val_dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            
            model = MeshToEFastNet(input_dim, output_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            for epoch in range(EPOCHS):
                running_loss = train_model(model, optimizer, criterion, train_loader, device)
                epoch_train_loss = running_loss / len(train_subset)
                if (epoch+1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.4f}")
            
            val_loss_total = evaluate_model(model, criterion, val_loader, device)
            epoch_val_loss = val_loss_total / len(val_subset)
            print(f"Fold {fold+1} Validation Loss: {epoch_val_loss:.4f}")
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = model.state_dict()
        
        # Use the best model from k-fold training.
        final_model = MeshToEFastNet(input_dim, output_dim).to(device)
        final_model.load_state_dict(best_model_state)
        print(f"\nBest Validation Loss from K-Fold: {best_val_loss:.4f}")
    
    else:
        # ----- Single Train/Validation Split -----
        train_subset, val_subset = train_test_split(train_val_dataset, test_size=0.2, random_state=42)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        print("Starting training on train/validation split...")
        final_model = MeshToEFastNet(input_dim, output_dim).to(device)
        optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            running_loss = train_model(final_model, optimizer, criterion, train_loader, device)
            epoch_train_loss = running_loss / len(train_subset)
            if (epoch+1) % 4 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.4f}")
        
        val_loss_total = evaluate_model(final_model, criterion, val_loader, device)
        epoch_val_loss = val_loss_total / len(val_subset)
        print(f"Validation Loss: {epoch_val_loss:.4f}")
    
    # Save the model checkpoint with a timestamp.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f"checkpoint_{timestamp}.pth"
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
    torch.save(final_model.state_dict(), final_checkpoint_path)
    print(f"\nModel checkpoint saved to {final_checkpoint_path}")
    
    # ----- Evaluate on the Test Set -----
    final_model.eval()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loss_total = evaluate_model(final_model, criterion, test_loader, device)
    test_loss = test_loss_total / len(test_dataset)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    
    # ----- Example Prediction on Test Samples -----
    with torch.no_grad():
        sample_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=4, pin_memory=True)
        sample_inputs, sample_labels = next(iter(sample_loader))
        sample_inputs = sample_inputs.to(device, non_blocking=True)
        predictions = final_model(sample_inputs)
        print("\nSample Predictions for first 5 test samples:")
        print(predictions.cpu())

if __name__ == '__main__':
    main()
