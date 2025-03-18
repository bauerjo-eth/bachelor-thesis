# train.py
import os
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from model import LandmarkSegmentationNet
from dataloader import EFASTDataset
import numpy as np

# Optionally import tqdm for a progress bar.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from sklearn.model_selection import KFold

###########################################
# Configuration
###########################################
OBJ_DIR = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\ML\smplx_big_test\smplx_meshes"               # Directory containing .obj meshes.
CSV_FILE = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\ML\smplx_big_test\smplx_labels\labels_reordered.csv"             # CSV file with ground truth eFAST landmark coordinates.
CHECKPOINT_DIR = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\ML\src\Training\PointNet\Checkpoints\smplx_big_test"      # Directory to save model checkpoints.
NEAREST_COUNT = 3                           # Number of nearest vertices to assign as landmark per eFAST label.
NUM_POINTS = 2048                            # Fixed number of points per sample (downsample or duplicate as needed).

USE_KFOLD = False                            # Set to True to use K-Fold training.
NUM_FOLDS = 5                                # Number of folds if K-Fold is enabled.

BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

###########################################
# Loss Functions
###########################################
def segmentation_loss(seg_logits, seg_labels, class_weights=None):
    """
    Computes weighted cross-entropy loss.
    seg_logits: Tensor of shape (B, N, num_classes)
    seg_labels: Tensor of shape (B, N)
    class_weights: Tensor of shape (num_classes,)
    """
    B, N, num_classes = seg_logits.size()
    seg_logits = seg_logits.view(B * N, num_classes)
    seg_labels = seg_labels.view(B * N)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_fn(seg_logits, seg_labels)
    return loss

###########################################
# Training and Evaluation Functions
###########################################
def train_one_epoch(model, dataloader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0.0
    # Use tqdm for batch progress.
    for points, seg_labels in tqdm(dataloader, desc="Training Batches"):
        points = points.to(device)
        seg_labels = seg_labels.to(device).long()
        optimizer.zero_grad()
        seg_logits = model(points)
        loss = segmentation_loss(seg_logits, seg_labels, class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, class_weights=None):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for points, seg_labels in tqdm(dataloader, desc="Evaluating Batches"):
            points = points.to(device)
            seg_labels = seg_labels.to(device).long()
            seg_logits = model(points)
            loss = segmentation_loss(seg_logits, seg_labels, class_weights)
            total_loss += loss.item()
    return total_loss / len(dataloader)

###########################################
# Main Training Function
###########################################
def train_model(model, dataset, device, num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE, class_weights=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    
    if USE_KFOLD:
        kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
        fold = 1
        for train_indices, val_indices in kf.split(np.arange(len(dataset))):
            print(f"\nStarting fold {fold}/{NUM_FOLDS}")
            train_subset = data.Subset(dataset, train_indices)
            val_subset = data.Subset(dataset, val_indices)
            train_loader = data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = data.DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
            
            for epoch in range(num_epochs):
                print(f"Fold {fold} - Epoch {epoch+1}/{num_epochs}")
                train_loss = train_one_epoch(model, train_loader, optimizer, device, class_weights)
                val_loss = evaluate(model, val_loader, device, class_weights)
                print(f"Fold {fold} - Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold{fold}_{timestamp}.pth")
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Saved best model for fold {fold} at {checkpoint_path}")
            fold += 1
    else:
        # Simple train/validation split (80/20 split).
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
        train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss = train_one_epoch(model, train_loader, optimizer, device, class_weights)
            val_loss = evaluate(model, val_loader, device, class_weights)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_{timestamp}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved best model at {checkpoint_path}")
    
    return model

###########################################
# Main Script: Create DataLoader and Train
###########################################
if __name__ == '__main__':
    # Create checkpoint directory if it does not exist.
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    # Create the dataset with a fixed number of points.
    dataset = EFASTDataset(obj_dir=OBJ_DIR, csv_file=CSV_FILE, num_points=NUM_POINTS, nearest_count=NEAREST_COUNT)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    model = LandmarkSegmentationNet(num_classes=7).to(device)
    
    # Define class weights: weight 1 for background (label 0) and 5 for each landmark (labels 1-6).
    class_weights = torch.tensor([1, 5, 5, 5, 5, 5, 5], dtype=torch.float32).to(device)
    
    print("Starting training...")
    trained_model = train_model(model, dataset, device,
                                num_epochs=NUM_EPOCHS,
                                learning_rate=LEARNING_RATE,
                                class_weights=class_weights)
