import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

############################################
# 1. Define the Dataset that Loads the CSVs
############################################

class JointsLabelDataset(Dataset):
    def __init__(self, joints_csv, labels_csv):
        # Load CSV files with pandas
        joints_df = pd.read_csv(joints_csv)
        labels_df = pd.read_csv(labels_csv)
        
        # Rename filename column if needed so that both DataFrames share the same key.
        if "filename" in joints_df.columns:
            joints_df.rename(columns={"filename": "Filename"}, inplace=True)
        
        # Merge on "Filename"
        self.df = pd.merge(joints_df, labels_df, on="Filename")
        
        # Determine number of joints and label points.
        # joints.csv: one column for Filename + 3 columns per joint.
        self.n_joints = (len(joints_df.columns) - 1) // 3
        # labels.csv: one column for Filename + 3 columns per label point.
        self.n_labels = (len(labels_df.columns) - 1) // 3

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Select joint columns (those that start with "joint")
        joint_cols = [col for col in self.df.columns if col.startswith("joint")]
        joints = row[joint_cols].values.astype(np.float32).reshape(self.n_joints, 3)
        
        # Select label columns (those that start with "P", e.g., P1_x, P1_y, ...)
        label_cols = [col for col in self.df.columns if col.startswith("P")]
        labels = row[label_cols].values.astype(np.float32).reshape(self.n_labels, 3)
        
        return torch.tensor(joints), torch.tensor(labels)

############################################
# 2. Define the UNet1D Model
############################################

class UNet1D(nn.Module):
    def __init__(self, in_channels=3, base_filters=64):
        super(UNet1D, self).__init__()
        # Encoder stage 1
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool1d(2)
        # Encoder stage 2
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_filters, base_filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_filters*2, base_filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool1d(2)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_filters*4, base_filters*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Decoder stage 2 (using output_padding for proper upsampling)
        self.up2 = nn.ConvTranspose1d(base_filters*4, base_filters*2, kernel_size=2, stride=2, output_padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv1d(base_filters*4, base_filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_filters*2, base_filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Decoder stage 1 (using output_padding)
        self.up1 = nn.ConvTranspose1d(base_filters*2, base_filters, kernel_size=2, stride=2, output_padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_filters*2, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Final convolution: map to 3 channels (x, y, z)
        self.final_conv = nn.Conv1d(base_filters, 3, kernel_size=1)
        # Adaptive pooling to force output to a fixed length of 6 (label points)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(6)
        
    def forward(self, x):
        # x shape: (batch, n_joints, 3)
        x = x.permute(0, 2, 1)  # -> (batch, 3, n_joints)
        enc1 = self.enc1(x)                     # -> (batch, base_filters, n_joints)
        enc2 = self.enc2(self.pool1(enc1))        # -> (batch, base_filters*2, n_joints/2)
        bottleneck = self.bottleneck(self.pool2(enc2))  # -> (batch, base_filters*4, n_joints/4)
        up2 = self.up2(bottleneck)                # -> (batch, base_filters*2, n_joints/2)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)                      # -> (batch, base_filters, n_joints)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        out = self.final_conv(dec1)               # -> (batch, 3, n_joints)
        out = self.adaptive_pool(out)             # -> (batch, 3, 6)
        out = out.permute(0, 2, 1)                # -> (batch, 6, 3)
        return out

############################################
# 3. Train-Test Split and Training Setup
############################################

if __name__ == '__main__':
    # Update these paths to point to your CSV files:
    joints_csv = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_joints_test\joints.csv"
    labels_csv = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_labels_test\labels.csv"
    
    # Create the full dataset.
    dataset = JointsLabelDataset(joints_csv, labels_csv)
    
    # Split into training (80%) and testing (20%)
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create DataLoaders for each split.
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Initialize the model, loss function, and optimizer.
    model = UNet1D(in_channels=3, base_filters=64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 25
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for joints, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(joints)  # outputs shape: (batch, 6, 3)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * joints.size(0)
        train_loss = running_loss / train_size
        
        # Validation on the test set.
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for joints, labels in test_loader:
                outputs = model(joints)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * joints.size(0)
        val_loss = running_val_loss / test_size
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
