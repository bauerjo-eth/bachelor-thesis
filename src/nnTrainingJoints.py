import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
import os
import datetime

# --- Configuration: File paths ---
joints_csv_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_joints\joints.csv"
labels_csv_path = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_labels\labels_reordered.csv"

# Define checkpoint folder.
checkpoint_dir = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\model_checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# --- Load CSV data using pandas ---
joints_df = pd.read_csv(joints_csv_path)
labels_df = pd.read_csv(labels_csv_path)
labels_df.rename(columns={"Filename": "filename"}, inplace=True)

# --- Merge dataframes on filename ---
data_df = pd.merge(joints_df, labels_df, on="filename")

# --- Determine Input and Output Columns ---
# Inputs: all columns from joints_df except "filename" and "gender"
input_columns = list(joints_df.columns)
input_columns.remove("filename")
input_columns.remove("gender")
# Outputs: all columns from labels_df except "filename"
output_columns = list(labels_df.columns)
output_columns.remove("filename")

print("Input columns count:", len(input_columns))    # e.g., 127*3 = 381
print("Output columns count:", len(output_columns))    # e.g., 6*3 = 18

# Convert data to NumPy arrays (float32)
X = data_df[input_columns].values.astype(np.float32)
Y = data_df[output_columns].values.astype(np.float32)
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# --- Hold out Test Set ---
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- Define a PyTorch Dataset ---
class JointToEFastDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_val_dataset = JointToEFastDataset(X_train_val, Y_train_val)
test_dataset = JointToEFastDataset(X_test, Y_test)

# --- Define the Neural Network Model ---
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

input_dim = X.shape[1]   # e.g., 381
output_dim = Y.shape[1]  # e.g., 18

# --- Training Hyperparameters ---
epochs = 100
learning_rate = 0.001
batch_size = 32
criterion = nn.MSELoss()

# --- K-Fold Cross Validation on Train/Val (k=5) ---
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold_train_losses = []
fold_val_losses = []

print("Starting K-Fold Cross Validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
    print(f"Fold {fold+1}/{k}")
    train_subset = Subset(train_val_dataset, train_idx)
    val_subset = Subset(train_val_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    model = JointToEFastNet(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_Y in train_loader:
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

# --- Final Evaluation on Hold-out Test Set is done later.
# --- Train Final Model on All Training+Validation Data ---
print("\nTraining final model on all training+validation data...")
final_model = JointToEFastNet(input_dim, output_dim)
final_optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)
final_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    final_model.train()
    running_loss = 0.0
    for batch_X, batch_Y in final_loader:
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
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
torch.save(final_model.state_dict(), checkpoint_path)
print(f"Final model checkpoint saved to {checkpoint_path}")

# --- Evaluate on the Test Set ---
final_model.eval()
test_loss = 0.0
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        outputs = final_model(batch_X)
        loss = criterion(outputs, batch_Y)
        test_loss += loss.item() * batch_X.size(0)
test_loss /= len(test_dataset)
print(f"\nFinal Test Loss: {test_loss:.4f}")

# --- Example Prediction on Test Samples ---
final_model.eval()
with torch.no_grad():
    sample_inputs = torch.from_numpy(X_test[:5])
    predictions = final_model(sample_inputs)
    print("\nSample Predictions for first 5 test samples:")
    print(predictions)
