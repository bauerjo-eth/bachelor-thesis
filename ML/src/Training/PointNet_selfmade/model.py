# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################
# Configuration
#############################################
# Set the fixed number of points per input sample.
NUM_POINTS = 2048  # Change this value as needed.

#############################################
# Helper Function for Downsampling/Upsampling
#############################################
def downsample_points(x, num_points):
    """
    Downsamples or upsamples the input point cloud x to have exactly num_points.
    
    Args:
        x: Tensor of shape (B, N, 3)
        num_points: Desired number of points per sample.
        
    Returns:
        x_down: Tensor of shape (B, num_points, 3)
    """
    B, N, C = x.size()
    if N == num_points:
        return x
    indices_list = []
    for b in range(B):
        if N > num_points:
            idx = torch.randperm(N, device=x.device)[:num_points]
        else:
            # If there are fewer points than required, sample with replacement.
            idx = torch.randint(0, N, (num_points,), device=x.device)
        indices_list.append(idx)
    indices = torch.stack(indices_list, dim=0)  # (B, num_points)
    # Expand indices to match the last dimension (C) and gather.
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, C)
    x_down = torch.gather(x, 1, indices_expanded)
    return x_down

#############################################
# Network Modules
#############################################
class PointNetBackbone(nn.Module):
    """
    A simplified PointNet backbone that processes an input point cloud of shape (B, N, 3)
    and produces per-point features and a global feature.
    """
    def __init__(self):
        super(PointNetBackbone, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        # x: (B, N, 3); transpose to (B, 3, N) for conv1d layers.
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))    # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))      # (B, 128, N)
        point_feats = F.relu(self.bn3(self.conv3(x)))  # (B, 256, N)
        # Global feature via max pooling (over dimension 2).
        global_feat, _ = torch.max(point_feats, 2)  # (B, 256)
        # Transpose point_feats back to (B, N, 256)
        point_feats = point_feats.transpose(2, 1)
        return point_feats, global_feat

class SegmentationHead(nn.Module):
    """
    Segmentation branch that outputs per-point logits.
    Input: concatenated per-point features (256-dim) and repeated global features (256-dim) = 512-dim.
    Output: per-point class logits for 7 classes (0: background, 1–6: landmarks).
    """
    def __init__(self, in_channels=512, num_classes=7):
        super(SegmentationHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: (B, N, in_channels)
        B, N, _ = x.size()
        x = x.view(B * N, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        logits = self.fc3(x)
        logits = logits.view(B, N, -1)
        return logits

class LandmarkSegmentationNet(nn.Module):
    """
    Full network that uses the PointNet backbone and segmentation head.
    The network downsample/upsamples input point clouds to a fixed number of points (NUM_POINTS) before processing.
    """
    def __init__(self, num_classes=7, num_points=NUM_POINTS):
        super(LandmarkSegmentationNet, self).__init__()
        self.num_points = num_points
        self.backbone = PointNetBackbone()
        self.seg_head = SegmentationHead(in_channels=512, num_classes=num_classes)
        
    def forward(self, x):
        # x: (B, N, 3) where N can vary.
        # Downsample (or upsample) to self.num_points.
        x = downsample_points(x, self.num_points)  # Now x has shape (B, num_points, 3)
        point_feats, global_feat = self.backbone(x)  # point_feats: (B, num_points, 256); global_feat: (B, 256)
        B, N, _ = point_feats.size()  # N is now self.num_points.
        global_feat_expanded = global_feat.unsqueeze(1).repeat(1, N, 1)
        seg_input = torch.cat([point_feats, global_feat_expanded], dim=2)  # (B, N, 512)
        seg_logits = self.seg_head(seg_input)  # (B, N, num_classes)
        return seg_logits

if __name__ == '__main__':
    # Quick test with dummy input.
    dummy_input = torch.rand(8, 3000, 3)  # Batch of 8, with 3000 points per sample.
    model = LandmarkSegmentationNet(num_classes=7, num_points=NUM_POINTS)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (8, NUM_POINTS, 7)
