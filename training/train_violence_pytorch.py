import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
import numpy as np
from pathlib import Path
import math

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- Device Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class ViolenceVideoDataset(Dataset):
    """PyTorch Dataset for Loading Violence Videos."""
    def __init__(self, video_paths, labels, num_frames=16, resize_dim=(224, 224), transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.resize_dim = resize_dim
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self._load_video(path)
        if len(frames) < self.num_frames:
            # Pad with last frame if too short
            last_frame = frames[-1] if frames else np.zeros((self.resize_dim[0], self.resize_dim[1], 3), dtype=np.float32)
            while len(frames) < self.num_frames:
                frames.append(last_frame)
        
        # Extract middle window
        mid = len(frames) // 2
        start = max(0, mid - self.num_frames // 2)
        window = frames[start:start + self.num_frames]
        
        # Stack and transpose to (B, C, H, W)
        # Final shape for PyTorch: (T, C, H, W)
        video_tensor = torch.stack(window)
        
        return video_tensor, torch.tensor(label, dtype=torch.float32)

    def _load_video(self, path):
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.resize_dim)
            
            # Apply normalization (as per MobileNetV2 standards)
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                
            frames.append(frame)
        cap.release()
        return frames

class ViolenceCNN_LSTM(nn.Module):
    """CNN-LSTM Architecture using MobileNetV2 + LSTM."""
    def __init__(self, num_frames=16, hidden_dim=128, num_layers=2):
        super(ViolenceCNN_LSTM, self).__init__()
        
        # MobileNetV2 backbone (frozen)
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(mobilenet.features))
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # LSTM Temporal Layer
        self.lstm = nn.LSTM(input_size=1280, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (Batch, Frames, C, H, W)
        batch_size, frames, c, h, w = x.shape
        
        # Reshape to (Batch * Frames, C, H, W) to process through CNN
        x = x.view(batch_size * frames, c, h, w)
        features = self.backbone(x)
        features = self.pool(features).view(batch_size * frames, -1)
        
        # Reshape back to (Batch, Frames, Features)
        features = features.view(batch_size, frames, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        
        # Take the last hidden state for classification
        out = self.fc(lstm_out[:, -1, :])
        return out

def train_model(epochs=30, batch_size=8, data_path=None):
    if data_path is None:
        data_path = os.path.join(config.BASE_DIR, "datasets", "violence")

    print(f"Indexing videos in {data_path}...")
    video_paths, labels = [], []
    for label, sub in enumerate(["violent", "non_violent"]):
        sub_dir = os.path.join(data_path, sub)
        if not os.path.exists(sub_dir): continue
        files = list(Path(sub_dir).glob("*.avi")) + list(Path(sub_dir).glob("*.mp4"))
        video_paths.extend(files)
        labels.extend([1.0 if sub == "violent" else 0.0] * len(files))

    if not video_paths:
        print("No videos found.")
        return

    print(f"Total: {len(video_paths)} videos. (Violent: {sum(labels)})")

    # Data transformation
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ViolenceVideoDataset(video_paths, labels, transform=preprocess)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ViolenceCNN_LSTM().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f"Starting Training on {DEVICE}...")
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        epoch_val_acc = val_correct / val_total
        print(f"--- Epoch {epoch+1} summary ---")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100*epoch_val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        # Save best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, "violence_detection_pytorch.pth"))
            print(f"✓ Saved Best Model (Acc: {100*best_acc:.2f}%)")

    print(f"Training Complete. Best Val Accuracy: {100*best_acc:.2f}%")

if __name__ == "__main__":
    train_model(epochs=30, batch_size=8)
