import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataset_loader import LipReadingDataset, char_list

print("="*50)
print("STEP 4: Building Lip Reading Model")
print("="*50)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Constants
NUM_CLASSES = len(char_list) + 1  # +1 for CTC blank
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2

class LipReadingModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE):
        super().__init__()
        
        # CNN for feature extraction from each frame
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 48x48
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 24x24
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 12x12
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 1x1
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
            bidirectional=True
        )
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        # Reshape for CNN
        x = x.view(-1, c, h, w)
        
        # Extract features
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.squeeze(-1).squeeze(-1)
        
        # Reshape back to sequence
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_features)
        
        # Classify each timestep
        output = self.classifier(lstm_out)
        
        # Log softmax for CTC loss
        output = nn.functional.log_softmax(output, dim=-1)
        
        return output

def collate_fn(batch):
    videos = torch.stack([item['video'] for item in batch])
    transcripts = [item['transcript'] for item in batch]
    transcript_lengths = torch.LongTensor([len(t) for t in transcripts])
    
    transcripts_padded = torch.nn.utils.rnn.pad_sequence(
        transcripts, batch_first=True, padding_value=0
    )
    
    return {
        'video': videos,
        'transcript': transcripts_padded,
        'transcript_lengths': transcript_lengths,
        'video_ids': [item['video_id'] for item in batch]
    }

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        videos = batch['video'].to(device)
        transcripts = batch['transcript'].to(device)
        transcript_lengths = batch['transcript_lengths'].to(device)
        
        video_lengths = torch.full(
            (videos.size(0),), videos.size(1), dtype=torch.long
        ).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(videos)
        output = output.permute(1, 0, 2)
        
        # Compute loss
        loss = criterion(
            output, 
            transcripts,
            video_lengths,
            transcript_lengths
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    return total_loss / len(dataloader)

# Test the model
print("\n🔧 Creating model...")
model = LipReadingModel().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
print("\n🔄 Testing forward pass...")
dataset = LipReadingDataset('train', csv_file='all_videos.csv', max_frames=100)
dataloader = DataLoader(
    dataset, 
    batch_size=4, 
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

# Get a batch
batch = next(iter(dataloader))
videos = batch['video'].to(device)
print(f"Input shape: {videos.shape}")

# Forward pass
output = model(videos)
print(f"Output shape: {output.shape}")
print(f"Output should be (batch, time, num_classes={NUM_CLASSES})")

# Test CTC loss
print("\n📉 Testing CTC loss...")
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
print(f"\n✅ Training step successful! Loss: {loss:.4f}")

print("\n" + "="*50)
print("✅ Model is ready for training!")
print("="*50)