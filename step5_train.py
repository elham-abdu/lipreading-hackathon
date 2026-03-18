import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataset_loader import LipReadingDataset, char_list
from step4_build_model import LipReadingModel, collate_fn
import os
from tqdm import tqdm

print("="*50)
print("STEP 5: Training Lip Reading Model")
print("="*50)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.001
MAX_FRAMES = 100

# Load dataset
print("\n📂 Loading dataset...")
train_dataset = LipReadingDataset('train', csv_file='all_videos.csv', max_frames=MAX_FRAMES)
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

print(f"   Training samples: {len(train_dataset)}")
print(f"   Batches per epoch: {len(train_loader)}")

# Initialize model
model = LipReadingModel().to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

# Training loop
print("\n🎯 Starting training...")
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        videos = batch['video'].to(device)
        transcripts = batch['transcript'].to(device)
        transcript_lengths = batch['transcript_lengths'].to(device)
        
        # Video lengths (all max_frames)
        video_lengths = torch.full(
            (videos.size(0),), videos.size(1), dtype=torch.long
        ).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(videos)
        output = output.permute(1, 0, 2)  # (time, batch, num_classes)
        
        # Compute loss
        loss = criterion(output, transcripts, video_lengths, transcript_lengths)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    scheduler.step(avg_loss)
    
    print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoint_epoch_{epoch+1}.pt')
        print(f"   💾 Saved checkpoint_epoch_{epoch+1}.pt")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.show()

print("\n" + "="*50)
print("✅ Training complete!")
print(f"Final loss: {train_losses[-1]:.4f}")
print("="*50)

# Save final model
torch.save(model.state_dict(), 'lipreading_model_final.pt')
print("💾 Model saved to lipreading_model_final.pt")