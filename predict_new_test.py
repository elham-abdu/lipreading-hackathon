import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
from step4_build_model import LipReadingModel
from dataset_loader import num_to_char
print("="*50)
print("Predicting on NEW Test Videos")
print("="*50)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = LipReadingModel().to(device)
checkpoint = torch.load('checkpoint_epoch_20.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✅ Model loaded successfully")

# Custom dataset for new test videos
class NewTestDataset(Dataset):
    def __init__(self, processed_folder, max_frames=100):
        self.processed_folder = processed_folder
        self.max_frames = max_frames
        self.videos = []
        
        # Find all video folders
        for video_id in os.listdir(processed_folder):
            video_path = os.path.join(processed_folder, video_id)
            if os.path.isdir(video_path):
                frames = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
                if frames:
                    self.videos.append({
                        'video_id': video_id,
                        'frames': frames,
                        'num_frames': len(frames)
                    })
        
        print(f"Found {len(self.videos)} videos")
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video = self.videos[idx]
        
        # Load frames
        frames = []
        video_path = os.path.join(self.processed_folder, video['video_id'])
        
        for frame_file in video['frames'][:self.max_frames]:
            img_path = os.path.join(video_path, frame_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            frames.append(img)
        
        # Pad if needed
        if len(frames) < self.max_frames:
            pad_frames = self.max_frames - len(frames)
            pad_img = np.zeros_like(frames[0])
            frames.extend([pad_img] * pad_frames)
        
        video_tensor = torch.FloatTensor(np.array(frames))
        
        return {
            'video': video_tensor,
            'video_id': video['video_id'],
            'num_frames': video['num_frames']
        }

def decode_predictions(output):
    """Convert model output to text"""
    predictions = []
    
    for batch_idx in range(output.shape[0]):
        pred_ids = torch.argmax(output[batch_idx], dim=-1).cpu().numpy()
        
        decoded = []
        prev = -1
        for idx in pred_ids:
            if idx != 0 and idx != prev:
                if idx in num_to_char:
                    decoded.append(num_to_char[idx])
            prev = idx
        
        predictions.append(''.join(decoded))
    
    return predictions

# Load dataset
print("\n📂 Loading processed test videos...")
dataset = NewTestDataset("processed_data/new_test", max_frames=100)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

# Generate predictions
print("\n🤖 Generating predictions...")
predictions = []
video_ids = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Predicting"):
        videos = batch['video'].to(device)
        batch_video_ids = batch['video_id']
        
        output = model(videos)
        batch_predictions = decode_predictions(output)
        
        predictions.extend(batch_predictions)
        video_ids.extend(batch_video_ids)

# Create submission
print("\n📝 Creating submission file...")

# Load sample submission to get correct format
sample_df = pd.read_csv("new_data/sample_submission.csv")

# Create mapping
pred_dict = dict(zip(video_ids, predictions))

# Fill predictions (sample_df already has correct paths like "00000.mp4")
sample_df['transcription'] = sample_df['path'].apply(
    lambda x: pred_dict.get(x.replace('.mp4', ''), '')
)

# Show sample
print("\n📊 Sample predictions:")
for i in range(min(10, len(sample_df))):
    if sample_df['transcription'].iloc[i]:
        print(f"   {sample_df['path'].iloc[i]}: {sample_df['transcription'].iloc[i]}")

# Save
sample_df.to_csv('submission_final.csv', index=False)
print("\n💾 Saved to submission_final.csv")

# Statistics
non_empty = sample_df['transcription'].str.len() > 0
print(f"\n📊 Statistics:")
print(f"   Total test videos: {len(sample_df)}")
print(f"   Videos with predictions: {non_empty.sum()}")
print(f"   Empty predictions: {(~non_empty).sum()}")

print("\n" + "="*50)
print("✅ Ready for final submission!")
print("="*50)