import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from dataset_loader import LipReadingDataset, char_list, num_to_char
from step4_build_model import LipReadingModel, collate_fn

print("="*50)
print("STEP 6: Generating Predictions on Test Set")
print("="*50)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the trained model from checkpoint
model = LipReadingModel().to(device)
checkpoint = torch.load('checkpoint_epoch_20.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Model loaded successfully (epoch {checkpoint['epoch']+1})")

# Load test dataset
print("\n📂 Loading test dataset...")
test_dataset = LipReadingDataset('test', csv_file='all_videos.csv', max_frames=100)
test_loader = DataLoader(
    test_dataset, 
    batch_size=4, 
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)
print(f"   Test samples: {len(test_dataset)}")

# CTC decode function
def decode_predictions(output):
    """Convert model output to text"""
    predictions = []
    
    for batch_idx in range(output.shape[0]):
        # Get most likely character at each timestep
        pred_ids = torch.argmax(output[batch_idx], dim=-1).cpu().numpy()
        
        # Decode with CTC logic (remove blanks and repeats)
        decoded = []
        prev = -1
        for idx in pred_ids:
            if idx != 0 and idx != prev:  # 0 is blank, skip repeats
                if idx in num_to_char:
                    decoded.append(num_to_char[idx])
            prev = idx
        
        predictions.append(''.join(decoded))
    
    return predictions

# Generate predictions
print("\n🤖 Generating predictions...")
all_predictions = []
all_video_ids = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        videos = batch['video'].to(device)
        video_ids = batch['video_ids']
        
        # Forward pass
        output = model(videos)
        
        # Decode
        predictions = decode_predictions(output)
        
        all_predictions.extend(predictions)
        all_video_ids.extend(video_ids)

# Create submission dataframe
print("\n📝 Creating submission file...")

# Load original submission format
submission_df = pd.read_csv("data/sample_submission.csv")

# Create a mapping from video path to prediction
pred_dict = {}
for vid_id, pred in zip(all_video_ids, all_predictions):
    # Convert video_id back to path format
    path = f"test/{vid_id}.mp4"
    pred_dict[path] = pred

# Fill predictions
submission_df['transcription'] = submission_df['path'].map(pred_dict).fillna('')

# Show sample predictions
print("\n📊 Sample predictions:")
for i in range(min(10, len(submission_df))):
    if submission_df['transcription'].iloc[i]:
        print(f"   {submission_df['path'].iloc[i]}: {submission_df['transcription'].iloc[i]}")

# Save submission
submission_df.to_csv('submission.csv', index=False)
print("\n💾 Saved predictions to submission.csv")

# Statistics
non_empty = submission_df['transcription'].str.len() > 0
print(f"\n📊 Statistics:")
print(f"   Total test videos: {len(submission_df)}")
print(f"   Videos with predictions: {non_empty.sum()}")
print(f"   Empty predictions: {(~non_empty).sum()}")

if non_empty.sum() > 0:
    avg_len = submission_df.loc[non_empty, 'transcription'].str.len().mean()
    print(f"   Average prediction length: {avg_len:.1f} characters")

print("\n" + "="*50)
print("✅ Ready for submission!")
print("="*50)