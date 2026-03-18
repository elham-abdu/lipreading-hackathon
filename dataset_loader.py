import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Character mappings
char_list = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_to_num = {c: i+1 for i, c in enumerate(char_list)}
num_to_char = {i+1: c for i, c in enumerate(char_list)}

class LipReadingDataset(Dataset):
    def __init__(self, data_split, csv_file='all_videos.csv', max_frames=100):
        self.data_split = data_split
        self.max_frames = max_frames
        
        # Load appropriate CSV
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
        else:
            self.df = pd.read_csv("data/sample_submission.csv")
        
        # Filter by split
        self.df = self.df[self.df['path'].str.startswith(data_split)]
        
        self.processed_path = os.path.join("processed_data", data_split)
        self.video_data = []
        self._prepare_data()
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Split: {data_split}")
        print(f"   Videos in CSV: {len(self.df)}")
        print(f"   Successfully loaded: {len(self.video_data)}")
    
    def _prepare_data(self):
        for idx, row in self.df.iterrows():
            video_path = row['path']
            transcript = row['transcription'] if pd.notna(row['transcription']) else ""
            
            parts = video_path.split('/')
            folder_id = parts[1]
            video_name = parts[2].replace('.mp4', '')
            
            frames_folder = os.path.join(self.processed_path, folder_id, video_name)
            
            if os.path.exists(frames_folder):
                frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
                if len(frame_files) > 0:
                    self.video_data.append({
                        'video_id': f"{folder_id}/{video_name}",
                        'frames_folder': frames_folder,
                        'frame_files': frame_files,
                        'transcript': transcript,
                        'num_frames': len(frame_files)
                    })
    
    def __len__(self):
        return len(self.video_data)
    
    def __getitem__(self, idx):
        data = self.video_data[idx]
        
        frames = []
        for frame_file in data['frame_files'][:self.max_frames]:
            img_path = os.path.join(data['frames_folder'], frame_file)
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
        
        # Encode transcript
        transcript = data['transcript'].lower()
        transcript_nums = [char_to_num[c] for c in transcript if c in char_to_num]
        transcript_tensor = torch.LongTensor(transcript_nums)
        
        return {
            'video': video_tensor,
            'transcript': transcript_tensor,
            'transcript_text': transcript,
            'video_id': data['video_id'],
            'num_frames': data['num_frames']
        }