import os
import cv2
import pandas as pd

print("="*50)
print("STEP 1: Checking Dataset")
print("="*50)

data_path = "data"

# ✅ Check if data folder exists
if not os.path.exists(data_path):
    print("❌ Data folder not found!")
    print("Make sure you extracted dataset into 'data/'")
    exit()

print("✅ Data folder found!")

# ✅ List contents of data folder
print(f"\n📁 Contents of '{data_path}':")
items = os.listdir(data_path)
for item in items:
    print(f"   - {item}")

# ===============================
# 📄 Check CSV (sample submission)
# ===============================
csv_files = [f for f in items if f.endswith('.csv')]

if csv_files:
    csv_path = os.path.join(data_path, csv_files[0])
    print(f"\n📄 Found CSV file: {csv_files[0]}")

    df = pd.read_csv(csv_path)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")

    print("\n📝 First 3 rows:")
    print(df.head(3))

else:
    print("\n❌ No CSV file found!")

# ===============================
# 🎥 Check video folders
# ===============================
video_folders = [
    f for f in items 
    if os.path.isdir(os.path.join(data_path, f)) and f != 'venv'
]

if video_folders:
    print(f"\n🎥 Found video folders: {video_folders}")

    # Pick first main folder (train or test)
    first_folder = os.path.join(data_path, video_folders[0])

    subfolders = os.listdir(first_folder)

    if subfolders:
        # Go one level deeper
        first_subfolder = os.path.join(first_folder, subfolders[0])

        videos = [
            f for f in os.listdir(first_subfolder)
            if f.endswith(('.mp4', '.avi', '.mov'))
        ]

        print(f"\n📹 Checking inside '{video_folders[0]}/{subfolders[0]}'")
        print(f"   Found {len(videos)} videos")

        if videos:
            sample_video = videos[0]
            print(f"   Sample video: {sample_video}")

            video_path = os.path.join(first_subfolder, sample_video)

            # Try opening video
            cap = cv2.VideoCapture(video_path)

            if cap.isOpened():
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                print(f"\n✅ Video properties:")
                print(f"   FPS: {fps}")
                print(f"   Frames: {frame_count}")
                print(f"   Resolution: {width}x{height}")
                print(f"   Duration: {frame_count / fps:.2f} seconds")

                cap.release()
            else:
                print("❌ Could not open video")

        else:
            print("❌ No videos found inside subfolder")

    else:
        print("❌ No subfolders found inside train/test")

else:
    print("\n❌ No video folders found!")

print("\n" + "="*50)
print("✅ Step 1 complete! Ready for Step 2")
print("="*50)