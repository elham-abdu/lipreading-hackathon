import os

print("="*50)
print("Checking Test Video Processing Progress")
print("="*50)

test_path = "processed_data/test"

if not os.path.exists(test_path):
    print("❌ Test path not found!")
    exit()

# Count folders
folders = [f for f in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, f))]
print(f"📁 Total folders processed: {len(folders)}")

# Count total videos
total_videos = 0
for folder in folders[:10]:  # Check first 10 folders
    folder_path = os.path.join(test_path, folder)
    videos = [v for v in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, v))]
    total_videos += len(videos)

print(f"🎥 Estimated videos processed: ~{total_videos * (len(folders)//10)}")

# Check if extraction is still running
import subprocess
result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                      capture_output=True, text=True)
if 'python.exe' in result.stdout:
    print("🐍 Python is STILL RUNNING - extraction in progress")
else:
    print("✅ Python not running - extraction may be complete")

print("\n💡 To check exact count, look inside a folder:")
if folders:
    sample = folders[0]
    sample_path = os.path.join(test_path, sample)
    videos = os.listdir(sample_path)[:3]
    print(f"   In {sample}: {videos}")