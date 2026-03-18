import os
import glob

print("="*50)
print("Checking for model files")
print("="*50)

# Look for all .pt files
pt_files = glob.glob("*.pt")

if pt_files:
    print(f"Found {len(pt_files)} model files:")
    for f in pt_files:
        size = os.path.getsize(f) / (1024*1024)  # Size in MB
        print(f"   📁 {f} ({size:.2f} MB)")
        
        # Try to load it to verify
        try:
            import torch
            checkpoint = torch.load(f, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print(f"      ✅ Checkpoint file (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                print(f"      ✅ Model weights file")
        except:
            print(f"      ❌ Could not load")
else:
    print("❌ No .pt files found!")
    
    # Also check for any checkpoint files
    checkpoints = glob.glob("checkpoint_epoch_*.pt")
    if checkpoints:
        print(f"\nFound checkpoint files: {checkpoints}")
    else:
        print("\nNo checkpoint files found either!")
        
    print("\n📁 Current directory contents:")
    for f in os.listdir('.'):
        if os.path.isfile(f):
            print(f"   {f}")