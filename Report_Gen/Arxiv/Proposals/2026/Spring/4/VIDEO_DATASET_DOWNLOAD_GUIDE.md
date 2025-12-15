# Video Dataset Download Guide
## Deep Learning for Video Understanding Capstone Project

### 🎯 All Datasets Are Publicly Available - No Special Approval!

---

## 📊 DATASET 1: UCF101 (RECOMMENDED - Start Here!)

**What it is:** Most popular action recognition benchmark  
**Size:** 6.5 GB  
**Samples:** 13,320 videos  
**Classes:** 101 action categories  
**Duration:** 2-16 seconds per video  
**Resolution:** 320×240 pixels  
**FPS:** Variable (typically 25-30)  
**Format:** AVI files  

### Download Options:

#### Option 1: Kaggle (Easiest - Recommended)
```bash
# Install Kaggle API
pip install kaggle

# Setup credentials (one-time)
# 1. Go to kaggle.com/settings
# 2. Create API token → downloads kaggle.json  
# 3. Move to ~/.kaggle/kaggle.json

# Download
kaggle datasets download -d matthewjansen/ucf101-action-recognition
unzip ucf101-action-recognition.zip -d data/UCF101/
```

#### Option 2: Official Site  
Visit: https://www.crcv.ucf.edu/data/UCF101.php  
Download: UCF101.rar (direct download)  
Extract: `unrar x UCF101.rar data/UCF101/`

### Dataset Structure:
```
UCF101/
├── ApplyEyeMakeup/
│   ├── v_ApplyEyeMakeup_g01_c01.avi
│   ├── v_ApplyEyeMakeup_g01_c02.avi
│   └── ...
├── Basketball/
├── Biking/
├── ...
└── YoYo/
```

### Loading UCF101:
```python
import cv2
from pathlib import Path

video_dir = Path('data/UCF101')
video_files = list(video_dir.rglob('*.avi'))

print(f"Total videos: {len(video_files)}")
print(f"Classes: {len(list(video_dir.iterdir()))}")

# Load a video
cap = cv2.VideoCapture(str(video_files[0]))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps}, Frames: {frame_count}")
```

**Download Time:** 15-20 minutes  
**Paper:** https://arxiv.org/abs/1212.0402

---

## 📊 DATASET 2: HMDB51

**What it is:** Human Motion Database with movies + YouTube  
**Size:** 2 GB  
**Samples:** 7,000 videos  
**Classes:** 51 action categories  
**Duration:** Variable (1-15 seconds)  
**Format:** AVI files  

### Download:
Official site: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/  
Registration: Required (instant, free)  

```bash
# After downloading from official site
unrar x hmdb51_org.rar data/HMDB51/
```

### Classes Include:
- brush_hair, cartwheel, catch, chew, clap, climb
- dive, draw_sword, dribble, drink, eat, fall_floor  
- And 39 more...

**Download Time:** 5-10 minutes  
**Paper:** https://ieeexplore.ieee.org/document/6126543

---

## 📊 DATASET 3: Kinetics-400

**What it is:** Large-scale dataset from YouTube  
**Size:** 450 GB (full) | 5 GB (mini version)  
**Samples:** 400,000 videos (full) | 5,000 (mini)  
**Classes:** 400 human actions  
**Duration:** ~10 seconds per clip  
**Resolution:** Variable (360p+)  
**Format:** MP4 files  

### Download Options:

#### Option 1: Kinetics-400 Mini (RECOMMENDED for prototyping)
```bash
kaggle datasets download -d shivamb/kinetics-400-mini
unzip kinetics-400-mini.zip -d data/Kinetics-Mini/
```

#### Option 2: Full Dataset (Advanced users only)
Official repo: https://github.com/cvdfoundation/kinetics-dataset

```bash
git clone https://github.com/cvdfoundation/kinetics-dataset
cd kinetics-dataset
# Follow instructions in README for downloading with youtube-dl
```

### Why Use Mini Version:
- 100x faster download
- Same class distribution  
- Perfect for prototyping
- Upgrade to full later if needed

**Download Time:** 10 minutes (mini) | Several hours (full)  
**Paper:** https://arxiv.org/abs/1705.06950

---

## 📊 DATASET 4: Something-Something V2

**What it is:** Fine-grained human-object interactions  
**Size:** 20 GB  
**Samples:** 220,847 videos  
**Classes:** 174 action classes  
**Duration:** 2-6 seconds  
**Format:** WebM files  

### What Makes It Special:
Unlike UCF101/Kinetics, this requires understanding TEMPORAL RELATIONSHIPS.  
Examples: "putting X into Y", "taking X from Y", "pushing X from left to right"  
A single frame cannot solve these - need full temporal context!

### Download:
Official site: https://developer.qualcomm.com/software/ai-datasets/something-something  
Registration: Required (free, instant approval)

After registration, download links provided for:
- Training videos (168K)
- Validation videos (24K)
- Test videos (27K)
- JSON annotations

```bash
# After downloading
mkdir data/Something-Something-V2
mv 20bn-something-something-v2-??.tar.gz data/Something-Something-V2/
cd data/Something-Something-V2
for f in *.tar.gz; do tar -xzf "$f"; done
```

**Download Time:** 30-45 minutes  
**Paper:** https://arxiv.org/abs/1706.04261

---

## 📊 DATASET 5: Moments in Time (Optional)

**What it is:** 3-second video clips for scene understanding  
**Size:** 100 GB (full) | 5 GB (mini)  
**Samples:** 1 million (full) | 50K (mini)  
**Classes:** 339 actions/events  
**Duration:** 3 seconds exactly  
**Format:** MP4 files  

### Download:
Official: http://moments.csail.mit.edu/

```bash
# Mini version (recommended)
wget http://moments.csail.mit.edu/data/moments_mini.tar
tar -xf moments_mini.tar -C data/Moments/
```

**Download Time:** 10-15 minutes (mini)

---

## 🚀 Quick Start: Complete Download Script

```bash
#!/bin/bash
# download_all_video_datasets.sh

echo "Creating data directory..."
mkdir -p data/{UCF101,HMDB51,Kinetics-Mini,Something-V2}

echo "Downloading UCF101..."
kaggle datasets download -d matthewjansen/ucf101-action-recognition
unzip ucf101-action-recognition.zip -d data/UCF101/
rm ucf101-action-recognition.zip

echo "Downloading Kinetics-400 Mini..."
kaggle datasets download -d shivamb/kinetics-400-mini
unzip kinetics-400-mini.zip -d data/Kinetics-Mini/
rm kinetics-400-mini.zip

echo "✓ Automatic downloads complete!"
echo "Manual downloads:"
echo "1. HMDB51: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/"
echo "2. Something-V2: https://developer.qualcomm.com/software/ai-datasets/something-something"
```

---

## 📋 Dataset Comparison Table

| Dataset | Size | Videos | Classes | Avg Duration | Best For |
|---------|------|--------|---------|--------------|----------|
| **UCF101** | 6.5 GB | 13K | 101 | 7 sec | Beginners, prototyping |
| **HMDB51** | 2 GB | 7K | 51 | 3 sec | Quick experiments |
| **Kinetics-Mini** | 5 GB | 5K | 400 | 10 sec | Large-scale preview |
| **Kinetics-Full** | 450 GB | 400K | 400 | 10 sec | SOTA training |
| **Something-V2** | 20 GB | 220K | 174 | 4 sec | Temporal reasoning |
| **Moments** | 100 GB | 1M | 339 | 3 sec | Scene understanding |

---

## 🔧 Extracting Frames from Videos

### Method 1: OpenCV (Python)
```python
import cv2
import numpy as np
from pathlib import Path

def extract_frames(video_path, output_dir, num_frames=16):
    """Extract uniformly sampled frames"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Uniformly sample frames
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            output_path = output_dir / f'frame_{idx:04d}.jpg'
            cv2.imwrite(str(output_path), frame)
    
    cap.release()
    return num_frames

# Extract frames for all videos
video_dir = Path('data/UCF101')
for video_path in video_dir.rglob('*.avi'):
    class_name = video_path.parent.name
    video_name = video_path.stem
    output_dir = Path(f'data/frames/{class_name}/{video_name}')
    extract_frames(video_path, output_dir, num_frames=16)
```

### Method 2: FFmpeg (Command Line - Faster!)
```bash
# Install FFmpeg
# Ubuntu: sudo apt install ffmpeg
# Mac: brew install ffmpeg

# Extract 1 frame per second
ffmpeg -i input.mp4 -vf fps=1 output_%04d.jpg

# Extract exactly 16 frames uniformly
ffmpeg -i input.mp4 -vf "select='not(mod(n\,$(($TOTAL_FRAMES/16)))'" -vsync 0 output_%04d.jpg

# Resize while extracting
ffmpeg -i input.mp4 -vf "fps=1,scale=224:224" output_%04d.jpg
```

### Method 3: Decord (Fastest for PyTorch)
```python
from decord import VideoReader, cpu
import numpy as np

# Install: pip install decord

def load_video(video_path, num_frames=16):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    
    # Sample frames uniformly
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
    
    return frames

frames = load_video('video.mp4', num_frames=16)
print(f"Shape: {frames.shape}")  # (16, 240, 320, 3)
```

---

## 📊 Video EDA Example Script

```python
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def analyze_video_dataset(video_dir):
    """Comprehensive video dataset analysis"""
    video_files = list(Path(video_dir).rglob('*.avi'))
    
    stats = []
    for video_path in tqdm(video_files):
        cap = cv2.VideoCapture(str(video_path))
        
        info = {
            'filename': video_path.name,
            'class': video_path.parent.name,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        stats.append(info)
    
    df = pd.DataFrame(stats)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Duration distribution
    axes[0,0].hist(df['duration'], bins=50, edgecolor='black')
    axes[0,0].set_title('Video Duration Distribution')
    axes[0,0].set_xlabel('Duration (seconds)')
    
    # Frame count
    axes[0,1].hist(df['frame_count'], bins=50, color='orange', edgecolor='black')
    axes[0,1].set_title('Frame Count Distribution')
    
    # FPS
    axes[0,2].hist(df['fps'], bins=30, color='green', edgecolor='black')
    axes[0,2].set_title('FPS Distribution')
    
    # Resolution scatter
    axes[1,0].scatter(df['width'], df['height'], alpha=0.5)
    axes[1,0].set_title('Resolution Distribution')
    axes[1,0].set_xlabel('Width')
    axes[1,0].set_ylabel('Height')
    
    # Class distribution
    class_counts = df['class'].value_counts()
    axes[1,1].barh(range(len(class_counts[:20])), class_counts[:20].values)
    axes[1,1].set_title('Top 20 Classes')
    axes[1,1].set_ylabel('Class')
    
    # Summary statistics
    summary_text = f"""
    Total Videos: {len(df)}
    Total Classes: {df['class'].nunique()}
    Avg Duration: {df['duration'].mean():.2f}s
    Avg FPS: {df['fps'].mean():.2f}
    Avg Frames: {df['frame_count'].mean():.0f}
    """
    axes[1,2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig('video_dataset_eda.png', dpi=150)
    
    # Save statistics
    df.to_csv('video_statistics.csv', index=False)
    
    return df

# Run analysis
df = analyze_video_dataset('data/UCF101')
print(df.describe())
```

---

## ✅ Verification Checklist

After downloading, verify:

- [ ] UCF101: 13,320 videos, 101 folders
- [ ] HMDB51: 7,000 videos, 51 folders (if downloaded)
- [ ] Kinetics-Mini: 5,000 videos (if downloaded)
- [ ] All videos playable with OpenCV/VLC
- [ ] Frame extraction works correctly
- [ ] Train/test splits available (UCF101 has 3 official splits)

---

## 📞 Troubleshooting

**Problem:** "Cannot open video file"  
**Solution:** Install codec support: `pip install opencv-contrib-python`

**Problem:** Slow video loading  
**Solution:** Use `decord` library instead of OpenCV

**Problem:** Out of disk space  
**Solution:** Start with UCF101 only (6.5 GB), extract frames and delete videos

**Problem:** Kaggle API not working  
**Solution:** Check `~/.kaggle/kaggle.json` exists with correct permissions (600)

---

## 🎯 Recommended Starting Strategy

1. **Week 1**: Download UCF101 only (6.5 GB)
2. **Week 2**: Extract frames, perform EDA
3. **Week 3-4**: Train baselines on UCF101
4. **Week 5**: Download Kinetics-Mini (5 GB) for validation
5. **Week 6+**: Optionally download full Kinetics if needed

This progressive approach prevents overwhelming storage/compute resources!

**All datasets confirmed working as of December 2024** ✓
