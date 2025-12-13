# Dataset Download Guide
## Multimodal Misinformation Detection Capstone Project

### All Datasets Are Publicly Available - No IRB or Special Approval Needed!

---

##  PRIMARY DATASETS

### Dataset 1: Fakeddit (RECOMMENDED - Start Here!)

**What it is:** The largest multimodal fake news dataset from Reddit  
**Access:** Free, Kaggle account required (instant approval)  
**Size:** ~1 million posts, ~10 GB with images  
**Labels:** 2-way (fake/real) + 6-way fine-grained classification  
**Format:** CSV (metadata) + Images

#### Installation Method 1: Kaggle Website (Easiest)
```
1. Create free account at https://www.kaggle.com
2. Visit: https://www.kaggle.com/datasets/mdepak/fakeddit
3. Click "Download" button (3 GB compressed)
4. Extract ZIP file to your data/ folder
```

#### Installation Method 2: Kaggle API (Recommended for Automation)
```bash
# Install Kaggle API
pip install kaggle

# Setup API credentials (ONE-TIME SETUP):
# 1. Go to https://www.kaggle.com/settings
# 2. Scroll to "API" section
# 3. Click "Create New Token" → downloads kaggle.json
# 4. Move to: ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\<You>\.kaggle\kaggle.json (Windows)
# 5. Set permissions (Linux/Mac only): chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d mdepak/fakeddit

# Unzip
unzip fakeddit.zip -d data/fakeddit/
```

#### Dataset Structure:
```
fakeddit/
├── train.csv               # Training set with labels
├── validate.csv            # Validation set
├── test.csv                # Test set (public)
├── test_for_submission.csv # Final test (no labels)
└── images/                 # Image files (download separately or on-demand)
    ├── 0a1b2c.jpg
    ├── 3d4e5f.jpg
    └── ...
```

#### Loading Fakeddit:
```python
import pandas as pd
from PIL import Image
import os

# Load metadata
train_df = pd.read_csv('data/fakeddit/train.csv')

# Columns: id, title, image_url, 2_way_label, 6_way_label, created_utc, ...

# Sample record
print(train_df.head(1))

# Load image (if downloaded locally)
img_id = train_df.iloc[0]['id']
img = Image.open(f'data/fakeddit/images/{img_id}.jpg')

# Or download on-demand from URL
import requests
from io import BytesIO

img_url = train_df.iloc[0]['image_url']
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))
```

**Download Time:** 10-15 minutes (metadata < 1 min, images 10-15 min)  
**Dataset Paper:** https://arxiv.org/abs/1911.03854

---

### Dataset 2: MEME Dataset (Hateful Memes + Multimodal)

**What it is:** Memes with text overlays and hate speech labels  
**Access:** Free, no account needed (GitHub) or Hugging Face  
**Size:** 10,000+ memes, ~500 MB  
**Labels:** Hateful/not-hateful, can adapt for misinformation  
**Format:** JSON + Images

#### Installation Method 1: Hugging Face (Easiest)
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("limjiayi/hateful_memes_expanded")

# Access samples
train = dataset['train']
print(train[0])  # {'id': 0, 'img': <PIL.Image>, 'text': '...', 'label': 0}

# Convert to pandas
import pandas as pd
df = pd.DataFrame(train)
df.to_csv('data/memes/memes.csv', index=False)
```

#### Installation Method 2: GitHub
```bash
# Clone repository
git clone https://github.com/TIBHannover/MM-Claims.git
cd MM-Claims

# Images are in images/ folder
# Annotations in annotations.json
```

#### Loading MEME Dataset:
```python
import json
from PIL import Image

# Load annotations
with open('data/memes/annotations.json', 'r') as f:
    data = json.load(f)

# Each entry has: id, img_path, text, label
for item in data[:5]:
    print(f"Text: {item['text']}")
    print(f"Label: {item['label']}")
    img = Image.open(item['img_path'])
    # Process...
```

**Download Time:** 2-3 minutes  
**Dataset Paper:** https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/

---

### Dataset 3: FakeNewsNet (Twitter + PolitiFact/GossipCop)

**What it is:** News articles with fact-check labels + social media engagement  
**Access:** Free, GitHub  
**Size:** 20,000+ news articles with images, ~2 GB  
**Labels:** Real/fake from professional fact-checkers  
**Format:** JSON + Images

#### Installation:
```bash
# Clone repository
git clone https://github.com/KaiDMML/FakeNewsNet.git
cd FakeNewsNet

# Dataset is organized by source:
# - politifact/  (political news)
# - gossipcop/   (entertainment news)

# Download images (they provide scripts)
cd code
python download_images.py
```

#### Dataset Structure:
```
FakeNewsNet/
├── politifact/
│   ├── fake/
│   │   ├── article_123.json  # News article metadata
│   │   ├── article_123_tweets.json  # Related tweets
│   │   └── article_123.jpg   # Article image
│   └── real/
│       └── ...
└── gossipcop/
    └── ...
```

#### Loading FakeNewsNet:
```python
import json
import glob

# Load all fake news articles
fake_articles = []
for filepath in glob.glob('data/fakenewsnet/politifact/fake/*.json'):
    if 'tweets' not in filepath:  # Skip tweet files
        with open(filepath, 'r') as f:
            article = json.load(f)
            fake_articles.append(article)

print(f"Loaded {len(fake_articles)} fake articles")

# Access fields: title, text, images, url, author, etc.
```

**Download Time:** 5-7 minutes  
**Dataset Paper:** https://arxiv.org/abs/1809.01286

---

## 📰 SUPPLEMENTARY DATASETS

### Dataset 4: LIAR (Text-Only Baseline)

**What it is:** Political statements with 6-level truthfulness ratings  
**Access:** Free, direct download or Hugging Face  
**Size:** 12,836 statements, ~10 MB  
**Labels:** pants-fire, false, barely-true, half-true, mostly-true, true  
**Format:** TSV

#### Installation Method 1: Direct Download
```bash
wget https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
unzip liar_dataset.zip -d data/liar/
```

#### Installation Method 2: Hugging Face
```python
from datasets import load_dataset

dataset = load_dataset("liar")
train = dataset['train']

# Convert to pandas
import pandas as pd
df = pd.DataFrame(train)
```

#### Loading LIAR:
```python
import pandas as pd

# Load TSV file
columns = ['id', 'label', 'statement', 'subject', 'speaker', 
           'job', 'state', 'party', 'barely_true_counts', 
           'false_counts', 'half_true_counts', 'mostly_true_counts', 
           'pants_on_fire_counts', 'context']

train_df = pd.read_csv('data/liar/train.tsv', sep='\t', names=columns)
```

**Download Time:** < 1 minute  
**Dataset Paper:** https://arxiv.org/abs/1705.00648

---

### Dataset 5: MultiOFF (Offensive Content Detection)

**What it is:** Twitter posts with images and offensive content labels  
**Access:** Free, GitHub  
**Size:** 10,000+ tweets, ~300 MB  
**Labels:** Offensive/not-offensive  
**Format:** CSV + Image URLs

#### Installation:
```bash
git clone https://github.com/bharathichezhiyan/Multimodal-Offensive-Dataset.git
cd Multimodal-Offensive-Dataset
```

#### Loading MultiOFF:
```python
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

# Load CSV
df = pd.read_csv('data/multioff/dataset.csv')

# Download image from URL
def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        return Image.open(BytesIO(response.content))
    except:
        return None

# Process samples
for idx, row in df.iterrows():
    text = row['tweet']
    img = download_image(row['image_url'])
    label = row['label']
    # Process...
```

**Download Time:** 2-3 minutes  
**Dataset Paper:** https://aclanthology.org/2020.emnlp-main.470/

---

### Dataset 6: Twitter Fake News Dataset (Text + Metadata)

**What it is:** News articles with fake/real labels from fact-checkers  
**Access:** Free, Kaggle  
**Size:** 44,000+ articles, ~150 MB  
**Labels:** Fake/real  
**Format:** CSV

#### Installation:
```bash
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset
unzip fake-and-real-news-dataset.zip -d data/twitter_news/
```

#### Loading:
```python
import pandas as pd

fake_df = pd.read_csv('data/twitter_news/Fake.csv')
real_df = pd.read_csv('data/twitter_news/True.csv')

# Add labels
fake_df['label'] = 0
real_df['label'] = 1

# Combine
df = pd.concat([fake_df, real_df], ignore_index=True)
```

**Download Time:** 2 minutes

---

## 🚀 Quick Start: Complete Download Script

Save as `download_all_datasets.py`:

```python
import os
import subprocess
from datasets import load_dataset
import pandas as pd

# Create data directory
os.makedirs('data', exist_ok=True)

print("="*80)
print("DOWNLOADING ALL DATASETS FOR MULTIMODAL MISINFORMATION DETECTION")
print("="*80)

# 1. Download LIAR (smallest, fastest)
print("\n[1/6] Downloading LIAR dataset...")
try:
    liar = load_dataset("liar")
    pd.DataFrame(liar['train']).to_csv('data/liar_train.csv', index=False)
    pd.DataFrame(liar['validation']).to_csv('data/liar_val.csv', index=False)
    pd.DataFrame(liar['test']).to_csv('data/liar_test.csv', index=False)
    print("✓ LIAR dataset downloaded successfully!")
except Exception as e:
    print(f"✗ Error: {e}")

# 2. Download MEME dataset
print("\n[2/6] Downloading MEME dataset...")
try:
    memes = load_dataset("limjiayi/hateful_memes_expanded")
    pd.DataFrame(memes['train']).to_csv('data/memes_train.csv', index=False)
    print("✓ MEME dataset downloaded successfully!")
except Exception as e:
    print(f"✗ Error: {e}")

# 3. Clone FakeNewsNet
print("\n[3/6] Cloning FakeNewsNet repository...")
try:
    subprocess.run(['git', 'clone', 'https://github.com/KaiDMML/FakeNewsNet.git', 
                   'data/FakeNewsNet'], check=True)
    print("✓ FakeNewsNet cloned successfully!")
except Exception as e:
    print(f"✗ Error: {e}")

# 4. Clone MultiOFF
print("\n[4/6] Cloning MultiOFF repository...")
try:
    subprocess.run(['git', 'clone', 
                   'https://github.com/bharathichezhiyan/Multimodal-Offensive-Dataset.git',
                   'data/MultiOFF'], check=True)
    print("✓ MultiOFF cloned successfully!")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*80)
print("AUTOMATIC DOWNLOADS COMPLETE!")
print("="*80)

print("\n📋 MANUAL DOWNLOADS REQUIRED:")
print("\n[5/6] Fakeddit (PRIMARY DATASET - 3 GB):")
print("   Option 1 - Kaggle Website:")
print("   1. Go to: https://www.kaggle.com/datasets/mdepak/fakeddit")
print("   2. Click 'Download'")
print("   3. Extract to data/fakeddit/")
print("\n   Option 2 - Kaggle API:")
print("   kaggle datasets download -d mdepak/fakeddit")
print("   unzip fakeddit.zip -d data/fakeddit/")

print("\n[6/6] Twitter News Dataset:")
print("   kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset")
print("   unzip fake-and-real-news-dataset.zip -d data/twitter_news/")

print("\n" + "="*80)
print("NEXT STEPS:")
print("1. Set up Kaggle API credentials (see guide above)")
print("2. Download Fakeddit using Kaggle")
print("3. Run data preprocessing script")
print("="*80)
```

Run with:
```bash
python download_all_datasets.py
```

---

##  Dataset Summary Table

| Dataset | Size | Samples | Modalities | Labels | Access | Download Time |
|---------|------|---------|------------|--------|--------|---------------|
| **Fakeddit** | 10 GB | 1M+ | Text + Image | 2-way/6-way | Kaggle | 10-15 min |
| **MEME** | 500 MB | 10K+ | Image + Text | Binary | Hugging Face | 2-3 min |
| **FakeNewsNet** | 2 GB | 20K+ | Text + Image | Binary | GitHub | 5-7 min |
| **LIAR** | 10 MB | 12K+ | Text | 6-way | Direct/HF | <1 min |
| **MultiOFF** | 300 MB | 10K+ | Text + Image | Binary | GitHub | 2-3 min |
| **Twitter News** | 150 MB | 44K+ | Text | Binary | Kaggle | 2 min |

**Total Storage:** ~13 GB  
**Total Samples:** 1.1M+ multimodal posts  
**Total Time:** ~25-35 minutes (excluding Fakeddit image download)

---

##  Troubleshooting

### Kaggle API Issues

**Problem:** "Could not find kaggle.json"
```bash
# Ensure kaggle.json is in correct location
# Linux/Mac: ~/.kaggle/kaggle.json
# Windows: C:\Users\<YourName>\.kaggle\kaggle.json

# Check file exists
ls ~/.kaggle/kaggle.json  # Linux/Mac
dir %USERPROFILE%\.kaggle\kaggle.json  # Windows

# Set permissions (Linux/Mac)
chmod 600 ~/.kaggle/kaggle.json
```

**Problem:** "403 Forbidden"
- Ensure you've accepted dataset rules on Kaggle website
- Regenerate API token if needed

### GitHub Clone Issues

**Problem:** "fatal: destination path exists"
```bash
# Directory already exists, remove or rename
rm -rf data/FakeNewsNet
git clone https://github.com/KaiDMML/FakeNewsNet.git data/FakeNewsNet
```

**Problem:** Slow download
```bash
# Use shallow clone (faster)
git clone --depth 1 https://github.com/KaiDMML/FakeNewsNet.git
```

### Image Download Issues

**Problem:** Images not loading from URLs
```python
# Add timeout and error handling
import requests
from PIL import Image
from io import BytesIO
import time

def download_image_safe(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            return Image.open(BytesIO(response.content))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue
            else:
                print(f"Failed to download {url}: {e}")
                return None
```

### Storage Issues

**Problem:** Running out of disk space
- Download datasets incrementally
- Process and delete source files after preprocessing
- Use data streaming for large datasets
- Sample subset for initial development (10-20%)

---

## Verification Checklist

After downloading, verify you have:

- [ ] **Fakeddit:** train.csv, validate.csv, test.csv (+ images folder)
- [ ] **MEME:** annotations.json, images/ folder
- [ ] **FakeNewsNet:** politifact/ and gossipcop/ folders
- [ ] **LIAR:** train.tsv, val.tsv, test.tsv
- [ ] **MultiOFF:** dataset.csv with image URLs
- [ ] **Twitter News:** Fake.csv and True.csv

**Quick verification script:**
```python
import os

datasets = {
    'Fakeddit': 'data/fakeddit/train.csv',
    'MEME': 'data/memes/annotations.json',
    'FakeNewsNet': 'data/FakeNewsNet/README.md',
    'LIAR': 'data/liar_train.csv',
    'MultiOFF': 'data/MultiOFF/dataset.csv',
    'Twitter News': 'data/twitter_news/Fake.csv'
}

print("Dataset Verification:")
for name, path in datasets.items():
    exists = "✓" if os.path.exists(path) else "✗"
    print(f"{exists} {name}: {path}")
```

---

## Support Resources

**Dataset Issues:**
- Fakeddit: https://github.com/entitize/Fakeddit/issues
- FakeNewsNet: https://github.com/KaiDMML/FakeNewsNet/issues
- Hugging Face: https://discuss.huggingface.co/

**Kaggle Help:**
- API Documentation: https://www.kaggle.com/docs/api
- Community Forum: https://www.kaggle.com/discussions

**General:**
- Create GitHub issue in your project repository
- Contact instructor: ajafari@gwu.edu
- Course discussion forum

---

## 🎯 Next Steps

After downloading datasets:

1. **Run preprocessing script** (see `data/preprocess.py`)
2. **Explore data** using provided Jupyter notebooks
3. **Create train/val/test splits** (stratified)
4. **Generate statistics** for paper
5. **Start with baselines** (TF-IDF, ResNet)

**All datasets confirmed working as of December 2024** ✓
