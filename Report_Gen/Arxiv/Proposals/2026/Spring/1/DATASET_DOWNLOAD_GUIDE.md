# Dataset Download Guide
## Financial Market Sentiment Analysis Capstone Project

### 🎯 All Datasets Are Immediately Available - No Waiting Period!

---

## 📊 Dataset 1: Stock Price Data (Yahoo Finance)

**Access:** Free, No Registration Required  
**Size:** Depends on stocks and date range selected  
**Format:** CSV, Pandas DataFrame

### Installation:
```bash
pip install yfinance
```

### Download Code:
```python
import yfinance as yf
import pandas as pd

# Download single stock
aapl = yf.download('AAPL', start='2015-01-01', end='2024-12-31')
aapl.to_csv('aapl_historical_prices.csv')

# Download multiple stocks
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'WMT']
data = yf.download(tickers, start='2015-01-01', end='2024-12-31', group_by='ticker')
data.to_csv('multiple_stocks.csv')

# Download S&P 500 ETF
spy = yf.download('SPY', start='2015-01-01', end='2024-12-31')
spy.to_csv('sp500_etf.csv')
```

### Features Available:
- Open, High, Low, Close prices
- Adjusted Close (accounts for splits/dividends)
- Volume
- Date index

**Download Link:** Run the code above (no website to visit)  
**Estimated Time:** 1-2 minutes for 10 years of data

---

## 📰 Dataset 2: Financial News Dataset (Kaggle)

**Access:** Free Kaggle Account Required  
**Size:** ~6 million articles, ~3 GB  
**Format:** CSV

### Option A: Manual Download
1. Create free account at https://www.kaggle.com
2. Visit: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
3. Click "Download" button
4. Extract ZIP file

### Option B: Kaggle API (Recommended)
```bash
# Install Kaggle API
pip install kaggle

# Setup API credentials
# 1. Go to https://www.kaggle.com/settings
# 2. Scroll to "API" section, click "Create New Token"
# 3. Save kaggle.json to ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\<YourName>\.kaggle\kaggle.json (Windows)

# Download dataset
kaggle datasets download -d miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests

# Unzip
unzip massive-stock-news-analysis-db-for-nlpbacktests.zip
```

### Dataset Contents:
- Headlines
- Article text
- Stock ticker symbols
- Publication timestamps
- Source (news outlet)

**Download Link:** https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests  
**Estimated Time:** 5-10 minutes depending on internet speed

---

## 🐦 Dataset 3: Twitter Financial News Sentiment

**Access:** Free Kaggle Account Required  
**Size:** 11,932 tweets, ~5 MB  
**Format:** CSV

### Download Instructions:
```bash
# Manual: Visit link below and click Download
# API method:
kaggle datasets download -d sulphatet/twitter-financial-news
unzip twitter-financial-news.zip
```

### Dataset Contents:
- Tweet text
- Sentiment labels (positive, negative, neutral)
- Timestamps
- Stock mentions

**Download Link:** https://www.kaggle.com/datasets/sulphatet/twitter-financial-news  
**Estimated Time:** < 1 minute

---

## 📈 Dataset 4: StockNet Dataset (GitHub)

**Access:** Public GitHub Repository  
**Size:** ~1 GB  
**Format:** JSON, CSV

### Download Instructions:
```bash
# Clone repository
git clone https://github.com/yumoxu/stocknet-dataset.git
cd stocknet-dataset

# Dataset includes:
# - price/raw/*.csv (historical prices)
# - tweet/raw/*.json (tweets mapped to stocks)
```

### Dataset Contents:
- 88 stocks from S&P 500
- Historical prices (2014-2016)
- Twitter data aligned by date
- Pre-processed features

**Download Link:** https://github.com/yumoxu/stocknet-dataset  
**Estimated Time:** 2-3 minutes

---

## 💬 Dataset 5: Reddit WallStreetBets Posts (Optional)

**Access:** Free Kaggle Account  
**Size:** ~500 MB  
**Format:** CSV

### Download Instructions:
```bash
kaggle datasets download -d gpreda/reddit-wallstreetsbets-posts
unzip reddit-wallstreetsbets-posts.zip
```

### Dataset Contents:
- Post titles and content
- Timestamps
- Upvotes, comments
- Stock ticker mentions

**Download Link:** https://www.kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts  
**Estimated Time:** 2-3 minutes

---

## 📚 Dataset 6: Financial PhraseBank (Optional - For Fine-tuning)

**Access:** Public Research Dataset  
**Size:** 4,840 sentences, ~1 MB  
**Format:** TXT, CSV

### Download Method 1: Direct Download
Visit: https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10

### Download Method 2: Hugging Face (Easiest)
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")

# Convert to pandas
import pandas as pd
df = pd.DataFrame(dataset['train'])
df.to_csv('financial_phrasebank.csv', index=False)
```

### Dataset Contents:
- Financial sentences
- Sentiment labels (positive, negative, neutral)
- Annotator agreement levels

**Download Link:** https://huggingface.co/datasets/financial_phrasebank  
**Estimated Time:** < 1 minute

---

## 🚀 Quick Start: Complete Download Script

Save this as `download_all_datasets.py`:

```python
import yfinance as yf
import os
from datasets import load_dataset

# Create data directory
os.makedirs('data', exist_ok=True)

print("Step 1: Downloading stock prices...")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'WMT', 'V', 'JNJ']
stock_data = yf.download(tickers, start='2015-01-01', end='2024-12-31', group_by='ticker')
stock_data.to_csv('data/stock_prices.csv')
print("✓ Stock prices saved to data/stock_prices.csv")

print("\nStep 2: Downloading Financial PhraseBank...")
fp_dataset = load_dataset("financial_phrasebank", "sentences_allagree")
import pandas as pd
pd.DataFrame(fp_dataset['train']).to_csv('data/financial_phrasebank.csv', index=False)
print("✓ Financial PhraseBank saved to data/financial_phrasebank.csv")

print("\n✓ Automatic downloads complete!")
print("\nManual downloads needed:")
print("1. Kaggle Financial News: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests")
print("2. Twitter Financial: https://www.kaggle.com/datasets/sulphatet/twitter-financial-news")
print("3. StockNet: git clone https://github.com/yumoxu/stocknet-dataset.git")
print("\nRun: kaggle datasets download -d <dataset-id> (after setting up Kaggle API)")
```

---

## 📋 Dataset Summary Table

| Dataset | Size | Format | Access | Download Time |
|---------|------|--------|--------|---------------|
| Yahoo Finance | Variable | CSV | Free API | 1-2 min |
| Financial News (Kaggle) | ~3 GB | CSV | Free Account | 5-10 min |
| Twitter Financial | 5 MB | CSV | Free Account | <1 min |
| StockNet | ~1 GB | JSON/CSV | Public GitHub | 2-3 min |
| Reddit WSB | 500 MB | CSV | Free Account | 2-3 min |
| Financial PhraseBank | 1 MB | TXT/CSV | Public | <1 min |

**Total Download Time:** ~15-20 minutes (excluding large Kaggle dataset)

---

## 🔧 Troubleshooting

### Kaggle API Not Working?
```bash
# Check if credentials are correctly placed
ls ~/.kaggle/kaggle.json  # Linux/Mac
dir %USERPROFILE%\.kaggle\kaggle.json  # Windows

# Set permissions (Linux/Mac)
chmod 600 ~/.kaggle/kaggle.json
```

### yfinance Errors?
```python
# Use period instead of dates
yf.download('AAPL', period='max')

# Download one ticker at a time if batch fails
for ticker in ['AAPL', 'MSFT']:
    yf.download(ticker, start='2015-01-01').to_csv(f'data/{ticker}.csv')
```

### GitHub Clone Issues?
```bash
# Use HTTPS instead of SSH
git clone https://github.com/yumoxu/stocknet-dataset.git

# Or download as ZIP
wget https://github.com/yumoxu/stocknet-dataset/archive/refs/heads/master.zip
```

---

## ✅ Verification Checklist

After downloading, verify you have:
- [ ] Stock price CSV files (10+ stocks, 2015-2024)
- [ ] Financial news CSV (6M+ articles)
- [ ] Twitter sentiment CSV (11K+ tweets)
- [ ] StockNet data (optional but recommended)
- [ ] Financial PhraseBank (for sentiment model fine-tuning)

Total disk space needed: ~5 GB

---

## 📞 Support

If you encounter issues:
1. Check Kaggle account is verified (email confirmation)
2. Ensure internet connection is stable for large downloads
3. Try downloading during off-peak hours for faster speeds
4. For Kaggle issues, visit: https://www.kaggle.com/discussions

**All datasets confirmed working as of December 2024**
