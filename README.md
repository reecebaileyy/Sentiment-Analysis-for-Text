# 🌟 IMDb Sentiment Analysis 🌟
Dive into the world of movie reviews and analyze sentiments using Python! 🎥🍿

## 📌 Introduction
This repository contains a Python-based Sentiment Analysis tool which classifies IMDb movie reviews into positive (👍) and negative (👎) sentiments. Whether you're a movie buff, a data enthusiast, or someone who just loves playing with texts, this tool is for you!

## 📂 Structure
.
├── IMDB
│   ├── Train.csv
│   ├── Valid.csv
│   ├── Test.csv
└── text_analysis.py

## 🚀 Getting Started

### 🛠 Prerequisites
- Make sure you have Python3 installed:
```bash
python3 --version
```

### ⬇ Installation
1. Clone this repository:
```bash
git clone [(https://github.com/reecebaileyy/Sentiment-Analysis-for-Text.git)]
cd [your-cloned-repo-folder]
```

2. Install required libraries:
```bash
pip install pandas numpy matplotlib nltk textblob scikit-learn wordcloud
```

### 🏃 Running the Tool
```bash
python3 text_analysis.py
```

### 📊 Outputs
- Displayed filenames in the IMDB folder.
- Pie chart 🥧 showing label distribution.
- Word clouds ☁️ for positive and negative reviews.
- Performance metrics 📈 of the model on validation and test datasets.

## ✨ Future Updates
 - Integrate more advanced NLP models for better accuracy.
 - Add a GUI interface.
 - Allow custom reviews for prediction.

## 🤝 Contribution
Feel free to fork this project, raise issues, and submit Pull Requests.

🙌 Acknowledgements
IMDb for the dataset.
Textblob and NLTK for processing tools.
