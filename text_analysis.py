#IMPORTS
import re
import os
import pandas as pd
from string import punctuation
from textblob import Word
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

# Downloading Additional NLTK Resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load and Explore Data
for dirname, _, filenames in os.walk('./IMDB'): 
    for filename in filenames:
        print(filename)

train = pd.read_csv("./IMDB/Train.csv")
print(train.head())
print(train.info())

# Visualize Data
fig = plt.figure(figsize=(5,5))
colors = ["skyblue", 'pink']
pos = train[train['label'] == 1]
neg = train[train['label'] == 0]
counts = [pos['label'].count(), neg['label'].count()]
plt.pie(counts, labels=["Positive", "Negative"], autopct='%1.1f%%', shadow=True, colors=colors, startangle=45, explode=(0, 0.1))
plt.show()

# Text Preprocessing
def transformations(dataframe):
    dataframe['text'] = dataframe['text'].apply(lambda words: re.sub('<.*?>','',words))  # HTML Tags removal
    dataframe['text'] = dataframe['text'].apply(word_tokenize)  # Word Tokenization
    dataframe['text'] = dataframe['text'].apply(lambda words: [x.lower() for x in words])  # Lower case conversion
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if x not in punctuation])  # Punctuation removal
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if not x.isdigit()])  # Number removal
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if x not in stopwords.words('english')])  # Stopword removal
    temp = dataframe['text'].apply(lambda words: " ".join(words))
    freq = pd.Series(temp).value_counts()[:10]
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if x not in freq.keys()])  # Frequent word removal
    dataframe['text'] = dataframe['text'].apply(lambda words: " ".join([Word(x).lemmatize() for x in words]))  # Lemmatization
    return dataframe

train = transformations(train)
valid = pd.read_csv("./IMDB/Valid.csv")
valid = transformations(valid)
test = pd.read_csv("./IMDB/Test.csv")
test = transformations(test)

# Word Cloud Visualization
def wordcloud_draw(data, color='white'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split() if (word != 'movie' and word != 'film')])
    wordcloud = WordCloud(stopwords=stopwords.words('english'), background_color=color, width=2500, height=2000).generate(cleaned_word)
    plt.figure(1, figsize=(10, 7))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

print("Positive Words")
positivedata = train[train['label'] == 1]['text']
wordcloud_draw(positivedata)

print("Negative Words")
negdata = train[train['label'] == 0]['text']
wordcloud_draw(negdata)

# Prepare Data for Model
X_train = train.text
Y_train = train.label
X_valid = valid.text
Y_valid = valid.label
X_test = test.text
Y_test = test.label

# Model Training
clf = Pipeline([
    ('preprocessing', CountVectorizer()),
    ('classifier', LogisticRegression(dual=False, max_iter=2000))
])
clf.fit(X_train, Y_train)

# Model Evaluation
print("Validation Score:", clf.score(X_valid, Y_valid))
print("Test Score:", clf.score(X_test, Y_test))

# Predictions
p = clf.predict(X_test)
print(f'Number of reviews classified as Positive: {list(p).count(1)}')
print(f'Number of reviews classified as Negative: {list(p).count(0)}')