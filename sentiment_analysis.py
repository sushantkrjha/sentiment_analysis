# %%
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import string

import matplotlib.pyplot as plt



df = pd.read_excel('/home/prashant/Download/user_review.xls')

df.dropna(inplace=True)  # Remove rows with null values

unnessaesary_col_list=[]
for un_col in unnessaesary_col_list:
    df.drop(un_col, axis='columns')


def preprocess_text(text):
    text = text.lower()
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


df['clean_review'] = df['review'].apply(preprocess_text)
#print(df.head())
#nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    sentiment_score = sid.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
    
df['sentiment'] = df['clean_review'].apply(analyze_sentiment)

# Generate summary report
sentiment_distribution = df['sentiment'].value_counts(normalize=True) * 100


# Plotting the distribution

plt.figure(figsize=(8, 6))
sentiment_distribution.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.show()

#print(df.head(1))