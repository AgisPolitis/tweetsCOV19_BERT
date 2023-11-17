import argparse
import string
import re
import numpy as np
import pandas as pd

import emoji
import preprocessor as p
import contractions

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def preprocess(tweets_df):
    """
    Applies the following preprocessing steps to 
    the complete tweets dataframe

    1. Duplicate and null text tweets removal
    2. URL, user mentions, numbers and reserved words removal
    3. Add spaces between emojis
    4. Expand contractions (e.g. you're -> you are)
    5. Removes punctuation and extra spaces
    6. Tokenization using nltk
    7. Removing stopwords
    8. Lowercase and lemmatization
    """
    # (Optional) if preprocessing is taking to long because of the
    # large size of the dataset, you can use a subset of the total tweets by
    # tweets_df = tweets_df.head(number_tweets_to_keep)
    tweets_df = tweets_df.drop('Unnamed: 0', axis=1)
    print("[INFO] Remove Duplicates and nulls ...")
    tweets_df.drop_duplicates(keep='first', inplace=True)
    tweets_df.dropna(subset=['TweetText'], inplace=True)
    print("[INFO] Remove URLs ...")
    tweets_df = remove_urls_mentions_reserved_numbers(tweets_df)
    print("[INFO] Remove spaces ...")
    tweets_df['TweetText'] = tweets_df['TweetText'].apply(lambda text: spaces_between_emojis(str(text)))
    print("[INFO] Expand contractions ...")
    tweets_df['TweetText'] = tweets_df['TweetText'].apply(lambda text: expand_contractions(str(text)))
    print("[INFO] Remove punctuation ...")
    tweets_df['TweetText'] = tweets_df['TweetText'].apply(lambda text: remove_punctuation(str(text)))
    print("[INFO] Remove extra spaces ...")
    tweets_df['TweetText'] = remove_useless_spaces_and_symbols(tweets_df)
    print("[INFO] Tokenize ...")
    tweets_df['TweetText'] = tweets_df['TweetText'].apply(lambda text: tokenize(str(text)))
    print("[INFO] Remove stopwords...")
    tweets_df['TweetText'] = tweets_df['TweetText'].apply(lambda text_tokens: remove_stop_words(text_tokens))
    print("[INFO] Lowercase ...")
    tweets_df['TweetText'] = tweets_df['TweetText'].apply(lambda text_tokens: lowercase(text_tokens)) 
    print("[INFO] Lematize ...")
    tweets_df['TweetText'] = tweets_df['TweetText'].apply(lambda text_tokens: lemmatize(text_tokens))
    print("[INFO] Remove underscores ...")
    tweets_df['TweetText'] = tweets_df['TweetText'].apply(lambda text_tokens: remove_underscores(text_tokens))
    print("[INFO] Remove empty token lists ...")
    tweets_df = tweets_df[tweets_df['TweetText'].apply(lambda text_tokens: len(text_tokens) > 0)]
    return tweets_df

def expand_contractions(text):
    expanded_words=[]
    for word in text.split():
        try:
            expanded_words.append(contractions.fix(word))
        except:
            expanded_words.append(word)
    return ' '.join(expanded_words)

def remove_punctuation(text):
    """
    Removes punctuation from the text of a tweet
    """
    punct = string.punctuation

    sentence = ""
    for word in text:
        if word not in punct:
            sentence += word
    return sentence

def remove_urls_mentions_reserved_numbers(tweets_df):
    """"
    Removes URLs, user mentions, numbers and 
    reserved words from the text field of tweets
    """
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.NUMBER)
    tweets_df['TweetText'] = tweets_df['TweetText'].apply(lambda x : p.clean(str(x)))
    return tweets_df

def spaces_between_emojis(text):
    return ''.join((' ' + c + ' ') if c in emoji.distinct_emoji_list(text) else c for c in text)

def remove_useless_spaces_and_symbols(tweets_df):
    return tweets_df['TweetText'].str.replace('"', '').str.replace('’', '').str.replace('“', '').str.replace('”', '').str.replace('…', '').str.replace('•', '').str.replace('…', '').str.replace('»', '').str.replace('«', '').str.replace('\s\s+', ' ')

def tokenize(text):
    """
    Tokenizes a tweet using nltk
    """
    word_tokens = word_tokenize(text)
    return word_tokens
    
def remove_stop_words(text):
    """
    Removes stopwords from the text of a tweet
    """
    stop_words = stopwords.words('english')
    stop_words = set(stop_words + ['–','•','…','us','amp','»','«'])
    filtered_sentence = [word.lower() for word in text if not word.lower() in stop_words]
    return filtered_sentence

def lowercase(text):
    sentence = [w.lower() for w in text]
    return sentence
    
def lemmatize(text):
    wordnet_lemmatizer = WordNetLemmatizer()

    lemas = []
    for word in text:
        lemas.append(wordnet_lemmatizer.lemmatize(word))
    return lemas

def remove_underscores(text):
    sentence=[]
    for word in text:
        if '—' in word:
            words = re.split('—', word)
            for t in words:
                if t != '':
                    sentence.append(t)
        else:
            sentence.append(word)
    return sentence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path of the tweet data file')
    parser.add_argument('--output_path', type=str, help='Path to save the processed tweet data')
    args = parser.parse_args()

    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("punkt")
    
    # Read data in dataframe format
    tweets_df = pd.read_csv(args.input_path, header=0)

    print("[INFO] Start preprocessing tweets ...")
    processed_tweets_df = preprocess(tweets_df)
    print("[INFO] End of tweets preprocessing ...")

    # Save processed data to output file
    print(f"[INFO] Saving processed tweets at {args.output_path} ...")
    processed_tweets_df.to_csv(args.output_path, index=False)
