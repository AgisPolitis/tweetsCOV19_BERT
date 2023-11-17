import numpy as np
import pandas as pd
from datasets import Dataset

from transformers import AutoTokenizer


def extract_positive_sentiment(y_total):
    """
    Extracts the positive sentiment from 
    the total sentiment of each tweet
    """
    y_pos = []
    for sentiment in y_total:
        y_pos.append(int(sentiment.split(" ")[0]))
    return np.array(y_pos)

def train_val_test_split(data_pos):
    """
    Splits the data into training, validation and test sets
    90% training, 5% val, 5% test
    """
    data_pos_train = data_pos.sample(frac=0.9, random_state=42)
    data_pos_remaining = data_pos.drop(data_pos_train.index)
    data_pos_test = data_pos_remaining.sample(frac=0.5, random_state=42)
    data_pos_val = data_pos_remaining.drop(data_pos_test.index)

    # Reduce further the size of the validation set for speed purposes
    data_pos_val = data_pos_val.sample(frac=0.1, random_state=42)

    # Sample equal ammount of tweets from each label to allow for balanced training
    # Change n to use a different amount of tweets (Only for training data)
    sampled_data_pos_train = data_pos_train.groupby('label').apply(lambda x: x.sample(n=4000, replace=True)).reset_index(drop=True)

    return sampled_data_pos_train, data_pos_val, data_pos_test

def tokenize_dataset(data):
    # Tokenizer from a pretrained model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(data["TweetText"], 
                     max_length=140, 
                     truncation=True, 
                     padding="max_length")

def prepare_data(tweets_df):
    """
    Prepares and returns the train, val and test datasets 
    in the Hugging face format
    """
    # Each tweet text field is a list of token so join them into a sentence
    tweet_text = tweets_df["TweetText"].values
    tweet_text_sentences = []
    for token_list in tweet_text:
        sentence = " ".join(token_list)
        tweet_text_sentences.append(sentence)
    tweet_text_sentences = np.array(tweet_text_sentences)

    # Concatenate the sentences with the sentiment
    y_pos = extract_positive_sentiment(tweets_df["Sentiment"].values)
    concatenated_pos = np.concatenate([tweet_text_sentences.reshape(-1, 1), y_pos.reshape(-1, 1)], axis=1)

    # Transform to dataframe
    data_pos = pd.DataFrame(concatenated_pos, columns=['TweetText', 'label'])

    # Change dataset labels to integers and modify to 0-4 interval
    data_pos["label"] =  data_pos["label"].astype(int)
    data_pos["label"] =  data_pos["label"].astype(int) - 1

    sampled_data_pos_train, data_pos_val, data_pos_test = train_val_test_split(data_pos)

    # Convert python dataframe to Hugging Face arrow dataset
    hg_data_pos_train = Dataset.from_pandas(sampled_data_pos_train)
    hg_data_pos_test = Dataset.from_pandas(data_pos_test)
    hg_data_pos_val = Dataset.from_pandas(data_pos_val)

    # Tokenize the datasets
    data_pos_train_tokenized = hg_data_pos_train.map(tokenize_dataset, batched=True)
    data_pos_test_tokenized = hg_data_pos_test.map(tokenize_dataset, batched=True)
    data_pos_val_tokenized = hg_data_pos_val.map(tokenize_dataset, batched=True)

    return data_pos_train_tokenized, data_pos_val_tokenized, data_pos_test_tokenized







