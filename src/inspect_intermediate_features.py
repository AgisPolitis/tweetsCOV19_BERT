import argparse
import numpy as np
import pandas as pd

import torch
import evaluate
import transformers, datasets, accelerate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_saved_model(model_dir):
    """
    Loads a saved model AutoModel from Hugging Face
    documentation: https://huggingface.co/transformers/v3.0.2/model_doc/auto.html
    """
    # Tokenizer and model should match, if using a different model
    # change to the appropriate tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer

def load_tweets(dataset_path):
    """
    Loads tweets for which we will get the intermediate
    activations from the finetuned BERT
    """
    tweets_df = pd.read_csv(args.input_path, header=0)
    tweets_df = tweets_df.drop(columns=[col for col in tweets_df.columns if col not in ['TweetText', 'Sentiment']])
    # Get the first 50 tweets
    tweets = tweets_df["TweetText"][:40]
    return tweets

activation = {}
def get_activation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_tweets_activations(model, tokenizer, tweets):
    """
    Get the activations after the bert pooler layer
    """
    activations_list = []
    hook = model.bert.pooler.activation.register_forward_hook(get_activation('bert.pooler.activation'))

    for tweet in tweets:
        encoded_input = tokenizer(tweet, return_tensors="pt")
        output = model(**encoded_input)
        activations_list.append(activation['bert.pooler.activation'].squeeze().tolist())

    return activations_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path of the processed tweet data file")
    parser.add_argument("--model_dir", type=str, help="Directory of the saved finetuned model")
    args = parser.parse_args()

    tweets = load_tweets(args.input_path)
    model, tokenizer = load_saved_model(args.model_dir)
    activations = get_tweets_activations(model, tokenizer, tweets)

    print(f"[INFO] Done extracting activations - Each activation is a {len(activations[0])} feature vector")


