# Sentiment classification in TweetsCOV19 dataset using BERT

This repository contains the code for the Natural Language Processing project at the Machine Learning for Healthcare course at ETH Zurich. 

The goal of the project is to predict the sentiment of tweets in the TweetsCOV19 dataset. For that purpose the pretrained **bert-base-uncased** model from the Hugging Face library was utilized and finetuned in a subset of the tweets in the dataset.

## Setup 
Clone the repository, create a conda environment and install the required packages as follows:

```
conda create --name tweetscov19 python=3.10
conda activate tweetscov19
pip install -r requirements.txt
```

## Code structure and how to use it 

1. The **preprocess_tweets.py** file contains the code for preprocessing the raw TweetsCOV19.csv dataset. It performs basic preprocessing operations such as cleaning the dataset from nan and duplicate entries, stopwords removal and lemmatization. It can be run with the following command: 

 ```
 python preprocess_tweets.py --input_path <PATH_TO_TweetsCOV19.csv> --output_path <PROCESSED_DATA_PATH>
 ```

 where `--input_path` indicates the path to the dataset and `--ouput_path` the path you wish the processed data to be saved.

 2. The **finetune_BERT.py** file contains the code for finetuning the **bert-base-uncased** model using the processed dataset from the previous step. It can be run with the following command:

 ```
 python finetune_BERT.py --input_path <PROCESSED_DATA_PATH> --frozen_layers <FROZEN_LAYERS_OPTIONS>
 ```

 where `--input_path` indicates the path to the processed dataset and `--frozen_layers` an option for finetuning different number of layers.

 3. The **data_utils.py** file contains necessary functions for loading the data into a format suitable for training.

 4. The **inspect_intermediate_features.py** contains code to extract feature vectors from the **bert.pooler.activation** layer for different input tweets. This can be useful for interpretability reasons as we can visualize (e.g. by applying PCA or TSNE) how the model represents tweets of different sentiment. It can be run with the following command:

```
 python inspect_intermediate_features.py --input_path <PROCESSED_DATA_PATH> --model_dir <DIRECTORY_OF_THE_SAVED_MODEL>
 ```
where `--input_path` indicates the path to the processed dataset and `--model_dir` the path to the finetuned BERT model

## TweetsCOV19 dataset
We used a subset of the TweetsCOV19 dataset that can be found [here](https://drive.google.com/file/d/1rt2HpivSqlO7PPX6WsTKGR-grC2pgUU4/view?usp=share_link). It reflects the societal discourse about COVID-19 on Twitter in the period of October 2019 until May 2020. The dataset has been annotated for the purpose of sentiment analysis. Each tweet has a score for positive (1 to 5) and negative (-1 to -5). We only consider the positive sentiments in our case. More information about the data can be found [here](https://data.gesis.org/tweetscov19/).