import argparse
import numpy as np
import pandas as pd

import torch
import evaluate
import transformers, datasets, accelerate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from data_utils import prepare_data


def compute_metrics(eval_pred):
    """ 
    Function to compute the validation metrics
    """
    metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")

def load_model(model_name, frozen_layers):
    """
    Function to load the specified pretrained model. The frozen_layers 
    parameter provides the option to either load the model with frozen
    weigths or finetune specific last layers.

    model_name: identifier of the model to load 
    frozen_layers: - "all" -> loads the model with frozen weights
                   - "all_but_last_1" -> unfreeze the final classification head
                   - other_options -> unfreeze the classification head, the pooler and the last 4 encoder blocks
    """
    # Load model for positive sentiment classification
    model_pos = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    
    # Set all the requires_grad parameters to False
    for name, param in model_pos.named_parameters():
        param.requires_grad = False
    
    if frozen_layers == "all":
        # Unfreeze the final classification head
        for name, param in model_pos.named_parameters():
            if "classifier" in name:
                  param.requires_grad = True
        
    elif frozen_layers == "all_but_last_1":
        # Unfreeze the classification head, the pooler and the last encoder block
        for name, param in model_pos.named_parameters():
            if "classifier" in name or "pooler" in name or "encoder.layer.11" in name:
                  param.requires_grad = True

    else: 
        # Unfreeze the classification head, the pooler and the last 4 encoder blocks
        for name, param in model_pos.named_parameters():
            if "classifier" in name or "pooler" in name or "encoder.layer.11" in name \
                or "encoder.layer.10" in name or "encoder.layer.9" in name or "encoder.layer.8" in name: 
                  param.requires_grad = True

    return model_pos

def train(train_dataset, eval_dataset, frozen_layers):
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/model/",          
        num_train_epochs=2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_pos = load_model("bert-base-uncased", frozen_layers=frozen_layers)
    model_pos.to(device)

    print(f"[INFO] Training model on {device} ...")

    # Train the model
    trainer = Trainer(
        model=model_pos,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

    return trainer

def test(trainer, test_dataset):
    """
    Evaluates the trained model on the test dataset
    """
    preds = trainer.predict(test_dataset)
    print(f"------ The results on test dataset are: ------")
    print(f"Accuracy is: {preds.metrics['test_loss']}")
    print(f"F1 score is: {preds.metrics['test_f1']}")
    return 

def save_model(model, save_dir):
    model.save_pretrained(save_dir)
    print("[INFO] Model has been saved succesfully")
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path of the processed tweet data file")
    parser.add_argument("--frozen_layers", type=str, choices=["all", "all_but_last_1", "other"], help="Which layers to keep frozen during finetuning")
    args = parser.parse_args()

    # Keep only the 'TweetText', 'Sentiment' columns
    tweets_df = pd.read_csv(args.input_path, header=0)
    tweets_df = tweets_df.drop(columns=[col for col in tweets_df.columns if col not in ['TweetText', 'Sentiment']])

    # Prepare train, validation and test data
    data_pos_train_tokenized, data_pos_val_tokenized, data_pos_test_tokenized = prepare_data(tweets_df)

    # Train the model
    trainer = train(data_pos_train_tokenized, data_pos_val_tokenized, args.frozen_layers)

    # Test the model
    test(trainer, data_pos_test_tokenized)

    # Save the model
    save_model(trainer.model, "./saved_models")




