# Sentiment Analysis using BERT-based Models

This repository contains code for sentiment analysis using BERT-based models leveraging PyTorch and Hugging Face's Transformers library. The code utilizes pre-trained BERT models for sentiment classification on text data.

## Overview

The code provides a comprehensive sentiment analysis pipeline employing BERT-based models. It includes:

- **Environment Setup**: Utilizes PyTorch for neural networks, Transformers for BERT-based models, and Datasets for dataset management.
  
- **Data Preprocessing**: Loads and preprocesses IMDb dataset for sentiment analysis, splitting it into training and testing subsets.
  
- **Model Configuration**: Initializes a BERT-based sequence classification model (BertForSequenceClassification) and sets up an AdamW optimizer.
  
- **Training Loop**: Runs a training loop over multiple epochs, optimizing the model using the training dataset.
  
- **Evaluation**: Evaluates the model's performance on the test dataset, computing accuracy for sentiment prediction.

## Usage

1. **Environment Setup**: Ensure Python environment is set up and required libraries are installed (`torch`, `transformers`, `datasets`, `sklearn`).
  
2. **Data Preparation**: Use `load_dataset` to fetch IMDb dataset and preprocess it with `train_test_split` for training and testing subsets.
  
3. **Model Training**: Configure and train the model using the prepared dataset and the provided training loop.

4. **Evaluation**: Evaluate the trained model's performance on the test set using accuracy or other relevant metrics.

## Dependencies

- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn

## Notes

- Ensure GPU availability for faster model training, although CPU usage is also supported.
  
- Experiment with hyperparameters, dataset sizes, and different BERT model variants for improved results.

Feel free to explore and modify the code for other text classification tasks or datasets!

