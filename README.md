# Fake News Detection

This repository contains a project focused on detecting fake news using machine learning techniques. The project involves data preprocessing, exploratory data analysis, and training multiple models to classify news articles as either *fake* or *real*.

## Project Overview

- **Data Processing**: Text cleaning, tokenization, removal of stop words, and lemmatization using NLTK and spaCy.
- **Feature Engineering**: 
  - Traditional TF-IDF approaches.
  - Word2Vec embeddings using Gensim.
- **Modeling**:
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - XGBoost Classifier
- **Model Evaluation**: Accuracy, precision, recall, F1-score, and classification reports are used to assess model performance.
- **Hyperparameter Tuning**: RandomizedSearchCV and GridSearchCV from scikit-learn are used to optimize model parameters.

## Project Structure

- `fake-news-detection.ipynb` — Main notebook containing all preprocessing, feature engineering, model training, evaluation, and tuning steps.

## Libraries Used

- `numpy`
- `pandas`
- `nltk`
- `spacy`
- `gensim`
- `scikit-learn`
- `xgboost`
- `ydata_profiling`
- `scipy`

## How to Run

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run the `fake-news-detection.ipynb` notebook in Jupyter or any compatible environment.

## Results

The models achieved good performance, with XGBoost generally outperforming the others after hyperparameter optimization.

## Future Improvements

- Experiment with deep learning models (e.g., LSTM, BERT).
- Further optimize text preprocessing and feature extraction.
- Expand dataset for better generalization.

## License

This project is licensed under the MIT License.
