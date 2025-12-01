# Fake News Detection

Detect fake news using machine learning techniques.

## Team Information

- **Ashish Bhusal** - 018320627
- **Harishita Gupta** - 018323331
- **Indraneel Sarode** - 018305092
- **Nishan Paudel** - 018280561

---

## Datasets

This project uses multiple public fake-news datasets:

- **Fake and Real News Dataset** (Kaggle)  
  - Stored as `data/Fake.csv` (fake articles) and `data/True.csv` (real articles).
- **WELFake Dataset** (Kaggle)  
  - Expected as `data/WELFake_Dataset.csv`.

These are merged into a single dataset containing article **title**, **text**, and **label** (real/fake). Duplicates and empty texts are removed before training.

> Note: The notebook assumes the CSV files are available in a `data/` directory.  
> If you are running this outside Google Colab, update the `base_dir` path at the top of the notebook to point to your local project folder.

## Project Overview

The rapid spread of misinformation on digital platforms poses serious social and financial risks, making automated fake news detection increasingly important.

In this project, we build a text-based fake news detector that takes the title and body of a news article as input and predicts whether the article is real or fake. Our goal is to balance simplicity, interpretability, and efficiency while still achieving strong performance. To that end, we compare two classical machine learning models—Multinomial Naive Bayes and Logistic Regression with hyperparameter tuning—against a transformer-based approach, where we fine-tune a DeBERTa model using the Hugging Face transformers library.

## Methods

### Preprocessing & Feature Engineering

All preprocessing and modeling steps are implemented in:

> `Fake_News_Detection.ipynb`

Key steps:

- Load and merge Kaggle Fake/Real and WELFake datasets  
- Clean text:
  - Lowercasing  
  - Removing punctuation and stopwords  
  - Stemming (NLTK)  
- Build a combined **text field** from title + body  
- Extract features:
  - **TF-IDF** representation of the text  
  - **Numerical style features**, e.g.  
    - Sentence count  
    - Average sentence length  
    - Counts of question marks, exclamation marks, and quotes  
    - Uppercase character proportion  

Sparse TF-IDF features and dense numerical features are combined for classical models using `scipy.sparse.hstack`.

### Classical Models

- **Multinomial Naive Bayes**  
- **Logistic Regression**
  - Trained on TF-IDF + numerical features  
  - Hyperparameter tuning with `GridSearchCV` and an F1-score–based scorer  
  - Final tuned model saved as:
    - `model/tuned_logreg_model.pkl` (via `joblib`)

### Transformer Model

- Uses Hugging Face `transformers` and `datasets` libraries
- Steps:
  - Create a concatenated `input_text` column (title + text)  
  - Train/validation/test split  
  - Tokenization up to 512 tokens  
  - Fine-tuning a DeBERTa-based model with `Trainer` and `TrainingArguments`  
  - Evaluate on the test set and compute accuracy, precision, recall, F1, and confusion matrix  
  - Model checkpoints saved under `model/` (e.g. `model/deberta_fake_news_model2` in the notebook)

## Evaluation

We evaluate models using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion matrix** (with fake = positive class)

The main focus is the **F1-score for the fake class**, since catching fake news is the primary goal.  
Results in the accompanying report show:

- Multinomial Naive Bayes performs worst among the three  
- Logistic Regression + TF-IDF + numerical features significantly improves performance  
- The transformer (DeBERTa) achieves the best overall metrics

## Key References

- CWang – LIAR dataset (2017)
- Hashmi et al. – BERT & XLNet
- Monti et al. – geometric deep learning / graph-based
- Kaliyar et al. – FakeBERT

  For full references, please check the project report.

## Roles

- **Ashish**: Dataset acquisition, cleaning, and preprocessing pipeline.
- **Nishan**: Feature engineering (TF-IDF, Word2Vec, BERT) and baseline models.
- **Harishita**: Deep learning model development (Deep learning Transformer).
- **Indraneel**: Evaluation metrics, error analysis, and final report/dashboard.

## Project Workflow

<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/6bd2331a-6a0d-47a9-b0a4-2f8ca84b3ecb" />

## Repository Structure

```text
fake-news-detection-main/
├─ Fake_News_Detection.ipynb     # Main notebook: EDA, features, models, evaluation
├─ data/
│  ├─ Fake.csv                   # Fake articles (Kaggle Fake & Real News)
│  ├─ True.csv                   # Real articles (Kaggle Fake & Real News)
│  ├─ WELFake_Dataset.csv        # (to be added) WELFake dataset
│  ├─ label_distribution.png     # Label distribution plot
│  └─ subject_distribution.png   # Subject distribution plot
├─ model/
│  └─ tuned_logreg_model.pkl     # Saved tuned Logistic Regression model
├─ README.md
├─ LICENSE
└─ .gitignore
