# nlp-sentiment-classification

# Sentiment classification of movie reviews (NLP – Exercise 9)

Binary sentiment classification (positive / negative) of movie reviews with
logistic regression on TF-IDF features, compressed with truncated SVD,
compared to a majority baseline. Based on the course lab notebook on text
classification (2026).

## Contents
- `2nd_nlp_text_classification.ipynb` – all code, with outputs
- `data/review_polarity.tar.gz` – the dataset (Cornell movie review polarity dataset v2.0, 2,000 reviews)
- `report.pdf` – the report
- `requirements.txt` – Python package versions

## How to run
1. Open the notebook in Google Colab (link at the top of the notebook) or in Jupyter.
2. Run all cells.

The dataset is included in `data/`. If it is missing, the notebook downloads it
automatically from https://www.cs.cornell.edu/people/pabo/movie-review-data/
and checks its SHA-256 checksum.

To run locally: `pip install -r requirements.txt`

## Method (summary)
- Train / development / test split: 1,050 / 350 / 600 reviews
- Text normalization (lowercase, negations expanded), TF-IDF unigrams + bigrams (5,000 features)
- Truncated SVD (k components) + logistic regression with L2 regularization (C)
- k and C tuned on the development set; the test set is used only once

## Results
| | Development macro-F1 | Test macro-F1 |
|---|---|---|
| Majority baseline | 0.333 | 0.333 |
| Logistic regression (SVD k = 500, C = 100) | 0.874 | 0.850 |
