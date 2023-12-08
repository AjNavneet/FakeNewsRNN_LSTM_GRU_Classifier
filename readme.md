# Detecting Fake News with RNN, LSTM, and GRU

## Business Overview

- Understanding Fake News:
  - Fake news involves the intentional dissemination of false or misleading information presented as news, designed to deceive the audience.

- Evolution of News and Digital Media:
  - News media has transitioned from traditional forms like newspapers to digital platforms such as online news portals, blogs, social media, and news mobile apps. While these platforms offer quick access to information, they also face the challenge of spreading fake news with negative motives, impacting various domains like politics, sports, health, entertainment, and science.

- Magnitude of the Problem:
  - The rise of the Internet and social media has facilitated the widespread distribution of untrue and biased information. Fake news is not confined to a specific domain and has witnessed a significant surge in the past decade. Instances like the 2016 US presidential election and the ongoing COVID-19 pandemic underscore the severity of the issue, with misinformation causing panic and influencing public opinion.

- Proposed Solution:
  - Given the pervasive nature of fake news, it is imperative to differentiate it from authentic news. Manual fact-checking is time-consuming and skill-intensive. Hence, leveraging machine learning and artificial intelligence for automated detection becomes crucial. Our focus is on text-based news content, employing advanced natural language processing techniques to identify deceptive articles.

---

## Data Description

- Dataset Overview:
  - Utilized the Fake News dataset from Kaggle to employ deep learning techniques, specifically Sequence to Sequence programming, for classifying unreliable news articles.
  - Attributes in the training dataset include id, title, author, text, and label (1 for unreliable, 0 for reliable).

---

## Tech Stack

- Language: `Python`
- Libraries: `Scikit-learn`, `Tensorflow`, `Keras`, `Glove`, `Flask`, `nltk`, `pandas`, `numpy`

---

## Approach

1. **Data Cleaning / Pre-processing:**
   - Addressing missing values, outliers, and categorical data.
   - Combining and cleaning text, removing special characters.

2. **Sequence Data Preparation:**
   - Tokenization post pre-processing.
   - Building a vocabulary for text filtering, determining maximum vocabulary size.
   - Preparing sequence data with considerations for vocabulary, maximum sequence length, and padding.

3. **Word Embedding:**
   - Utilizing pre-trained GloVe embeddings to convert text data into meaningful numerical vectors.

4. **Building Sequence Models:**
   - Constructing sequence layers with embedding, Dense, and Dropout, featuring:
     - Simple RNN
     - LSTM
     - GRU

5. **Model Training Validation:**
   - Finalizing models based on evaluation metrics like confusion matrix and accuracy.

6. **Model Comparison:**
   - Comparing models in terms of performance, stability, and computation time.

---

## Modular Code Overview

### 1. `input`

   - Contains data files for analysis (e.g., `submit.csv`, `test.csv`, `train.csv`).
   - Includes a `glove` folder housing the GloVe embedding file.

### 2. `src`

   - Core project folder housing modularized code.
   - Includes:
      - `ML_pipeline`: Functions organized into Python files.
      - `engine.py`: Main execution file calling functions from `ML_pipeline`.

### 3. `output`

   - Contains two subfolders:
      - `models`: Stores trained models for future use.
      - `reports`: Includes a CSV file documenting model details and accuracy.

### 4. `lib`

   - Reference folder containing the original IPython notebook.

---
