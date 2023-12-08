# Fake News Classification Using RNN, LSTM and GRU

## Business Overview

- What is Fake News?
  - Fake news is the deliberate presentation of (typically) false or misleading claims as news, where the claims are misleading by design.

- How News and digital media evolved?
  - The news media evolved from newspapers, tabloids, and magazines to a digital form such as online news platforms, blogs, social media feeds, and many news mobile apps. News outlets benefitted from the widespread use of social media/mobile platforms by providing updated news in near real time to its subscribers. It became easier for consumers to acquire the latest news at their fingertips. So, These digital media platforms become very powerful due to their easy accessibility to the world and ability to allow users to discuss and share ideas and debate over issues such as democracy, education, health, research, and history. However, apart from advantage, false/fake news articles on digital platforms are getting very common and mainly used with a negative intent for their own benefit such as political and financial benefit, creating biased opinions, manipulating mindsets, and spreading absurdity.

- How big is this Problem?
  - With the rapid adoption of Internet, social media, and digital platforms (such as Facebook, Twitter, news portals, or any social media), anybody can spread untrue and biased information. It is virtually impossible to prevent Fake News from being created. There has been a rapid increase in the spread of fake news in the last decade, it's not limited to any one domain like politics but covering various other domains such as sports, health, history, entertainment and also science and research. If we take the 2016 US presidential election, there were lots of biased and fake news published to influence. Another example could be of COVID-19, we generally come across many misleading/fake news every day which can have serious consequences and may lead to create panic among people and spread the pandemic more rapidly.

- What is Solution?
  - Therefore, It is important and absolutely necessary to identify and differentiate Fake News from real news. One of the ways is to determine by expert and fact check of every news, but this is time-consuming and requires skills which cannot be shared. Second, we can automate the detection of Fake News by using the techniques of Machine learning and Artificial Intelligence. The Online news content has diverse unstructured format data(such as documents, videos, and audios), here we will concentrate on text format news. With the advancement of and Natural language processing It is possible now that we can identify the deceptive and fake nature of articles or sentences. There is widespread study and experimentation happening in this area to identify the Fake news for all medium(Video, audio and Text) news.

---

## Data Description

- In our study, we used the Fake news dataset from Kaggle to classify unreliable news articles as Fake news using Deep learning Technique Sequence to Sequence programming.
- A full training dataset with the following attributes:
  - id: unique id for a news article
  - title: the title of a news article
  - author: author of the news article
  - text: the text of the article; could be incomplete
  - label: a label that marks the article as potentially unreliable
    - 1: unreliable
    - 0: reliable

---

## Tech Stack

- Language: `Python`
- Libraries: `Scikit-learn`, `Tensorflow`, `Keras`, `Glove`, `Flask`, `nltk`, `pandas`, `numpy`

---

## Approach

1. Data cleaning / Pre-processing (outlier/missing values/categorical):
   - Removing Missing record
   - Merge all text together
   - Removing special characters from text

2. Sequence Data Preparation:
   - Tokenizing text after preprocessing
   - Build Vocabulary to filter text sets: Choose the length of the maximum vocabulary size
   - Sequence data preparation:
     - Use vocab
     - Maximum sequence length
     - Padding

3. Word Embedding:
   - This is a step where we convert text data to meaningful numerical vectors. We use a pre-trained glove to convert into a numeric vector.

4. Build Sequence Model:
   - Building Sequence layer with embedding, Dense, Dropout with below sequence layer:
     - Simple RNN
     - LSTM
     - GRU

5. Validate Model Training:
   - Which model will be finalized on the basis of the following:
     - Confusion matrix
     - Accuracy

6. Model comparison:
   - Model comparison in terms of performance, stability, and computation time

---

## Modular Code Overview

### 1. input

The `input` folder contains all the data that we have for analysis. In our case, it will contain three CSV files which are:
- `submit.csv`
- `test.csv`
- `train.csv`

It also has another folder called `glove` which contains the glove embedding file.

### 2. src

The `src` folder is the heart of the project. This folder contains all the modularized code for all the above steps in a modularized manner. It further contains the following:
- `ML_pipeline`
- `engine.py`

The `ML_pipeline` is a folder that contains all the functions put into different Python files which are appropriately named. These Python functions are then called inside the `engine.py` file.

### 3. output

The `output` folder contains two folders. They are:
- `models`: The models folder contains all the models that we trained for this data saved as reusable files. These models can be easily loaded and used for future use, and the user need not have to train all the models from the beginning.
- `reports`: The report folder contains a CSV file which stores all the models trained along with their accuracy and other details.

### 4. lib

The `lib` folder is a reference folder. It contains the original IPython notebook.

---
