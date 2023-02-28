# Sentiment Analysis in Python
This repository contains a Jupyter notebook that performs sentiment analysis in Python using two different techniques: VADER (Valence Aware Dictionary and sEntiment Reasoner) - a bag of words approach, and Roberta Pretrained Model from Huggingface Pipeline.

# Libraries

* `pandas`;
* `numpy`;
* `matplotlib.pyplot`;
* `seaborn`;
* `nltk`;
* `tqdm`;
* `nltk.sentiment.SentimentIntensityAnalyzer`;
* `transformers`;
* `pipeline`;

# Data:

The data used in this project is available on the internet, but you can also use your own dataset. The data used in this project is the "Amazon Fine Food Reviews" dataset which is available on Kaggle. You can download it from here: **https://www.kaggle.com/snap/amazon-fine-food-reviews**.
You need to download this dataset and save it to your working directory.

Once you have downloaded the dataset, you can read it into a pandas DataFrame using the pd.read_csv() function:
``df = pd.read_csv('Reviews.csv')``

# Read in Data and NLTK Basics:

The notebook starts by importing the required libraries and reading in a dataset of Amazon Fine Food Reviews. The data is then reduced to the first 500 rows for faster processing. Basic exploratory data analysis is performed using matplotlib and seaborn. Then, NLTK is used to tokenize and tag the text data.

# VADER Seniment Scoring:

NLTK's SentimentIntensityAnalyzer is used to get the neg/neu/pos scores of the text. This uses a "bag of words" approach: stop words are removed and each word is scored and combined to a total score. The polarity score is run on the entire dataset using a for loop.

# Huggingface Pipeline Sentiment Analysis:

The notebook then uses the Hugging Face Transformers library to perform sentiment analysis on the same dataset. A pre-trained Roberta model is used to predict the sentiment of the text using the Hugging Face pipeline function. The pipeline function is also used to perform sentiment analysis on a single example text.
