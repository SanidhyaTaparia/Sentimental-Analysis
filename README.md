# Twitter-Sentiment-Analysis
It is a Natural Language Processing Problem where Sentiment Analysis is done by classifying the **Positive tweets** from **Negative tweets** by machine learning models for classification,  text mining, text analysis, data analysis 
and data visualization. Sentiment analysis has become an
integral part of marketing and can also help government agencies to **analyze mental health problems.**

## Dataset Used
We have used the twitter sentiment analysis dataset ([Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)) which is
available on kaggle.

## Pre-Processing
First we have Cleaned and organised the data into something more readable and
manageable and then Preprocessed the data such that the observations (reviews) are
more concise, and more obviously positive or negative.
Text files are usually not the easiest to clean due to the flexibility of language. However,
there’re a few standard ways to clean text data.
  1. Filter out symbols (eg. question marks, full stops, etc)
  2. Remove stop words (ie. the non-polarising words like “wasn’t”, “shan’t”, etc)
  3. Lemmatising or Stemming

## Model Training

### XGBoost
XGBoost, which stands for **Extreme Gradient Boosting**, is an implementation of Gradient
Boosted decision trees. In this algorithm, decision trees are created in **sequential form**. Weights
play an important role in XGBoost. Weights are assigned to all the independent variables
which are then fed into the decision tree which predicts results.

#### Steps involved in XGBoost Model-
* Before we vectorize the text we need to tokenize it and then normalise it. Further
we called a Function to extract the content of Hashtags, because it also describes
the quality of a tweet. Whereas a Tag(@) does not describe the quality of a tweet
so we will ignore it.
* After that,the text data is into a matrix of token counts
and we are going to use Bag of Words (bow) using a CountVectorizer.
* After that the code implementation is done, which can be found at [XGBoost.ipynb](https://github.com/SanidhyaTaparia/Sentimental-Analysis/blob/main/Training/XgBoost.ipynb)


### Embedding + Stacked LSTM
LSTM (Long Short Term Memory Network) is an advanced type of RNN
(Recurrent neural network). RNN is a type of artificial neural network which uses
sequential data or time series data.While traditional deep neural networks
assume that inputs and outputs are independent of each other, the output of
recurrent neural networks depend on the prior elements within the sequence.

#### Steps involved in LSTM Model-
* **Tokenization**: It is a particular kind of document segmentation. It breaks up
text into smaller chunks or segments called tokens. A tokenizer breaks
unstructured data, natural language text, into chunks of information that can be
counted as discrete elements
* **Word embeddings using GloVe**: Embedding matrix is used in embedding
layer in the model to embed a token into its vector representation, that contains
information regarding that token or word.
* **Model Training**: The model consisted of two bidirectional LSTM layers, two
Dropout layers, one relu dense layer, and one sigmoid dense layer.


### Naive Bias
A Naive Bayes classifier is a probabilistic machine learning model that’s used for
classification tasks. The crux of the classifier is based on the Bayes theorem.
Naive Bayes algorithms are mostly used in sentiment analysis, spam filtering,
recommendation systems etc.

BERT makes use of **Transformer**, an attention mechanism that learns contextual
relations between words (or sub-words) in a text. In its vanilla form, Transformer includes
two separate mechanisms — an **encoder** that reads the text input and a **decoder** that
produces a prediction for the task. Since BERT’s goal is to generate a language model, only
the encoder mechanism of the Transformer is necessary.

