**Binary Tweets Classification**

# Introduction
The rise of social media in couple of years has changed the general perspective of networking, socialization, and personalization. Use of data from social networks for different purposes, such as election prediction, sentimental analysis, marketing, communication, business, and education, is increasing day by day. Precise extraction of valuable information from short text messages posted on social media like Twitter, is a collaborative task. Twitter has emerged as a prominent platform for users to share their thoughts, opinions, and experiences in the form of short text messages, known as tweets. These tweets not only serve as a means of personal expression but also contribute in enriching the real-time data.

In this paper, we analyse tweets to classify data into a political or sports tweets. The challenge, however, lies in the sheer volume and brevity of tweets. With a limited character count, tweets often embody informal language, abbreviations, slang, and context-dependent expressions, making their accurate classification a complex endeavour. The information from tweets is extracted the bag-of word technique and using the XGBoost as approach.
# Data Preprocessing 
- ## **Data**
Labelled Data:  6525 rows of data (TweetId, Tweets, Labels)

Train set: 80% of all data, used for training.

Test set: 20 of all data, used for testing.

Unlabelled Data:  2610 rows of data (TweetId, Tweets)
- ## **Classes Balance**
` `For further exploring the data, I looked at the distribution of the different classes and there is a slightly gab between the two classes.

![Aspose Words d0a3de52-a4be-4b2e-9b07-bbf146c1fe71 005](https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/c21ec4a2-9214-4781-9828-7642f37c0658)


- ## **Missing Data**
We observed that there is no missing data in each column.

![Aspose Words d0a3de52-a4be-4b2e-9b07-bbf146c1fe71 006](https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/b62f76d6-68f8-47bf-8593-09ff855f9329)

- ## **Cleaning Data**
In this step we will perform some steps to clean the data as shown in the schema below, to reduce the features vector size without removing any word that can give meaning to the classification.

![Aspose Words d0a3de52-a4be-4b2e-9b07-bbf146c1fe71 007](https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/908e6d15-86e6-4702-be97-1ada8c5b6d47)


# Features Extraction
Now that we have a cleaned tweet, we are ready for extracting the features, we will use the "Bag of Words" technique (BoW), that is a popular technique used in natural language processing (NLP) to represent text data in a numerical format that can be used by machine learning algorithms. It's a simple and effective way to convert text data into a structured format suitable for various tasks like classification, clustering, and more. 

However, it doesn't capture the word order or context, which might be important for some tasks (like sentiment analysis), but because we have a simple task (binary classification) with a simple tweet, we just need to extract the main and the most important keywords for example:

- Politics Words: election, vote, government …
- Sports Words: team, player, game, win …

![Aspose Words d0a3de52-a4be-4b2e-9b07-bbf146c1fe71 008](https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/dea0aaf6-3f9a-4ecf-b7e2-08df6c28c373)


# Model training 
Now that we have the features, each tweet is represented by a high-dimensional vector based on the size of the vocabulary, as mentioned before we have just a simple task, so we chose as approach the **XGBoost** model, it's a powerful gradient boosting algorithm that is known for its ability to: 

- Capture complex relationships and non-linear patterns in the data.
- Provides feature importance scores, helping to understand which words or features are contributing the most to the classification decisions.
- Prevent overfitting by using **Regularization Techniques**.
- Combining multiple weak learners to create a strong model.

XGBoost will learn to find patterns in the word occurrences that are indicative of the class labels by analysing which words from the vocabulary contribute the most to the classification decisions. However, as with any algorithm, success depends on proper preprocessing, feature engineering, and hyperparameter tuning. to achieve the best performance for the classification task.

![Aspose Words d0a3de52-a4be-4b2e-9b07-bbf146c1fe71 009](https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/a9520ded-87a4-4fe9-b077-b688fe1e4e04)




# Hyper-parameter tuning
To achieve the best performance for the task, we apply the grid search algorithm that is a technique used for hyperparameter tuning to find the best combination of hyperparameters that maximize the model's performance along with cross validation, here some parameters that we chose (based on research and testing):

![Aspose Words d0a3de52-a4be-4b2e-9b07-bbf146c1fe71 010](https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/a4bac1fe-1901-4f75-86af-27bff38dc969)


The best hyperparameter combination is {'max\_depth': 7 ,'learning\_rate': 0.5,'n\_estimators': 200, 'gamma': 0.4} with the accuracy of 92%.
# Performance
- ## **Confusing Matrix** 
Also known as an error matrix, is a table used to describe the performance of a classification model, it provides a detailed view of the model's performance across different classes, allowing you to assess where it might be making mistakes.

The main diagonal of the confusion matrix (from top left to bottom right) represents the instances that were correctly classified, while the off-diagonal elements represent the instances that were misclassified.

![Aspose Words d0a3de52-a4be-4b2e-9b07-bbf146c1fe71 011](https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/736f74c0-eb7a-4b9e-b808-59a990fbfdb6)


- ## **Accuracy and Error Rate**
From the confusing matrix, we can calculate the error rate and the accuracy of the model, which is the most common metric used in the evaluation.

![Aspose Words d0a3de52-a4be-4b2e-9b07-bbf146c1fe71 012](https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/70fe4958-c9e5-45c3-b5fe-dc110ff5f6a6)


The accuracy of the model is:

Before Grid Search: 90.11%

After Grid Search: 92.33%
- ## **Accuracy Paradox**
We can’t always depend on accuracy because we can find some scenarios (2) that give us misleading accuracy

![Aspose Words d0a3de52-a4be-4b2e-9b07-bbf146c1fe71 013](https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/11da71cc-3bb6-4195-a3bd-581759206cf3)

- ## **K-Fold**
k-fold is a technique used in machine learning to evaluate the performance of a model. The data set is divided into k subsets, and the model is trained and tested k times, each time using a different subset for testing and the remaining subsets for training.

<table>
  <tr>
    <td ><img src='https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/4eb534fc-c48e-4ddc-a507-354cbd7ced10'></td>
    <td><img src='https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/ac94dbdf-10cc-4dfa-a599-56226e29d428'></td>
  </tr>
</table>

We chose 5 folds and this the result:

![Aspose Words d0a3de52-a4be-4b2e-9b07-bbf146c1fe71 016](https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/24af66b8-4007-47b7-b516-6f56b4afffdb)


# Other Approaches
- ## **Word Embeddings**
Word embeddings are dense vector representations of words in a continuous space. Techniques like Word2Vec, GloVe, and FastText learn these embeddings based on the context in which words appear. Word embeddings capture semantic relationships between words and allow the model to understand word similarities.

![Aspose Words d0a3de52-a4be-4b2e-9b07-bbf146c1fe71 017](https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/a2493f46-f800-48ce-93d6-2392cc1b6d9e)


- ## **LSTM**
LSTMs have memory cells that allow them to capture information over long sequences. The architecture includes an input gate, an output gate, and a forget gate. These gates control the flow of information through the memory cells. In the case of tweet classification, the LSTM would process the sequences of word embeddings from the tweet.

In our case, it is an interesting approach, because it can capture both the simplicity of BoW and the sequence modelling capabilities of LSTMs.

![Aspose Words d0a3de52-a4be-4b2e-9b07-bbf146c1fe71 018](https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/feb64b91-7e62-4e29-9670-853152143a87)


- ## **Transfer Learning**
Transfer learning involves leveraging pre-trained models that have been trained on large datasets for general language understanding and then fine-tuning them on the specific task with a smaller dataset.

![Aspose Words d0a3de52-a4be-4b2e-9b07-bbf146c1fe71 019](https://github.com/SDAllouche/tweet_classification_xgboost/assets/102489525/aae23033-e993-4c00-b5c1-c78d7758af58)


For our task, a common approach is to use pre-trained transformer models, such as **BERT** (Bidirectional Encoder Representations from Transformers). the model has shown exceptional performance on a wide range of natural language processing tasks and it can be highly effective for tweet classification without requiring specific code.
