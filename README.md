# TweetDigest

TweetDigest is a tool for performing analysis on the tweets posted on Twitter website. Analyzing tweets can provide insights into public opinion, sentiment, trends, and behavior on a particular topic or issue. Twitter's real-time nature and vast user base make it a valuable source of data for social and political analysis, business intelligence, and market research. 

TweetDigest is an application developed using python and HTML/CSS. The model creation, its application and evaluation is developed using python. The back-end server is developed using Flask web framework. The front-end UI is developed using HTML/CSS, Bootstrap and Javascript.

The objective of the project is to fetch tweets based on a search keyword and perform analysis on the tweets fetched. The tweets are fetched using snscrape library. Once the tweets are collected, they are cleaned to remove any noise like URLs, hashtags and user mentions.
This cleaned data is applied on the pre-trained transformer models of HuggingFace to perform analysis. We have performed the following analysis on tweets:
1. Text summarization using BART model
2. Multi-label and multi-class classification using zero-shot-classification
3. Sentiment analysis using sentiment-analysis
4. Emotional analysis using zero-shot-classification

Once the models are developed, the models are evaluated for pre-fetched tweets that are stored in a csv file. The following metrics are used for evaluation:
1. bert_score to evaluate text summarization
2. hamming_loss to evaluate text classification, sentiment analysis and emotional analysis

To run the application, the following command needs to be executed to install all the packages needed:
** pip install -r requirements.txt **

This command can be executed to install the packages globally. The above command can be executed after the creation of virtual environment. Once the required libraries are installed, the back-end server can be run using the below command:
** python3 app.py **
Once the server is run, you can see that the server is now available on port 5000. 

To view the webpage, open a browser and type in the following URL to open the homepage:
** http://localhost:5000/ **
On navigation to different tabs, the UI triggers different analysis to be performed. The user can type in the search keyword. On clicking the buttons, the application executes different analysis to be performed. Once the result is predicted, the UI displays the result.
