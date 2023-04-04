import tensorflow as tf
import transformers as ts
import tweets
import logging
import analysis

logger = logging.getLogger("service")
logging.basicConfig(level=logging.INFO)
logger.setLevel('INFO')

"""
Function used by flask app to generate summary using bart model for tweets based on keyword
@param: keyword entered by user
@return: summary generated
"""
def summarize_bart(keyword):
    model_name = 'facebook/bart-large-cnn'
    cache_dir = 'cache/'
    logging.info("***Starting to create model and summarize***")
    # Models to perform summarization
    tokenizer = ts.AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
    model = ts.TFAutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir = cache_dir)
    bart_summarizer = ts.pipeline("summarization", model=model, tokenizer=tokenizer)
    logging.info("***Fetching tweets***")
    # Calling function to fetch tweets
    tweets_data = tweets.get_tweets(keyword) 
    # Summarizing tweets fetched and returning summary
    bart_summary = analysis.bart_analysis(bart_summarizer, tweets_data)
    logging.info("***Completed generating summary and returning the summary***")
    return bart_summary

"""
Function used by flask app to classify tweets based on keyword
@param: keyword entered by user
@return: classifier response
"""
def classify(keyword):
    cache_dir = 'cache/'
    logging.info("***Starting to create model and classify***")
    # Model to perform classification
    classifier = ts.pipeline('zero-shot-classification', cache_dir=cache_dir)
    # Labels that tweets will be classified into
    labels = ['politics', 'sports', 'science', 'finance', 'entertainment']
    # Calling function to fetch tweets
    tweets_data = tweets.get_tweets(keyword)
    logging.info("***Fetching tweets***")
    # Classifying the tweets received and returning classifier response
    response = classifier(tweets_data, labels)
    logging.info("***Completed generating summary and returning the summary***")
    return response

"""
Function used by flask app to perform sentiment analysis on tweets based on keyword
@param: keyword entered by user
@return: classifier response
"""
def sentimentAnalysis(keyword):
    cache_dir = 'cache/'
    logging.info("***Starting to create model and analyze***")
    # Model to perform sentiment analysis
    classifier = ts.pipeline('zero-shot-classification', cache_dir=cache_dir)
    # Labels that tweets will be classified into 
    labels = ['positive', 'negative']
    # Calling function to fetch tweets
    logging.info("***Fetching tweets***")
    tweets_data = tweets.get_tweets(keyword)
    # Classifying the tweets received and returning classifier response
    response = classifier(tweets_data, labels)
    logging.info("***Completed generating summary and returning the summary***")
    return response