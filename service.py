import tensorflow as tf
import transformers as ts
import tweets
import logging
import analysis

logger = logging.getLogger("service")
logging.basicConfig(level=logging.INFO)
logger.setLevel('INFO')

"""
Function to convert dictionary to text as a tree
@param: Dictionary 
@return: String as a tree
"""
def nested_dict_to_text(d, depth=0):
    text = ""
    for key, value in d.items():
        # New element in the dictionary
        text += "  " * depth + str(key) + ":\n"
        # New element is a key
        if isinstance(value, dict):
            text += nested_dict_to_text(value, depth+1)
        # New element is a list
        elif isinstance(value, list):
            for val in value:
                text += "  " * (depth+1) + "- " + str(val) + "\n"
        else:
            text += "  " * (depth+1) + str(value) + "\n" 
    return text

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
    labels = ['Technology', 'Politics', 'Entertainment', 'Sports', 'Science']
    # Calling function to fetch tweets
    tweets_data = tweets.get_tweets(keyword)
    logging.info("***Fetching tweets***")
    # Classifying the tweets received and returning classifier response
    response = classifier(tweets_data, labels)
    logging.info("***Completed generating summary and returning the summary***")
    return response

"""
Function used by flask app to perform multi-level and multi-class classification of tweets based on keyword
@param: keyword entered by user
@return: classifier response
"""
def multi_level_classify(keyword):
    cache_dir = 'cache/'
    logging.info("***Starting to create model and classify***")
    # Model to perform multi-level classification 
    classifier = ts.pipeline('zero-shot-classification', cache_dir=cache_dir)
    # Dictionary of the labels into which the tweets will be classified into
    labels = ['Health','Technology', 'Politics', 'Entertainment', 'Sports', 'Science']
    logger.debug("***Starting first-level classification***")
    tree = {
        'Health': ['Disease','Nutrition','Diet'],
        'Technology': ['Hardware', 'Software', 'AI'],
        'Politics': ['Domestic', 'International'],
        'Entertainment': {
            'Movies': ['Drama', 'Comedy', 'Action'], 
            'Music': ['Rock', 'Pop', 'Hip Hop'], 
            'Television': ['Reality', 'Comedy', 'Thriller']
        },
        'Sports': {
            'Team sports':['Football', 'Basketball', 'Tennis'],
            'Individual': ['Chess', 'Golf']
        },
        'Science': ['Physics', 'Chemistry', 'Biology']
    }
    tweets_data = tweets.get_tweets(keyword)
    # Classification of tweets based on the first-level labels
    response = classifier(tweets_data, labels)
    logger.info(f"***Response from classification model: {response}***" )
    classified_labels = {}
    for i in range(len(response['scores'])):
        # Picking all the labels with threshold greater than 0.2
        if response['scores'][i] > 0.2:
            predicted_label = response['labels'][i]
            # If the label is Entertainment or Sports, then classification is 3-level
            if predicted_label == 'Entertainment' or predicted_label == 'Sports':
                classified_labels[predicted_label] = {}
            # Other labels have 2-level classification
            else: 
                classified_labels[predicted_label] = []
    logger.debug("***Beginning second and third-level classification***")
    for label in classified_labels:
        # If next level is a dict, then keys of dict is picked
        if isinstance(tree[label],dict):
            labels = list(tree[label].keys())
        # Otherwise all the values for the current key is picked
        else:
            labels = tree[label]
        # Classification of tweets based on the second-level labels 
        response = classifier(tweets_data, labels)
        for i in range(len(response['scores'])):
            if response['scores'][i] > 0.4:
                if label == 'Entertainment' or label == 'Sports':
                    classified_labels[label][response['labels'][i]] = []
                    inner_labels = tree[label][response['labels'][i]]
                    # Classification of tweets based on the third-level labels
                    logger.debug("***Third-level classification is being done***")
                    inner_response = classifier(tweets_data, inner_labels)
                    for j in range(len(response['scores'])):
                        if inner_response['scores'][j] > 0.4:
                            classified_labels[label][response['labels'][i]].append(inner_response['labels'][j])
                else:
                    classified_labels[label].append(response['labels'][i])
    logger.info("***Classified tree: " + str(classified_labels))
    # Returning the classified labels as a text to display and dictionary for evaluation
    return nested_dict_to_text(classified_labels), classified_labels

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
    labels = ['Positive', 'Negative', 'Neutral']
    # Calling function to fetch tweets
    logging.info("***Fetching tweets***")
    tweets_data = tweets.get_tweets(keyword)
    # Classifying the tweets received and returning classifier response
    response = classifier(tweets_data, labels)
    logging.info("***Completed generating summary and returning the summary***")
    return response

"""
Function used by flask app to perform sentiment analysis on tweets based on keyword
@param: keyword entered by user
@return: classifier response
"""
def sentiment_emotional_analysis(keyword):
    logging.info("***Starting to create model and analyze***")
    # Models for sentiment and emotional analysis
    sentiment_analyzer = ts.pipeline('sentiment-analysis', model='distilbert-base-uncased')
    emotion_analyzer = ts.pipeline('zero-shot-classification')
    # Tree and corresponding map to labels
    tree = {
        "Positive": ["happy", "excited", "love", "satisfied", "confident"],
        "Negative": ["angry", "sad", "disappointed", "anxious", "frustrated"],
        "Neutral": ["neutral", "calm", "bored", "curious", "indifferent"]
    }
    label_dict = {
        'LABEL_0': 'Neutral',
        'LABEL_1': 'Positive',
        'LABEL_2': 'Negative'
    }
    # Sentiment and emotional models
    logger.info("***Performing sentiment and emotional analysis***")
    tweets_data = tweets.get_tweets(keyword)
    sentiment_label = label_dict[sentiment_analyzer(tweets_data)[0]['label']]
    response = emotion_analyzer(tweets_data, tree[sentiment_label])
    emotion_labels = []
    # Best emotions are picked
    for i in range(len(response['scores'])):
        if response['scores'][i] > 0.2:
            emotion_labels.append(response['labels'][i])
    logger.info(f"***Sentiment Label: {sentiment_label}, Emotional Labels: {emotion_labels}***")
    response_label = {}
    response_label[sentiment_label] = emotion_labels
    # Returning sentiment and emotion labels as text to display and dictionary for evaluation
    return nested_dict_to_text(response_label), response_label