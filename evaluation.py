import service
import tweets
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from bert_score import score
import re

"""
Function to evaluate summarization
@param: List of tweets and list of summaries
@return: F1 score
"""
def evaluate_summarization(input_data, summaries):
    precision, recall, f1 = score(input_data, summaries, lang='en', verbose=False)
    return f1.mean().item()

"""
Function to evaluate multi-label multi-class classification
@param: Predicted classes
@return: Hamming loss score
"""
def evaluate_classification(pred):
    # True labels
    y_true = [['Technology'], ['Health', 'Vaccine', 'Disease'], ['Entertainment', 'TV Shows', 'Thriller'], ['Entertainment'], ['Politics','Domestic']]
    # Binarize the labels
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(y_true)
    y_pred = mlb.transform(pred)
    # Calculate the hamming loss
    hamming_loss_score = hamming_loss(y_true, y_pred)
    return hamming_loss_score

"""
Function to evaluate sentiment and emotional analysis
@param: Predicted sentiment and emotional labels
@return: Hamming loss score
"""
def evaluate_emotional_analyzer(pred):
    # True labels
    y_true = [['Positive', 'happy', 'excited', 'satisfied'], ['Negative', 'sad'], ['Neutral', 'curious'], ['Positive', 'confident', 'excited'], ['Neutral', 'curious']]
    # Binarize the labels
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(y_true)
    y_pred = mlb.transform(pred)
    # Calculate the hamming loss
    hamming_loss_score = hamming_loss(y_true, y_pred)
    return hamming_loss_score

"""
Function to generate array of arrays from a tree for hamming loss evaluation
@param: Tree as a dictionary
@return: Array of arrays
"""
def generate_arr(tree): 
    arr = []
    # For every label in the tree, the key is appended
    # Further the list of values corresponding to each key is appended
    # Also key within each key and its values are added
    for key in tree:
        arr.append(key)
        if isinstance(tree[key], dict):
            for key1 in tree[key]:
                arr.append(key1)
                if len(tree[key][key1]) > 0:
                    arr.extend(tree[key][key1])
        else:
            if len(tree[key]) > 0:
                arr.extend(tree[key])
    return arr

"""
Function to remove special characters that would not contribute in analysis
@param: data containing special characters
@return: data after removing special characters
"""
def clean_data(data):
    # Regex to include only letters and digits
    new_data = re.sub('[^A-Za-z0-9 ]+', '', data)
    return new_data

"""
Function to run evaluation for all the models
"""
def run_evaluation():
    data = tweets.get_tweets_from_file()
    # Fetching all the search keywords from the file
    keywords = data['word'].unique()
    classify_pred = []
    sentiment_pred = []
    input_data = []
    summaries = []
    # Running evaluation for each keyword
    for keyword in keywords:
        # Picking tweets for each search keyword
        tweets_data = data[data['word'] == keyword]
        tweets_data = tweets_data['content'].to_numpy()
        tweets_data = [item.split(':')[1] for item in tweets_data if ':' in item]
        # Removing garbage characters
        tweets_data = [clean_data(item) for item in tweets_data]
        tweets_data = ' '.join([item for item in tweets_data])
        # Creating summary, classification tree and generating sentiment and emotional labels
        summary = service.summarize_bart(tweets_data)
        text,tree = service.multi_level_classify(tweets_data)
        text,etree = service.sentiment_emotional_analysis(tweets_data)
        # Converting the predicted dict to an array for measuring hamming loss 
        classify_pred.append(generate_arr(tree))
        sentiment_pred.append(generate_arr(etree))
        # Given data and summarized data for measuring F1 score
        input_data.append(tweets_data)
        summaries.append(summary)
    # Calculating F1 score and hamming loss
    score_summarize = evaluate_summarization(input_data, summaries)
    hloss_classify = evaluate_classification(classify_pred)
    hloss_sentiment = evaluate_emotional_analyzer(sentiment_pred)
    # Displaying all the results
    print("\n\n***********************************************\n\n")
    print("Analysis results: ")
    print("\n*************************************************\n")
    print("F1 score for summarization: ", str(score_summarize))
    print("Hamming loss for text classification: ", str(hloss_classify))
    print("Hamming loss for sentiment/emotional analysis: ", str(hloss_sentiment))
    print("\n*************************************************\n")

run_evaluation()
