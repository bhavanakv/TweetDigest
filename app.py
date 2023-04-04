from flask import Flask, request, render_template
import service
import logging

app = Flask(__name__)

logger = logging.getLogger("app")
logging.basicConfig(level=logging.DEBUG)
logger.setLevel('DEBUG')

"""
API to render home page
@return: HTML page
"""
@app.route("/")
def index():
    return render_template("index.html")

"""
API to render tweet summarization page
@return: HTML page
"""
@app.route("/summarize")
def summarize():
    return render_template("summarize.html")

"""
API to render tweet classification page
@return: HTML page
"""
@app.route('/classify')
def classify():
    return render_template("classify.html")

"""
API to render tweet sentiment analysis page
@return: HTML page
"""
@app.route('/analyze')
def analyze():
    return render_template("sentimentAnalysis.html")

"""
API to receive keyword from webpage and call the function that summarizes it
@return: summary of the tweet based on the keyword
"""
@app.route("/getSummary")
def getSummary():
    # Fetch the keyword entered by user
    keyword = request.args.get('keyword')
    logger.debug("***Received request to summarize for keyword: {}***".format(keyword))
    # Calling function to summarize
    summary = service.summarize_bart(keyword)
    logger.debug(f"***Summary for keywword: {keyword} -> {summary}***")
    return summary

"""
API to receive keyword from webpage and call the function that classifies it
@return: classified label for tweets based on the keyword
"""
@app.route("/getClassification")
def getClassification():
    # Fetch the keyword entered by user
    keyword = request.args.get('keyword')
    logger.debug(f"***Received request to classify for keyword: {keyword}***")
    # Calling function to classify that returns classified labels and their corresponding scores
    response = service.classify(keyword)
    logger.info(f"***Response from classifier: {response}***")
    labels = response['labels']
    # Converting the score to scale of 100
    scores = [score * 100 for score in response['scores']]
    classified_labels = []
    # Selecting all the labels with score more than 40
    for i in range(len(scores)):
        if scores[i] >= 40:
            classified_labels.append(labels[i])
    # If the scores are less than 40, then top labels are picked with very less difference of scores among them
    if len(classified_labels) == 0:
        max_val = max(scores)
        for i in range(len(scores)):
            if scores[i] >= max_val or abs(max_val - scores[i]) <= 3:
                classified_labels.append(labels[i])
    # Converting array to string separated by newline character
    classified_labels = '\n'.join(classified_labels)
    logger.info(f"***Classified labels: {classified_labels}***")
    return classified_labels

"""
API to receive keyword from webpage and call the function that performs sentiment analysis on it
@return: 'positive' or 'negative' depending on the sentiment analysis performed on the keyword
"""
@app.route("/getSentimentAnalysis")
def getSentimentAnalysis():
    # Fetch the keyword entered by user
    keyword = request.args.get('keyword')
    logger.debug(f"***Received request to analyze for keyword: {keyword}***")
    # Calling function to classify that returns classified labels and their corresponding scores
    response = service.sentimentAnalysis(keyword)
    labels = response['labels']
    scores = response['scores']
    logger.info(f"***Response from classifier: {response}***")
    classified_label = ''
    # Selecting the label with high score
    if scores[0] > scores[1]:
        classified_label = labels[0]
    else:
        classified_label = labels[1]
    logger.info(f"***Classified label: {classified_label}***")
    return classified_label

app.run()

