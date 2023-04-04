from flask import Flask, request, render_template
import analysis
import logging

app = Flask(__name__)

logger = logging.getLogger("app")
logging.basicConfig(level=logging.DEBUG)
logger.setLevel('DEBUG')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize")
def summarize():
    return render_template("summarize.html")

@app.route('/classify')
def classify():
    return render_template("classify.html")

@app.route('/analyze')
def analyze():
    return render_template("sentimentAnalysis.html")

@app.route("/getSummary")
def getSummary():
    keyword = request.args.get('keyword')
    logger.debug("***Received request to summarize for keyword: {}***".format(keyword))
    summary = analysis.summarize_bart(keyword)
    logger.debug(f"***Summary for keywword: {keyword} -> {summary}***")
    return summary

@app.route("/getClassification")
def getClassification():
    keyword = request.args.get('keyword')
    logger.debug(f"***Received request to classify for keyword: {keyword}***")
    response = analysis.classify(keyword)
    logger.info(f"***Response from classifier: {response}***")
    labels = response['labels']
    scores = [score * 100 for score in response['scores']]
    classified_labels = []
    for i in range(len(scores)):
        if scores[i] >= 40:
            classified_labels.append(labels[i])
    if len(classified_labels) == 0:
        max_val = max(scores)
        for i in range(len(scores)):
            if scores[i] >= max_val or abs(max_val - scores[i]) <= 3:
                classified_labels.append(labels[i])
    classified_labels = '\n'.join(classified_labels)
    logger.info(f"***Classified labels: {classified_labels}***")
    return classified_labels

@app.route("/getSentimentAnalysis")
def getSentimentAnalysis():
    keyword = request.args.get('keyword')
    logger.debug(f"***Received request to analyze for keyword: {keyword}***")
    response = analysis.sentimentAnalysis(keyword)
    labels = response['labels']
    scores = response['scores']
    logger.info(f"***Response from classifier: {response}***")
    classified_label = ''
    if scores[0] > scores[1]:
        classified_label = labels[0]
    else:
        classified_label = labels[1]
    logger.info(f"***Classified label: {classified_label}***")
    return classified_label

app.run()

