from flask import Flask, request, render_template
import analysis

app = Flask(__name__)

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
    summary = analysis.summarize(keyword)
    return summary

@app.route("/getClassification")
def getClassification():
    keyword = request.args.get('keyword')
    response = analysis.classify(keyword)
    labels = response['labels']
    scores = response['scores']
    classified_labels = []
    for i in range(len(scores)):
        if scores[i] >= 40:
            classified_labels.append(labels[i])
    if len(classified_labels) == 0:
        max_val = max(scores)
        for i in range(len(scores)):
            if scores[i] >= max_val:
                classified_labels.append(labels[i])
    classified_labels = '\n'.join(classified_labels)
    return classified_labels

@app.route("/getSentimentAnalysis")
def getSentimentAnalysis():
    keyword = request.args.get('keyword')
    response = analysis.sentimentAnalysis(keyword)
    labels = response['labels']
    scores = response['scores']
    classified_label = ''
    if scores[0] > scores[1]:
        classified_label = labels[0]
    else:
        classified_label = labels[1]
    return classified_label

app.run()

