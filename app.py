from flask import Flask, request, render_template
import analysis

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize")
def summarize():
    return render_template("summarize.html")

@app.route("/getSummary")
def test():
    keyword = request.args.get('keyword')
    summary = analysis.summarize(keyword)
    return summary

app.run()

