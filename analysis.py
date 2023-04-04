import tensorflow as tf
import transformers as ts
from bert_score import score
from summarizer import TransformerSummarizer
import tweets
import logging

logger = logging.getLogger("analysis")
logging.basicConfig(level=logging.DEBUG)
logger.setLevel('DEBUG')

"""
Function to initialize all the models
@return: all models namely bart, gpt, xlnet and pegasus
"""
def initialize_models():
    model_name = 'facebook/bart-large-cnn'
    cache_dir = 'cache/'
    tokenizer = ts.AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
    model = ts.TFAutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir = cache_dir)
    bart_summarizer = ts.pipeline("summarization", model=model, tokenizer=tokenizer)
    gpt_summarizer = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
    xlnet_summarizer = TransformerSummarizer(transformer_type="XLNet", transformer_model_key="xlnet-base-cased")
    pegasus_summarizer = ts.pipeline("summarization", model = "google/pegasus-cnn_dailymail", max_length=100)
    return bart_summarizer, gpt_summarizer, xlnet_summarizer, pegasus_summarizer

"""
Function to calculate accuracy of summary
@param: Text to be summarized, summary generated
@return: score 
"""
def get_score(input_text, summary):
    _,_,f1 = score([summary], [input_text], lang='en', verbose=False)
    return f1.item()

"""
Function to create summary using bart model
@param: bart model, text to be summarized
@return: summary generated
"""
def bart_analysis(bart_summarizer, input_text):
    summary = bart_summarizer(input_text, batch_size = 3, max_length = 100, min_length = 20, do_sample = False, num_beams = 4, max_time = 5, length_penalty = 4.0)[0]['summary_text']
    return summary

"""
Function to create summary using GPT2 model
@param: gpt model, text to be summarized
@return: summary generated
"""
def gpt_analysis(gpt_summarizer, input_text):
    summary = ''.join(gpt_summarizer(input_text, min_length=20, max_length=100))
    return summary

"""
Function to create summary using XLNet model
@param: xlnet model, text to be summarized
@return: summary generated
"""
def xlnet_analysis(xlnet_summarizer, input_text):
    summary = ''.join(xlnet_summarizer(input_text, min_length=20, max_length=100))
    return summary

"""
Function to create summary using Pegasus model
@param: pegasus model, text to be summarized
@return: summary generated
"""
def pegasus_analysis(pegasus_summarizer, input_text):
    pipe_out = pegasus_summarizer(input_text)
    return pipe_out[0]['summary_text']

"""
Function that performs analysis and displays accuracy of all the models being compared
"""
def analyze():
    input_text = """
        Researchers have discovered a new species of dinosaur in Argentina, which they believe is 
        the oldest-known member of the titanosaur group. The dinosaur lived 140 million years ago 
        and measured about 20 feet long. It has been named Ninjatitan zapatai in honor of Argentine 
        paleontologist Sebastian Apesteguia, who is also known as "The Ninja".
        """

    bart_summarizer, gpt_summarizer, xlnet_summarizer, pegasus_summarizer = initialize_models()
    print("Models created")
    bart_summary = bart_analysis(bart_summarizer, input_text)
    bart_score = get_score(input_text, bart_summary)
    print("**************")
    print("Bart Summary: " + bart_summary)
    print("Bart Score: " + str(bart_score * 100))

    print("**************")
    gpt_summary = gpt_analysis(gpt_summarizer, input_text)
    gpt_score = get_score(input_text, gpt_summary)
    print("GPT Summary: " + gpt_summary)
    print("GPT Score: " + str(gpt_score * 100))

    print("**************")
    xlnet_summary = xlnet_analysis(xlnet_summarizer, input_text)
    xlnet_score = get_score(input_text, xlnet_summary)
    print("XLNet Summary: " + xlnet_summary)
    print("XLNet Score: " + str(xlnet_score * 100))

    print("**************")
    pegasus_summary = pegasus_analysis(pegasus_summarizer, input_text).replace(".<n>",". ")
    pegasus_score = get_score(input_text, pegasus_summary)
    print("Pegasus summary: " + pegasus_summary)
    print("Pegasus score: " + str(pegasus_score * 100))

"""
Function used by flask app to generate summary using pegasus model for tweets based on keyword
@param: keyword entered by user
@return: summary generated
"""
def summarize(keyword):
    logging.debug("***Starting to create model and summarize***")
    # Model to perform summarization
    pegasus_summarizer = ts.pipeline("summarization", model = "google/pegasus-cnn_dailymail", max_length = 100)
    # Calling function to fetch tweets
    tweets_data = tweets.get_tweets(keyword) 
    logging.debug("***Fetching tweets***")
    # Summarizing tweets fetched and returning summary
    pipe_out = pegasus_summarizer(tweets_data)
    summary = pipe_out[0]['summary_text'].replace(".<n>",". ")
    logging.info("***Completed generating summary and returning the summary***")
    return summary