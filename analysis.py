import tensorflow as tf
import transformers as ts

def bart_analysis():
    model_name = 'facebook/bart-large-cnn'
    cache_dir = 'cache/'
    tokenizer = ts.AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
    model = ts.TFAutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir = cache_dir)
    input_text = """
    Researchers have discovered a new species of dinosaur in Argentina, which they believe is 
    the oldest-known member of the titanosaur group. The dinosaur lived 140 million years ago 
    and measured about 20 feet long. It has been named Ninjatitan zapatai in honor of Argentine 
    paleontologist Sebastian Apesteguia, who is also known as "The Ninja".
    """
    summarizer = ts.pipeline("summarization", model=model, tokenizer=tokenizer)
    summary = summarizer(input_text, batch_size = 3, max_length = 100, min_length = 20, do_sample = False, num_beams = 4, max_time = 5, length_penalty = 4.0)[0]['summary_text']
    return summary

print(bart_analysis())