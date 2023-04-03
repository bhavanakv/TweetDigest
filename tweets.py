import snscrape.modules.twitter as sntwitter
import re

def clean_data(data):
    new_data = re.sub('[^A-Za-z0-9 ]+', '', data)
    return new_data

def get_tweets(query):
    query = query + " lang:en"
    tweets = []
    limit = 10
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if limit == 0:
            break
        else:
            tweets.append(clean_data(tweet.rawContent))
            limit = limit -1
    print("Received: {}".format(len(tweets)))
    tweets = ' '.join([item for item in tweets])
    print(tweets)
    return tweets