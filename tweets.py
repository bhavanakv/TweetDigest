import snscrape.modules.twitter as sntwitter
import re
import logging

logger = logging.getLogger("tweets")
logging.basicConfig(level=logging.INFO)
logger.setLevel('INFO')

def clean_data(data):
    new_data = re.sub('[^A-Za-z0-9 ]+', '', data)
    return new_data

def get_tweets(query):
    query = query + " lang:en"
    logger.info("***Query: {}***".format(query))
    tweets = []
    limit = 10
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if limit == 0:
            break
        else:
            tweets.append(clean_data(tweet.rawContent))
            limit = limit -1
    logger.info("***Received: {} tweets***".format(len(tweets)))
    tweets = ' '.join([item for item in tweets])
    logger.info("***Tweets: {}***".format(tweets))
    return tweets