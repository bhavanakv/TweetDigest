import snscrape.modules.twitter as sntwitter
import re
import logging

logger = logging.getLogger("tweets")
logging.basicConfig(level=logging.INFO)
logger.setLevel('INFO')

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
Function to fetch the tweets 
@param: query as entered by user
@return: list of tweets 
"""
def get_tweets(query):
    # Appending language of tweets to be picked
    query = query + " lang:en"
    logger.info("***Query: {}***".format(query))
    tweets = []
    limit = 10
    # Loop to fetch tweets until limit is reached
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if limit == 0:
            break
        else:
            # Cleaning data and appending to list of tweets
            tweets.append(clean_data(tweet.rawContent))
            limit = limit -1
    logger.info("***Received: {} tweets***".format(len(tweets)))
    # Converting list of tweets to string separated by space
    tweets = ' '.join([item for item in tweets])
    logger.info("***Tweets: {}***".format(tweets))
    return tweets