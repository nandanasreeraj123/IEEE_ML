import tweepy
from tweepy import API
from tweepy import Cursor
from tweepy import  Stream
from tweepy.streaming import StreamListener
import pandas as pd
from textblob import TextBlob
import re
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# following the above mentioned procedure.
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""

class Client():
    def __init__(self,twitter_user=None):
        self.auth = Authetication().authenticate()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user
    def get_twitter_client(self):
        return  self.twitter_client
    def get_tweets(self, num):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id = self.twitter_user).items(num):
            tweets.append(tweet)
        return tweets

class Authetication():
    def authenticate(self):
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        return auth

class TwitterStreamer():
    def __init__(self):
        self.twitter_authenticator = Authetication()

    def stream_tweets(self, f):
        lit = twitter_authen(f)

        auth = self.twitter_authenticator.authenticate()
        stream = Stream(auth, lit)
        stream.filter(track=['obama', 'donald trump'])


class twitter_authen(StreamListener):
    def __init__(self, f):
        self.f = f
    def on_data(self, data):
        print(data)
        with open(self.f, 'a') as fl:
            fl.write(data)
        return True
    def on_error(self, status_code):
        if status_code == 420:
            return False
        print(status_code)

class analysing():
    def cleaning(self,tweets):
        return ' '.join(re.sub('[^a-zA-Z]', ' ',tweets).split())

    def sentiments(self, tweet):
        analysis = TextBlob(self.cleaning(tweet))
        if analysis.sentiment.polarity >=0:
            return 1
        else:
            return 0
    def framing(self, tweets):
        df = pd.DataFrame(data = [tweet.text for tweet in tweets], columns=["tweets"])
        return df
if __name__ == "__main__":
    f = "tweets.json"
    client_twitter = Client()
    analysers = analysing()
    api = client_twitter.get_twitter_client()
    tweets = api.user_timeline(screen_name = "realDonaldTrump", count =30)
    df = analysers.framing(tweets)

    df['sentiment'] = np.array([analysers.sentiments(tweet) for tweet in df['tweets']])
    print(df.head(10))
    X = df['tweets']
    from sklearn.feature_extraction.text import CountVectorizer
    corpus = []
    ps = PorterStemmer()
    review = [ps.stem(word) for word in X if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(X).toarray()
    y = df.iloc[:, 1].values

    # Splitting the dataset into the Training set and Test set


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print(y_pred)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    print(cm)