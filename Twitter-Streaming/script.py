"""
author: sominwadhwa
date: 2 Feb 23:11
"""

import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import time
import json
import string
import re
from collections import Counter
from collections import defaultdict
import operator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import casual, casual_tokenize
from nltk import bigrams

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import matplotlib
sns.set_style('darkgrid')
matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('figure', titlesize=20)

consumer_key = '<EnterConsumerKey>'
consumer_secret = '<EnterConsumerSecret>'
access_token = '<EnterAccessToken>'
access_secret = '<AccessSecret>'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_secret)
api = tweepy.API(auth)

#Reading Own Timeline
for status in tweepy.Cursor(api.home_timeline).items(2):
    print(status.text)

#Reading own timeline-- JSON
for status in tweepy.Cursor(api.home_timeline).items(1):
    print(status._json)

class MyListener(StreamListener):
    def __init__(self, time_limit=60):
        self.start_time = time.time()
        self.limit = time_limit
        self.saveFile = open('collection.json', 'a')
        super(MyListener, self).__init__()

    def on_data(self, data):
        if (time.time() - self.start_time) < self.limit:
            self.saveFile.write(data)
            return True
        else:
            self.saveFile.close()
            return False
twitter_stream = Stream(auth, MyListener(time_limit = 20))
twitter_stream.filter(track=['Data'])

#Body of a tweet-- Preprocessing
with open('collection.json','r') as f:
    line = f.readline()
    tweet = json.loads(line)
    print (json.dumps(tweet, indent=3))

tweet = 'RT @marcobonzanini: just an example! :D http://example.com #NLP'
tk = TweetTokenizer()
tk.tokenize(tweet)

st = ''
with open('collection.json','r') as f:
    for line in f:
        tweet = json.loads(line)
        print (tk.tokenize(tweet['text']))
        st += ' '.join(tk.tokenize(tweet['text']))

""" Since the above results are not interesting we need to remove stopwords: """
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['RT','They','Be', 'via', '…', 'I','The']

with open('collection.json', 'r') as f:
    count_all = Counter()
    for line in f:
        tweet = json.loads(line)
        if "text" in tweet:
            terms = [term for term in tk.tokenize(tweet['text']) if term not in stop and not term.startswith('http') and not term.startswith('@')]
            count_all.update(terms)
    print (count_all.most_common(20))

with open('collection.json', 'r') as f:
    count_all = Counter()
    for line in f:
        tweet = json.loads(line)
        if "text" in tweet:
            terms = [term for term in tk.tokenize(tweet['text']) if term.startswith('#')]
            count_all.update(terms)
    print (count_all.most_common(10))

wordcloud = WordCloud(max_font_size=40, width = 500, height = 100, stopwords=stop,
                     ).generate(st)
plt.figure(figsize=(17,27))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

print (count_all.most_common(5))
word_freq = count_all.most_common(20)
labels, freq = zip(*word_freq)
height = np.arange(len(labels))
word_freq

plt.figure(figsize=(30,10))
plt.bar(height, freq)
plt.xticks(height, labels)
plt.ylabel("Occurences")
plt.xlabel("Word/Text")
plt.show()

#Bigrams -- Terms adjacent to each other that occur frequently
with open('collection.json', 'r') as f:
    count_all = Counter()
    for line in f:
        tweet = json.loads(line)
        if "text" in tweet:
            terms = [term for term in tk.tokenize(tweet['text']) if term not in stop and not term.startswith('http') and not term.startswith('@')]
            term_pairs = bigrams(terms)
            count_all.update(term_pairs)
    print (count_all.most_common(20))

#Co-Occurences (Within Tweets)
com = defaultdict(lambda: defaultdict(int))
with open('collection.json', 'r') as f:
    count_all = Counter()
    for line in f:
        tweet = json.loads(line)
        if "text" in tweet:
            terms = [term for term in tk.tokenize(tweet['text']) if term not in stop and not term.startswith('http') and not term.startswith('@')]
            for i in range(len(terms)-1):
                for j in range (i+1, len(terms)):
                    t1, t2 = sorted([terms[i],terms[j]])
                    if t1 != t2:
                        com[t1][t2] += 1

#Extract 5 most common co-occurences
com_max = []
for t1 in com:
    t1_max_term = sorted(com[t1].items(), key = operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_term:
        com_max.append(((t1,t2), t2_count))
terms_max = sorted(com_max, key = operator.itemgetter(1), reverse= True)
print (terms_max[:5])

#Co-Occurences with external word
search_word = 'machine'  #ENTER SEARCH WORD HERE
with open('collection.json', 'r') as f:
    count_all = Counter()
    for line in f:
        tweet = json.loads(line)
        terms = [term for term in tk.tokenize(tweet['text']) if term not in stop and not term.startswith('http') and not term.startswith('@')]
        if search_word in terms:
            count_all.update(terms)
    print (count_all.most_common(5))

"""
Sentiment Analysis: Extracting subjective information by means of text mining.

Collect -> Preprocess/Tokenize -> Basic Visualization -> Analyse -> Future Course of Action

Limitations of PMI based Approach: The semantic scores are calculated on terms, meaning that there is no notion of 
“entity” or “concept” or “event”. For example, it would be nice to aggregate and normalise all the references to the 
team names, e.g. #ita, Italy and Italia should all contribute to the semantic orientation of the same entity. 
Some aspects of natural language are also not captured by this approach, more notably modifiers and negation: 
how do we deal with phrases like not bad (this is the opposite of just bad) or very good (this is stronger than 
just good)?

"""
number_of_tweets_collected = 618 #From collection.json

p_term = {}
p_com_terms = defaultdict(lambda : defaultdict(int))

for term, n in count_all.items():
    p_term[term] = n/number_of_tweets_collected
    for t2 in com[term]:
        p_com_terms[term][t2] = com[term][t2]/number_of_tweets_collected

positive_vocab = [
    'good', 'nice', 'Great', 'awesome', 'outstanding','fair'
    'fantastic', 'terrific', ':)', ':-)', 'like', 'love' 
]
negative_vocab = [
    'bad', 'terrible', 'crap', 'useless', 'hate', ':(', ':-('
]
PMI = defaultdict(lambda: defaultdict(int))

for t1 in p_term:
    for t2 in com[t2]:
        deno = p_term[t1] * p_term[t2]
        p_fin = p_com_terms[t1][t2]/deno
        PMI[t1][t2] = math.log2(p_fin if p_fin>0 else 1)

SO = {} #Semantic Orientation
for t1 in p_term.items():
    pos = sum(PMI[t1][tp] for tp in positive_vocab)
    neg = sum(PMI[t1][tn] for tn in negative_vocab)
    SO[t1] = pos - neg
semantic_sorted = sorted(SO, key=operator.itemgetter(1), reverse=True)
top_pos = semantic_sorted[:5]
top_neg = semantic_sorted[-5:]

print(top_pos)

