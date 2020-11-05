# help from https://www.kaggle.com/
# venugopalvasarla/using-word2vec-and-glove-for-word-embeddings/notebook

import nltk
import gensim

# nltk.download('gutenberg')
from nltk.corpus import gutenberg

import re


# importing relevant text from nltk database

print(gutenberg.fileids())
emma_sents = gutenberg.raw(gutenberg.fileids()[0])
emma_sents = emma_sents.split('\n')
print("The length of the sentences before cleaning is: ", len(emma_sents))
emma = nltk.corpus.gutenberg.words('austen-emma.txt')

# libraries necessary to clean the data

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stopWords = stopwords.words("english")
charfilter = re.compile("[a-zA-Z]+")


#tokenize the words
def simple_filter(Sent):
    
    #converting all tokens to lower case:
    words = sent.split()
    word_lower = []
    for word in words:
        word_lower.append(word.lower())

    #removing all stop words:
    word_clean = [word for word in word_lower if word not in stopWords]

    #removing all the characters and using only characters
    tokens = list(filter(lambda token : charfilter.match(token),word_clean))

    #stemming all words
    ntokens = []
    for word in tokens:
        ntokens.append(PorterStemmer().stem(word))
    return tokens

# converting emma data to tokens using the above function

sentences = []
for sent in emma_sents:
    tokens = simple_filter(sent)
    if len(tokens) > 0:
        sentences.append(tokens)

# Word2Vec
# Training gensim on emma data

from gensim.models import Word2Vec
model_emma = Word2Vec(sentences, min_count = 1, size = 50, workers = 3, window = 5, sg = 0)
wv = model_emma.wv

        
