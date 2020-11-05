# help from https://www.kaggle.com/
# venugopalvasarla/using-word2vec-and-glove-for-word-embeddings/notebook

import nltk
import gensim

# nltk.download('brown')
from nltk.corpus import brown

import re
import string
import io
string.punctuation = string.punctuation + " "


# importing relevant text from nltk database

news_sents = brown.sents(categories = "news")


# libraries necessary to clean the data

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stopWords = stopwords.words("english")
charfilter = re.compile("[a-zA-Z]+")


#tokenize the words
def simple_filter(Sent):
    
    #converting all tokens to lower case:
    for word in sent:
        if word in string.punctuation:
            sent.remove(word)
                    
    word_lower = []
    for word in sent:
        word_lower.append(word.lower())

    #removing all stop words:
    word_clean = [word for word in word_lower if word not in stopWords]

    #removing all the characters and using only characters
    tokens = list(filter(lambda token : charfilter.match(token),word_clean))

    #stemming all words
    ntokens = []
    for word in tokens:
        ntokens.append(PorterStemmer().stem(word))
 
    for position, word in enumerate(tokens):
        for char in word:
            if char in string.punctuation:
                new_word = word.replace(char, "")
                tokens[position] = new_word
    return tokens

# converting emma data to tokens using the above function

sentences = []
for sent in news_sents:
    tokens = simple_filter(sent)
    if len(tokens) > 0:
        sentences.append(tokens)

# Word2Vec
# Training gensim on emma data

from gensim.models import Word2Vec
model_news = Word2Vec(sentences, min_count = 1, size = 50, workers = 3, window = 5, sg = 0)
wv = model_news.wv

def saveWordVecPairInFile(wv):
      out_v = io.open('vectorsNews.tsv', 'w', encoding='utf-8')
      out_m = io.open('metadataNews.tsv', 'w', encoding='utf-8')

      vocabulary = list(wv.vocab.keys())


      for index, word in enumerate(vocabulary):
         vec = wv[word]
         out_v.write('\t'.join([str(x) for x in vec]) + "\n")
         out_m.write(word + "\n")
      out_v.close()
      out_m.close()

saveWordVecPairInFile(wv)
