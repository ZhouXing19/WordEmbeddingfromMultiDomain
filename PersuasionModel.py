'''
References:
   - nltk corpus : https://www.nltk.org/book/ch02.html
   - the meaning of the categories in the corpus reuters of NLTK: https://stackoverflow.com/questions/25134160/whats-the-meaning-of-the-categories-in-the-corpus-reuters-of-nltk
   - venugopalvasarla/using-word2vec-and-glove-for-word-embeddings/notebook
   - https://www.kaggle.com/
'''

import nltk
import gensim
import io

# nltk.download('gutenberg')
from nltk.corpus import gutenberg

import re
import string
string.punctuation = string.punctuation + " "


# importing relevant text from nltk database

print(gutenberg.fileids())
persuasion_sents = gutenberg.raw(gutenberg.fileids()[1])
persuasion_sents = persuasion_sents.split('\n')
print("The length of the sentences before cleaning is: ", len(persuasion_sents))
persuasion = nltk.corpus.gutenberg.words('austen-persuasion.txt')

# libraries necessary to clean the data

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stopWords = stopwords.words("english")
charfilter = re.compile("[a-zA-Z]+")


#tokenize the words
def simple_filter(Sent):
    
    #converting all tokens to lower case:
    words = re.split("[" + string.punctuation + "]+", sent)
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
for sent in persuasion_sents:
    tokens = simple_filter(sent)
    if len(tokens) > 0:
        sentences.append(tokens)

# Word2Vec
# Training gensim on emma data

from gensim.models import Word2Vec
model_persuasion = Word2Vec(sentences, min_count = 1, size = 50, workers = 3, window = 5, sg = 0)
wv = model_persuasion.wv


def saveWordVecPairInFile(wv):
      out_v = io.open('vectorsPersuasion.tsv', 'w', encoding='utf-8')
      out_m = io.open('metadataPersuasion.tsv', 'w', encoding='utf-8')

      vocabulary = list(wv.vocab.keys())


      for index, word in enumerate(vocabulary):
         vec = wv[word]
         out_v.write('\t'.join([str(x) for x in vec]) + "\n")
         out_m.write(word + "\n")
      out_v.close()
      out_m.close()

saveWordVecPairInFile(wv)


