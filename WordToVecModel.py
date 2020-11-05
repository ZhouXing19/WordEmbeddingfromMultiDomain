'''
References:
   - nltk corpus : https://www.nltk.org/book/ch02.html
   - the meaning of the categories in the corpus reuters of NLTK: https://stackoverflow.com/questions/25134160/whats-the-meaning-of-the-categories-in-the-corpus-reuters-of-nltk
   - venugopalvasarla/using-word2vec-and-glove-for-word-embeddings/notebook
   - https://www.kaggle.com/
'''

import nltk
import gensim
#nltk.download('brown')
#nltk.download('gutenberg')
from nltk.corpus import brown, gutenberg, stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models import Word2Vec
import re
import io

stopWords = stopwords.words("english")
charfilter = re.compile("[a-zA-Z]+")

class WordToVecModel:

   '''
   @:param {string} category: news / novel / ...
   '''

   def __init__(self, category):
      self.category = category

   def getSentences(self):
      if self.category == "novel":
         sentences = gutenberg.raw(gutenberg.fileids()[0])
         sentences = sentences.split('\n')

      elif self.category == "news":
         sentences = brown.sents(categories='news')

      return sentences

   def cleanSingleSentence(self, sent):
      words = sent.split()

      # converting all tokens to lower case:
      wordLower = [word.lower() for word in words]

      # removing all stop words:
      wordClean = [word for word in wordLower if word not in stopWords]

      return wordClean

   def tokenizeWordsInSingleSentence(self, sent):

      # removing all the characters and using only characters
      tokens = list(filter(lambda token: charfilter.match(token), sent))

      # stemming all words
      ntokens = []
      for word in tokens:
         ntokens.append(PorterStemmer().stem(word))

      return tokens

   def tokenizeWordsInAllSentences(self, sentences):
      tokenizedSentences = []
      for sent in sentences:
         tokens = self.tokenizeWordsInSingleSentence(sent)
         if len(tokens) > 0:
            tokenizedSentences.append(tokens)
      return tokenizedSentences

   def getVectorsForWordsFromAllSentences(self, min_count = 1, size = 50, workers = 3, window = 5, sg = 0):
      sentences = self.getSentences()
      sentences = [self.cleanSingleSentence(sent) for sent in sentences]
      tokenizedSentences = self.tokenizeWordsInAllSentences(sentences)

      model = Word2Vec(tokenizedSentences, min_count = min_count, size = size, workers = workers, window = window, sg = sg)
      Word2VecKeyedVectors = model.wv
      return Word2VecKeyedVectors

   def saveWordVecPairInFile(self, wv):
      out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
      out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

      vocabulary = list(wv.vocab.keys())


      for index, word in enumerate(vocabulary):
         vec = wv[word]
         out_v.write('\t'.join([str(x) for x in vec]) + "\n")
         out_m.write(word + "\n")
      out_v.close()
      out_m.close()


if __name__ == "__main__":
   w2vModel = WordToVecModel("novel")
   wv = w2vModel.getVectorsForWordsFromAllSentences()
   print("word: {} \n"
         "vector: {}".format("emma", wv["emma"]))
   w2vModel.saveWordVecPairInFile(wv)

