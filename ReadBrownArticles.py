import nltk
# nltk.download('brown')
# nltk.download('nonbreaking_prefixes')
# nltk.download('perluniprops')

from nltk.corpus import brown
from mosestokenizer import *

mdetok = MosesDetokenizer()

with MosesDetokenizer('en') as detokenize:
    brown_natural = [detokenize(' '.join(sent).replace('``', '"').replace("''", '"').replace('`', "'").split())  for sent in brown.sents(categories = "news")]

with open("brownArticles.txt", "a") as file:
    for sent in brown_natural:
        file.write(sent)
        file.write("\n")
file.close()