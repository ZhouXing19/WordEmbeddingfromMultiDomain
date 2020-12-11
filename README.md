# Examining the Effect of Varying Domains on Vector-Space Representations 
Final Project for LING 28610 (Autumn 2020)

### Members

- Nancy Li 
- Deniz Türkçapar
- Zhou Xing

### Abstract
This paper aims to explore differences in word embeddings of vector-space models trained on varying domains. The domains analyzed in this study were literature and news. Due to more frequent use of figurative language in literature, as opposed to news, it was hypothesized that the model trained on literature was more likely to exhibit word embeddings that related more heavily to abstract meanings, while the model trained on news was more likely to exhibit word embeddings that related more heavily to concrete meanings. Overall, many of the words shared in both the models had vastly different similar words, suggesting that the meanings constructed by the two models were indeed different. Compared with the news model, the literature model more closely aligned with benchmark datasets meant to demonstrate human behavior. 

### Visualization

1. Please go to [tensorflow projector](http://projector.tensorflow.org/)
2. On the left panel, click on **Load** and upload the metadata and vectordata with the same model name from [tsv dir](/tsvs)

### Models

 - [EmmaModel](./EmmaModel.py)
 - [NewsModel](./NewsModel.py)
 - [MobyModel](./MobyModel.py)
 - [PersuasionModel](./PersuasionModel.py)

### Analysis
 - Top Similar Words for Model Pair: [here](./top_similar_words_edited.py)
 - Concreteness Analysis: [here](./top_similar_words_edited.py)
 - Similarity Comparison: [here](similarity_comparison_class.py)

