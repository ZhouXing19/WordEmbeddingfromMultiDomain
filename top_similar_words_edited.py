import NewsModel
import EmmaModel

def compare_top_similar(word):
	out_news = set()
	out_novel = set()

	preprocessed_news_similarities = NewsModel.model_news.wv.most_similar(positive=word, topn=10)
	preprocessed_lit_similarities = EmmaModel.model_emma.wv.most_similar(positive=word, topn=10)

	for i in preprocessed_news_similarities:
		out_news.add(i[0])

	for j in preprocessed_lit_similarities:
		out_novel.add(j[0])

	print("\n\n News Analysis: Words Most Similar to ", word, ":", out_news)
	print("\n\n Novel Analysis: Words Most Similar to ", word, ":", out_novel)


compare_top_similar("father")

vocabulary_emma = list(EmmaModel.wv.vocab.keys())
vocabulary_news = list(NewsModel.wv.vocab.keys())

vocabulary = [item for item in vocabulary_emma if item in vocabulary_news]

for word in vocabulary:
        compare_top_similar(word)
